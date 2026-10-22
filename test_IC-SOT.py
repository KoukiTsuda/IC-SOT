import pandas as pd
from datasets import load_dataset
import json   


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


BASE_MODEL = "openai/whisper-medium"  # Whisper のモデル名
TRAINED_MODEL  = "/home/tsuda/M1_research/IC-SOT/IC-SOT-weight/checkpoint-32130"  # 学習済みモデルのフォルダ
#TEST_CSV = "/home/tsuda/M1_research/IC-SOT/make_data_IC/out/2spk_timestamp"  # テストCSVのあるフォルダ
TEST_CSV = "/home/tsuda/M1_research/IC-SOT/make_data_名市大/out/2spk_timestamp"  # テストCSVのあるフォルダ
EVAL_TMP = "/home/tsuda/M1_research/IC-SOT/eval_tmp"  # 一時ファイルの置き場
NORMALIZE_DIR = "/home/tsuda/M1_research/IC-SOT/make_data_IC" # テキスト正規化のファイル、句読点削除に使用

ts_dataset = load_dataset(
    "csv",
    data_files={
        #"test":  "/home/tsuda/M1_research/IC-SOT/make_data_IC/out/2spk_timestamp/test.csv",
        "test":  f"{TEST_CSV}/test.csv",
    },
    keep_default_na=False,
)


#学習に不要なカラムを削除する
ts_dataset = ts_dataset.remove_columns(["conversationID", "groupID"])

#30秒を超える音声は学習に使わないようにする(Whisper は 30 秒以降を切り捨てるため)
#startTime / endTime は map の remove_columns で消えるので、map より前に行う
ts_dataset = ts_dataset.filter(lambda b: b["endTime"] - b["startTime"] <= 30.0)


#デバッグ用
import os
DEBUG = os.environ.get("DEBUG", "0") == "1"

if DEBUG:
    ts_dataset["test"]  = ts_dataset["test"].select(range(8))

#プロセッサーの作成（音声→音響特徴量、テキスト→ベクトルに変換する前処理をしてくれる）
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language="ja", task="transcribe")
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="ja", task="transcribe")


from scipy.signal import resample_poly

def load_segment(path, start, end, target_sr=16000):
    sr = sf.info(path).samplerate
    audio, _ = sf.read(path, start=int(start * sr), stop=int(end * sr))
    if audio.ndim == 2:          # 2ch は L/R がほぼ同じなので平均してモノラルにする
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
    return audio


import numpy as np
import soundfile as sf

# 適当な使わない言語トークンをSCトークンとして流用
SC_ID = tokenizer.convert_tokens_to_ids("<|bo|>")  # チベット語

# デコーダ入力時の特殊トークン
SOT_ID         = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
LANG_ID        = tokenizer.convert_tokens_to_ids("<|ja|>")
TASK_ID        = tokenizer.convert_tokens_to_ids("<|transcribe|>")

# タイムスタンプトークンを得る
def get_timestamp_token_id(seconds: float) -> int:
    rounded = round(seconds / 0.02) * 0.02
    rounded = max(0.0, min(30.0, rounded))
    return tokenizer.convert_tokens_to_ids(f"<|{rounded:.2f}|>")

#生データをモデルが扱えるようにする関数
def prepare_dataset(batch):
    # CH1・CH2それぞれの音声を読み込む
    # load_segment の中で 16kHz モノラルに揃えているので、戻り値は音声の配列だけ
    ch1_audio = load_segment(batch["IC1_audio_path"], batch["startTime"], batch["endTime"])
    ch2_audio = load_segment(batch["IC2_audio_path"], batch["startTime"], batch["endTime"])

    # 各チャンネルのlog-melを計算(80次元 x 3000フレーム、30秒分に自動padding)
    ch1_mel = feature_extractor(ch1_audio, sampling_rate=16000).input_features[0]  # (80, 3000)
    ch2_mel = feature_extractor(ch2_audio, sampling_rate=16000).input_features[0]  # (80, 3000)

    # CH1とCH2を連結して160次元にする(あなたの研究のConv1d(160→80)融合層に入れる前提の形) #メモリが足りないのでastype(np.float16) 追加した
    batch["input_features"] = np.concatenate([ch1_mel, ch2_mel], axis=0).astype(np.float16)  # (160, 3000)

    # 正解テキスト: 話者1・話者2のテキストファイルを読み込む
    segs1 = json.loads(batch["IC1_segments"])
    segs2 = json.loads(batch["IC2_segments"])
    
    labels = [SOT_ID, LANG_ID, TASK_ID]
    for seg in segs1:
        ids = tokenizer(seg["text"], add_special_tokens=False).input_ids
        labels += [get_timestamp_token_id(seg["start"])] + ids + [get_timestamp_token_id(seg["end"])]
    labels += [SC_ID]   # ← 話者切り替わりの境界に1個だけ
    for seg in segs2:
        ids = tokenizer(seg["text"], add_special_tokens=False).input_ids
        labels += [get_timestamp_token_id(seg["start"])] + ids + [get_timestamp_token_id(seg["end"])]
    labels += [tokenizer.eos_token_id]
    
    batch["labels"] = labels
            

    return batch


# map で消える前の元データを取っておく（推論結果の可視化に必要）
test_raw = ts_dataset["test"]   

# float16で特徴量を保存し、map中のメモリ使用量を抑える
#ts_dataset = ts_dataset.map(prepare_dataset, remove_columns=ts_dataset.column_names["train"], num_proc=1, writer_batch_size=1)
ts_dataset["test"] = ts_dataset["test"].map(
    prepare_dataset,
    remove_columns=ts_dataset["test"].column_names,
    num_proc=1,
    writer_batch_size=1,
)


#モデルの2ch入力対応　
#whisperの音声認識モデルのクラス（WhisperForConditionalGeneration）を継承して実現してる
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")だった
#継承してるんだからWhisperForConditionalGenerationのメソッド理解する必要あるかも
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration
from transformers import GenerationMixin

class TwoChannelWhisper(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.fusion = torch.nn.Conv1d(160, 80, kernel_size=1)  # ★追加した層はこれだけ

    def forward(self, input_features=None, **kwargs):
        if input_features is not None:
            input_features = self.fusion(input_features)
        return super().forward(input_features=input_features, **kwargs)
    
    #whisperのタイムスタンプ制約を外すため改造した
    def generate(self, input_features=None, **kwargs):
        kwargs.pop("labels", None)
        input_features = self.fusion(input_features)
        # 学習と同じプロンプト（notimestamps なし）を自分で渡す
        prompt = torch.tensor([[SOT_ID, LANG_ID, TASK_ID]], device=input_features.device)
        prompt = prompt.repeat(input_features.size(0), 1)
        # Whisper専用の generate（notimestamps の付与・タイムスタンプ制約）を通さない
        return GenerationMixin.generate(
            self, input_features=input_features, decoder_input_ids=prompt, **kwargs
        )


model = TwoChannelWhisper.from_pretrained(TRAINED_MODEL)
#デコーダ入力時の特殊トークン
model.generation_config.language = "ja"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


#データコレーターの作成（データセット→ミニバッチ生成を担当）

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


#評価指標
import evaluate
import sys
import sys
sys.path.append(NORMALIZE_DIR)
from cejc_normalize import normalize

metric = evaluate.load("cer")

TS_BEGIN = tokenizer.convert_tokens_to_ids("<|0.00|>")

def compute_metrics(pred):
    pred_ids  = pred.predictions.copy()
    label_ids = pred.label_ids.copy()
    
    for a in (pred_ids, label_ids):
        a[a == -100] = tokenizer.pad_token_id
        a[a >= TS_BEGIN] = tokenizer.pad_token_id
        
    pred_str  = [normalize(s) for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize(s) for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    
    return {"cer": 100 * metric.compute(predictions=pred_str, references=label_str)}

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

eval_args = Seq2SeqTrainingArguments(
    output_dir=EVAL_TMP,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=300,
    fp16=True,
    remove_unused_columns=False,
    report_to=[],
)

trainer = Seq2SeqTrainer(
    args=eval_args,
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

results = trainer.predict(ts_dataset["test"], metric_key_prefix="test")
print(results.metrics)   # test_cer が出る



# ===== 推論結果と正解を並べてCSVに保存 =====
import numpy as np
import pandas as pd

pred_ids  = np.where(results.predictions == -100, tokenizer.pad_token_id, results.predictions)
label_ids = np.where(results.label_ids   == -100, tokenizer.pad_token_id, results.label_ids)

def to_text(ids):
    s = tokenizer.decode(ids, skip_special_tokens=False)
    for t in ["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|endoftext|>"]:
        s = s.replace(t, "")
    return s.replace("<|bo|>", " [SC] ").strip()

rows = []
for k in range(len(pred_ids)):
    rows.append({
        "idx": k,
        "ref":  to_text(label_ids[k]),   # 正解
        "pred": to_text(pred_ids[k]),    # 推論
    })

pd.DataFrame(rows).to_csv("test_predictions_timestamp.csv", index=False, encoding="utf-8-sig")