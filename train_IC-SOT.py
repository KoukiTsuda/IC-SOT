import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
import json

#libri2mix_train-clean-100_manifest.csv等の中身
#/home/tsuda/M1_research/pyroom/make_room_mix3.pyで作成された
#mixture_id,speaker1_audio_path,speaker2_audio_path,ch1_mic_audio_path,ch2_mic_audio_path,speaker1_text_path,speaker2_text_path
#混交音声のid,スピーカ１の音声パス,スピーカ２の音声パス,マイク１の音声パス,マイク２の音声パス,スピーカ１のテキストパス,スピーカ２のテキストパスで構成されてる

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


BASE_MODEL = "openai/whisper-medium"  # Whisper のモデル名
EXPORT_TRAINED_MODEL  = "/home/tsuda/M1_research/IC-SOT/IC-SOT-weight"  # 学習済みモデルの保存フォルダ
TRAIN_CSV_ROOT = "/home/tsuda/M1_research/IC-SOT/make_data_IC/out/2spk_timestamp"  # 学習用CSVのあるフォルダ
NORMALIZE_DIR = "/home/tsuda/M1_research/IC-SOT/make_data_IC" # テキスト正規化のファイル、句読点削除に使用

ts_dataset = load_dataset(
    "csv",
    data_files={
        "train": f"{TRAIN_CSV_ROOT}/train.csv",
        "dev":   f"{TRAIN_CSV_ROOT}/dev.csv",
    },
    keep_default_na=False,
)

print(ts_dataset)


#学習に不要なカラムを削除する
ts_dataset = ts_dataset.remove_columns(["conversationID", "groupID"])

#30秒を超える音声は学習に使わないようにする(Whisper は 30 秒以降を切り捨てるため)
#startTime / endTime は map の remove_columns で消えるので、map より前に行う
ts_dataset = ts_dataset.filter(lambda b: b["endTime"] - b["startTime"] <= 30.0)


#デバッグ用
import os
DEBUG = os.environ.get("DEBUG", "0") == "1"

if DEBUG:
    ts_dataset["train"] = ts_dataset["train"].select(range(32))
    ts_dataset["dev"]   = ts_dataset["dev"].select(range(8))
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

# float16で特徴量を保存し、map中のメモリ使用量を抑える
#ts_dataset = ts_dataset.map(prepare_dataset, remove_columns=ts_dataset.column_names["train"], num_proc=1, writer_batch_size=1)
for split in ("train", "dev"):
    ts_dataset[split] = ts_dataset[split].map(
        prepare_dataset,
        remove_columns=ts_dataset[split].column_names,
        num_proc=1,
        writer_batch_size=1,
    )
    

# ラベルの長さを確認して、300トークンを超えるものがあるときだけ除く
for split in ("train", "dev"):
    max_len = max(len(l) for l in ts_dataset[split]["labels"])
    print(split, "最大ラベル長:", max_len)
    if max_len > 300:
        ts_dataset[split] = ts_dataset[split].filter(
            lambda labels: len(labels) <= 300,
            input_columns=["labels"],   # labels列だけ読むので速い
        )

print(ts_dataset)

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
    
    #def generate(self, input_features=None, **kwargs):
        #if input_features is not None:
        #    input_features = self.fusion(input_features)
        #kwargs.pop("labels", None)  # ★generateには不要なので渡さない
        #return super().generate(input_features=input_features, **kwargs)
    
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


model = TwoChannelWhisper.from_pretrained(BASE_MODEL)
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
sys.path.append(NORMALIZE_DIR)
from cejc_normalize import normalize

metric = evaluate.load("cer")


TS_BEGIN = tokenizer.convert_tokens_to_ids("<|0.00|>")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    for a in (pred_ids, label_ids):
            a[a == -100] = tokenizer.pad_token_id      # パディング
            a[a >= TS_BEGIN] = tokenizer.pad_token_id  # タイムスタンプを除外

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 学習は句読点ありのテキストで行い、評価のときだけ句読点・空白を外して CER を計算する
    pred_str = [normalize(s) for s in pred_str]
    label_str = [normalize(s) for s in label_str]

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


#学習の設定
from transformers import Seq2SeqTrainingArguments

#学習で用いるハイパーパラメータの設定
training_args = Seq2SeqTrainingArguments(
    output_dir=EXPORT_TRAINED_MODEL,  # change to a repo name of your choice
    per_device_train_batch_size=16,    # GPU１台あたり16件のデータを処理する
    gradient_accumulation_steps=1,     # バッチをいくつためてから勾配を計算するかの設定。
    learning_rate=1e-5,                # 学習率
    num_train_epochs=11,               # 学習する総エポック数を指定
    save_total_limit=4,
    eval_strategy="epoch",             # 評価のタイミングをエポック単位で行う
    save_strategy="epoch",             # モデルの保存を行うエポック数
    gradient_checkpointing=True,       # 勾配チェックポイントを有効にすることで、メモリ使用量を削減する
    fp16=True,                         # 16bit精度で学習することで、メモリ使用量を削減する 
    per_device_eval_batch_size=8,      # GPU１台あたり8件のデータを処理する
    predict_with_generate=True,        # compute_metricsで評価するさいgenerate()を用いて予測することにより、推論時と同じ形式で評価できる
    generation_max_length=300,         # generate()で生成する最大トークン数を指定する
    logging_steps=25,                  #ログを出力するステップ数
    report_to=["tensorboard"],         #ログの出力先をTensorBoardに設定
    load_best_model_at_end=True,       #学習終了時に最も性能の良かったモデルをロードする
    metric_for_best_model="cer",       #学習終了時に最も性能の良かったモデルを選択する際の評価指標をCERに設定
    greater_is_better=False,           #CERは小さいほど良いのでFalseに設定(小さいほど良い評価指標はFalseに設定する,高いほど良い評価指標はTrueに設定する)
    #push_to_hub=True,                 #学習終了後にモデルをHugging Face Hubにアップロードする
    remove_unused_columns=False,   # ★これを追加


)

from transformers import Seq2SeqTrainer

#学習で用いるモデル、データ、関数等の実態の設定
trainer = Seq2SeqTrainer(
    args=training_args,                     #Seq2SeqTrainingArgumentsで設定されたハイパーパラメーター
    model=model,                            #学習するモデル
    train_dataset=ts_dataset["train"],      #学習用データセット
    eval_dataset=ts_dataset["dev"],         #評価用データセット
    data_collator=data_collator,            #データコレーター
    compute_metrics=compute_metrics,        #評価指標の計算関数 Seq2SeqTrainingArgumentsのevaluation_strategyで設定されたタイミングでテストデータセットを用いて現在のモデルのCERを測定する。
    processing_class=processor.feature_extractor,  #トークナイザー
)

trainer.train()