import csv
import re
import sys
import json
from collections import defaultdict

# ===== 設定 =====
PRED_CSV      = "/home/tsuda/M1_research/IC-SOT/result/test_predictions_timestamp.csv"
TEST_CSV      = "/home/tsuda/M1_research/IC-SOT/make_data_IC/out/1and2spk_timestamp/test.csv"
NORMALIZE_DIR = "/home/tsuda/M1_research/IC-SOT/make_data_IC"   # cejc_normalize.py のあるフォルダ
MAX_DURATION  = 30.0                     # テスト用プログラムの filter と同じ値にする
#OUT_GROUP_CSV = "result/test_cer_per_group.csv"
#OUT_FILE_CSV  = "result/test_cer_per_file.csv"
# ================

sys.path.append(NORMALIZE_DIR)
from cejc_normalize import normalize

# <|0.00|> などのタイムスタンプトークン
TIMESTAMP_RE = re.compile(r"<\|\d+\.\d+\|>")

# 正規化を行う関数
def to_eval_text(s):
    s = TIMESTAMP_RE.sub("", s).replace("[SC]", "\uE003")    # 先にタイムスタンプを消す　[SC]は残す
    s = normalize(s)                                         # 句読点・空白を消す（タグは残る）
    s = s.replace("<unknown>", "\uE000") \
         .replace("<sing>", "\uE001")    \
         .replace("<beep>", "\uE002")    \
         .replace("\uE003", "[SC]")                          # タグを私用領域の文字に割り当てる
    return s

# 編集距離を計算する関数 (ここよく分かってない)
def edit_distance(ref, pred):
    prev = list(range(len(pred) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(pred)
        for j, h in enumerate(pred, start=1):
            cur[j] = min(prev[j] + 1,              # 脱落
                         cur[j - 1] + 1,           # 挿入
                         prev[j - 1] + (r != h))   # 置換（一致なら 0）
        prev = cur
    return prev[-1]

# CER出力
def cer(refs, preds):
    total_ref_len = 0
    total_errors = 0
    
    for ref, pred in zip(refs, preds):
        
        #　[SC]の削除
        ref  = ref.replace("[SC]", "")
        pred = pred.replace("[SC]", "")
        
        #　正解ラベルの文字数の合計
        total_ref_len += len(ref)
        #　編集距離の合計
        total_errors  += edit_distance(ref, pred) 
    
    cer = 100 * total_errors / total_ref_len
    print(f"CER   : {cer:.2f}%  (誤り {total_errors} / 正解 {total_ref_len} 文字)")

# SA-CER出力
def sa_cer(refs, preds):
    total_ref_len = 0
    total_errors = 0
    
    for ref, pred in zip(refs, preds):
        
        # [SC]で分割
        ref_ch1, _, ref_ch2   = ref.partition("[SC]")
        pred_ch1, _, pred_ch2 = pred.partition("[SC]")
        
        # SA-CER計算
        #　正解ラベルの文字数の合計
        total_ref_len += len(ref_ch1) + len(ref_ch2)
        #　編集距離の合計
        total_errors  += edit_distance(ref_ch1, pred_ch1) + edit_distance(ref_ch2, pred_ch2)
        
    # SA-CER の計算
    sa_cer = 100 * total_errors / total_ref_len
    print(f"SA-CER: {sa_cer:.2f}%  (誤り {total_errors} / 正解 {total_ref_len} 文字)")

# チャンネルごとのCER出力
def cer_per_side(refs, preds):
    """[SC] の前（話者1）と後（話者2）で別々に CER を計算する"""
    ref_len = {"ch1": 0, "ch2": 0}
    errors  = {"ch1": 0, "ch2": 0}

    for ref, pred in zip(refs, preds):
        # [SC]で分割（2つ目以降の [SC] は消す）
        ref_ch1, _, ref_ch2   = ref.partition("[SC]")
        pred_ch1, _, pred_ch2 = pred.partition("[SC]")

        ref_len["ch1"] += len(ref_ch1)
        ref_len["ch2"] += len(ref_ch2)
        errors["ch1"]  += edit_distance(ref_ch1, pred_ch1)
        errors["ch2"]  += edit_distance(ref_ch2, pred_ch2)
        
    print(f"=== チャンネルごとのCER ===")
    for side in ("ch1", "ch2"):
        cer = 100 * errors[side] / ref_len[side]
        print(f"{side} CER: {cer:.2f}%  (誤り {errors[side]} / 正解 {ref_len[side]} 文字)")

#　複数話者のみのデータを使って実験
def multi_only(refs, preds):
    
    multi_refs  = []
    multi_preds = []
    for ref, pred in zip(refs, preds):
        # [SC]で分割
        ref_ch1, _, ref_ch2   = ref.partition("[SC]")
        
        if ref_ch1 and ref_ch2:
            multi_refs.append(ref)
            multi_preds.append(pred)
            
    print(f"=== 複数話者のみ ({len(multi_refs)} 件) ===")
    cer(multi_refs, multi_preds)
    sa_cer(multi_refs, multi_preds)
    


def main():
    
    #　正解ラベルと推論を配列に入れる
    refs  = []
    preds = []
    with open(PRED_CSV, encoding="utf-8-sig", newline="") as f:
        for i in csv.DictReader(f):
            
            # CSCファイルから取り出す
            ref  = i["ref"]
            pred = i["pred"]
            
            #　データの正規化
            ref  = to_eval_text(ref)
            pred = to_eval_text(pred)
            
            #　配列に追加
            refs.append(ref)
            preds.append(pred)
    
    # CERの計算
    cer(refs, preds)
    # SA-CERの計算
    sa_cer(refs, preds)
    
    #追加実験
    # [SC]の前後で誤り率が違うのか分析するため、チャンネルごとにCER出力
    cer_per_side(refs, preds)
    #　複数話者のみのデータを使って実験
    multi_only(refs, preds)
    

if __name__ == "__main__":
    main()