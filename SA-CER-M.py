import csv
import re
import sys
import json
from collections import defaultdict

# ===== 設定 =====
PRED_CSV      = "result/test_predictions_timestamp-M.csv"
TEST_CSV      = "/home/tsuda/M1_research/IC-SOT/make_data_名市大/out/1and2spk_timestamp/test.csv"   # ← 名市大に変更
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

    # テスト用プログラムと同じ filter（30 秒以下）をかけた順に会話IDを読む → 推論結果の行と 1 対 1 で対応
    conv_ids = []
    with open(TEST_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if float(r["endTime"]) - float(r["startTime"]) <= MAX_DURATION:
                conv_ids.append(r["conversationID"])

    # 会話IDごとに 正解・推論 を分けて入れる
    refs  = defaultdict(list)
    preds = defaultdict(list)
    with open(PRED_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(conv_ids), f"件数が合いません 推論 {len(rows)} / テスト {len(conv_ids)}"

    for conv, i in zip(conv_ids, rows):
        refs[conv].append(to_eval_text(i["ref"]))
        preds[conv].append(to_eval_text(i["pred"]))

    # 全体 → 会話ごと の順に出力
    all_refs  = sum(refs.values(), [])
    all_preds = sum(preds.values(), [])
    for name, r, p in [("全体", all_refs, all_preds)] + \
                      [(c, refs[c], preds[c]) for c in sorted(refs)]:
        print(f"\n########## {name} ({len(r)} 件) ##########")
        cer(r, p)
        sa_cer(r, p)
        cer_per_side(r, p)
        multi_only(r, p)
    

if __name__ == "__main__":
    main()