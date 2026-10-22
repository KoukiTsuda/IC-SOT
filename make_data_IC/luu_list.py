#!/usr/bin/env python3
"""
CEJC の全会話の luu.csv のパスを 1 つの CSV にまとめる。

出力(1 行 = 1 会話):
  luu_csv   長い発話単位 CSV のパス(cp932)

使い方:
  下の設定値を書き換えてから  python luu_list.py  で実行する。
"""
import csv
import os

# ===== 設定 =====
CEJC_ROOT = "/home/tsuda/dataset/CEJC"
OUTPUT_CSV = "out/luu_list.csv"
# ================

#/home/tsuda/dataset/CEJC/Version1.0/data
DATA_DIR = os.path.join(CEJC_ROOT, "Version1.0", "data")
#/home/tsuda/dataset/CEJC/metaInfo/conversation.csv
CONVERSATION_META = os.path.join(CEJC_ROOT, "metaInfo", "結合用", "conversation.csv")


def main():
    # 会話の一覧(会話ID, セッションID, ...)を読み込む
    with open(CONVERSATION_META, encoding="cp932", newline="") as f:
        meta = list(csv.DictReader(f))

    paths = []
    for m in meta:
        conv_id, session_id = m["会話ID"], m["セッションID"]
        # data/協力者ID(セッションIDの先頭4文字)/セッションID/会話ID-luu.csv
        luu_csv = os.path.join(DATA_DIR, session_id[:4], session_id, f"{conv_id}-luu.csv")
        if not os.path.exists(luu_csv):
            print(f"[警告] luu.csv がありません: {luu_csv}")
            continue
        paths.append(luu_csv)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["luu_csv"])
        for p in paths:
            w.writerow([p])

    print(f"{len(paths)} 会話 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
