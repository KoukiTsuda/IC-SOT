#!/usr/bin/env python3
"""
anotetion フォルダ以下(サブフォルダも含む)にある CSV ファイルのパスを 1 つの CSV にまとめる。

出力(1 行 = 1 ファイル):
  conversationID   ファイル名から .csv を除いたもの(例: 1792580127975_2025-10-13 23.15.25)
  annotation_csv   CSV ファイルのパス

使い方:
  下の設定値を書き換えてから  python annotation_list.py  で実行する。
"""
import csv
import glob
import os

# ===== 設定 =====
ANNOTATION_DIR = "/home/tsuda/dataset/名市大/annotation/ver2"              # 探すフォルダ
OUTPUT_CSV = "out/luu_list.csv"    # 出力先
# ================


def main():
    
    paths = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.csv"), recursive=True))
    if not paths:
        print(f"[警告] CSV が見つかりません: {ANNOTATION_DIR}")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["luu_csv"])
        for p in paths:
            w.writerow([p])

    print(f"{len(paths)} ファイル → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
