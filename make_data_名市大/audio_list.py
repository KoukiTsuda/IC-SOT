#!/usr/bin/env python3
"""
audio フォルダ直下にある wav ファイルのパスを 1 つの CSV にまとめる。
wav はチャンネルごとに分かれていて、ファイル名の末尾が _CHx.wav になっている前提。

出力(1 行 = 1 wav):
  conversationID   ファイル名から _CHx.wav を除いたもの
                   (例: 1792580127975_2025-10-13 23.15.25_CH1.wav → 1792580127975_2025-10-13 23.15.25)
  ic               チャンネル番号に対応する IC (CH1 → IC1, CH2 → IC2)
  audio_path       wav のパス

使い方:
  下の設定値を書き換えてから  python audio_list.py  で実行する。
"""
import csv
import glob
import os

# ===== 設定 =====
AUDIO_DIR = "/home/tsuda/dataset/名市大/audio"              # 探すフォルダ
OUTPUT_CSV = "out/audio_list.csv"    # 出力先
# ================


def main():
    # フォルダ直下の wav を探し、名前順に並べる
    paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    if not paths:
        print(f"[警告] WAV ファイルが見つかりません: {AUDIO_DIR}")
        
    # 出力先のフォルダ(out)が無い場合フォルダを作る
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    
    # CSV に書き込む
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conversationID", "ic", "audio_path"])
        for p in paths:
            
            # 空の wav はスキップする
            if os.path.getsize(p) == 0:
                print(f"[警告] 空の wav(スキップ): {p}")
                continue
            
            # ファイル名からconversationIDとCH番号を抽出
            file_name = os.path.splitext(os.path.basename(p))[0]
            conv_id, ch = file_name.rsplit("_CH", 1)

            # CSVに書き込む
            w.writerow([conv_id, f"IC{ch}", p])

    print(f"{len(paths)} ファイル → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
