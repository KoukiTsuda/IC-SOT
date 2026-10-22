#!/usr/bin/env python3
"""
CEJC の全会話について、話者ごとの IC レコーダ音声(ICxx.wav)のパスを 1 つの CSV にまとめる。
会話の一覧は luu_list.csv(luu_list.py で作成)を使う。

対象:
  {会話ID}_IC01.wav, {会話ID}_IC02.wav, ... (IC + 数字 2 桁)
対象外:
  IC0A / IC0B  … 話者個人ではない 2ch の録音
  ICxxef.wav   … 一部の会話にだけある別バージョン(通常版と同じ長さ)
  読めない wav … 0 バイト等(7z の展開失敗。check_cejc.py で確認できる)

形式の注意:
  ほとんどは 16kHz モノラルだが、44.1kHz / 2ch のファイルもある(2ch は L/R がほぼ同一)。
  ここでは変換せず samplerate / channels を記録するだけにし、切り出し時にモノラル化・16kHz 化する。

出力(1 行 = 1 会話 × 1 IC):
  conversationID  会話ID (例: K003_002a)
  ic              IC 番号 (例: IC01)。luu.csv の speakerID の先頭 4 文字と対応する
  audio_path      wav のパス

使い方:
  下の設定値を書き換えてから  python audio_list.py  で実行する。
"""
import csv
import glob
import os

import soundfile as sf


# ===== 設定 =====
LUU_LIST_CSV = "out/luu_list.csv"     # luu.csv のパス一覧(luu_list.py で作成)
OUTPUT_CSV = "out/audio_list.csv"     # 出力先
# ================


def read_luu_list(path):
    """luu_list.csv から luu.csv のパスのリストを返す"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["luu_csv"] for r in csv.DictReader(f)]


def main():
    luu_paths = read_luu_list(LUU_LIST_CSV)

    rows = []
    for luu_csv in luu_paths:
        #例
            #luu_csv = "~/K003/K003_002/K003_002a-luu.csv"
            #conv_id = "K003_002a"
            #conv_dir = "~/K003/K003_002"
        conv_id = os.path.basename(luu_csv).rsplit("-luu.csv", 1)[0]
        conv_dir = os.path.dirname(luu_csv)
        
        # 音声ファイルの命名規則に従ってファイルを探し、名前順に並べる。
        # conv_dir/conv_id_IC[0-9][0-9].wav
        wavs = sorted(glob.glob(os.path.join(conv_dir, f"{conv_id}_IC[0-9][0-9].wav")))
        found = 0
        
        for wav in wavs:
            # "K003_002a_IC01.wav" → "IC01"だけ取り出す
            ic = os.path.basename(wav).removeprefix(f"{conv_id}_").removesuffix(".wav")
            try:
                info = sf.info(wav)
            except sf.LibsndfileError:
                print(f"[警告] 読めない wav(スキップ): {wav}")
                continue
            rows.append({"conversationID": conv_id, "ic": ic, "audio_path": wav})
            found += 1

        if found == 0:
            print(f"[警告] IC 音声がありません: {conv_id}")

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conversationID", "ic", "audio_path"])
        w.writeheader()
        w.writerows(rows)

    n_conv = len({r["conversationID"] for r in rows})
    print(f"{n_conv} 会話 / {len(rows)} ファイル → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
