#!/usr/bin/env python3
"""
発話グループごとに、グループに登場する IC(話者)の音声パスと発話列を JSONL で出力する。
1 行 = 1 グループ、IC は配列なので IC 数の上限なし。単体で動作する(IC_sot.py 不要)。

入力:
  out/groups/groups_{会話ID}.csv          グループの startTime / endTime(utterance_groups.py)
  out/normalized/normalized_{会話ID}.csv  groupID 付きの正規化済み発話(cejc_normalize.py)
  out/audio_list.csv                      会話 × IC の wav パス(audio_list.py)

出力:
  out/ic_sot_jsonl/ic_sot_{会話ID}.jsonl   会話ごと
  out/ic_sot_all.jsonl                     全会話まとめ

1 行の例:
  {"conversationID": "C001_001", "groupID": "2", "startTime": 0.133, "endTime": 1.866,
   "num_IC": 3,
   "ics": [{"order": 1, "id": "IC01", "audio_path": ".../C001_001_IC01.wav",
            "utterances": [{"start": 0.0, "end": 1.733, "text": "それで落ち落ち武者は何。"}]},
           ...]}

  ics の並び順はグループ内で最初に話し始めた順(SOT の FIFO 順)。同時刻なら IC 番号順。
  utterances の start / end はグループの startTime を 0 とした相対秒。

使い方:
  utterance_groups.py → cejc_normalize.py → audio_list.py の後に
  python ic_sot_jsonl.py                     # 全会話
  python ic_sot_jsonl.py C001_001 C001_002   # 指定した会話だけ(動作確認用、まとめファイルは作らない)

読み込み例:
  import json
  with open("out/ic_sot_all.jsonl", encoding="utf-8") as f:
      groups = [json.loads(line) for line in f]
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

# ===== 設定 =====
GROUPS_DIR = "out/groups"
NORMALIZED_DIR = "out/normalized"
AUDIO_LIST_CSV = "out/audio_list.csv"
OUTPUT_DIR = "out/ic_sot_jsonl"            # 会話ごとの出力先
OUTPUT_ALL_JSONL = "out/ic_sot_all.jsonl"  # 全会話をまとめた出力
MAX_IC = None      # None なら上限なし。数値を入れるとそれを超えるグループは除外
SKIP_EMPTY = True  # 正規化で発話が全部消えたグループは出力しない
# ================


def read_audio_list(path):
    """{(会話ID, "IC01"): wav パス}"""
    audio = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            #左辺：キー 右辺：値
            audio[(r["conversationID"], r["ic"])] = r["audio_path"]
    return audio


def build_records(conv_id, audio):
    """1 会話分のグループレコード(dict)のリストを返す"""
    with open(os.path.join(GROUPS_DIR, f"groups_{conv_id}.csv"), encoding="utf-8-sig", newline="") as f:
        groups = list(csv.DictReader(f))

    # groupID → IC → [(start, end, text), ...]
    utts = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(NORMALIZED_DIR, f"normalized_{conv_id}.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            utts[r["groupID"]][r["IC"]].append(
                (float(r["startTime"]), float(r["endTime"]), r["text"]))

    records = []
    for g in groups:
        gid, g_start, g_end = g["groupID"], float(g["startTime"]), float(g["endTime"])
        by_ic = utts.get(gid, {})
        if SKIP_EMPTY and not by_ic:
            continue
        # 先着順(最初の発話開始時刻 → IC 番号)
        order = sorted(by_ic, key=lambda ic: (min(u[0] for u in by_ic[ic]), ic))
        ics = []
        for k, ic in enumerate(order, start=1):
            path = audio.get((conv_id, ic), "")
            if not path:
                print(f"[警告] 音声がありません: {conv_id} {ic} (group {gid})")
            ics.append({
                "order": k,
                "id": ic,
                "audio_path": path,
                "utterances": [{"start": round(s - g_start, 3), "end": round(e - g_start, 3), "text": t}
                               for s, e, t in sorted(by_ic[ic])],
            })
        records.append({"conversationID": conv_id, "groupID": gid,
                        "startTime": g_start, "endTime": g_end,
                        "num_IC": len(ics), "ics": ics})
    return records


def write_jsonl(f, records):
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    audio = read_audio_list(AUDIO_LIST_CSV)
    if len(sys.argv) > 1:
        conv_ids, write_all = sys.argv[1:], False
    else:
        conv_ids = sorted(os.path.basename(p)[len("groups_"):-len(".csv")]
                          for p in glob.glob(os.path.join(GROUPS_DIR, "groups_*.csv")))
        write_all = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_conv = n_group = dropped = 0
    # 全体ファイルは一時ファイルに書いてから置き換え(途中で止まっても壊れたファイルを残さない)
    tmp_all = OUTPUT_ALL_JSONL + ".tmp"
    fall = open(tmp_all, "w", encoding="utf-8") if write_all else None
    try:
        for conv_id in conv_ids:
            if not os.path.exists(os.path.join(NORMALIZED_DIR, f"normalized_{conv_id}.csv")):
                print(f"[警告] normalized がありません(スキップ): {conv_id}")
                continue
            records = build_records(conv_id, audio)
            if MAX_IC:
                kept = [rec for rec in records if rec["num_IC"] <= MAX_IC]
                dropped += len(records) - len(kept)
                records = kept
            with open(os.path.join(OUTPUT_DIR, f"ic_sot_{conv_id}.jsonl"), "w", encoding="utf-8") as f:
                write_jsonl(f, records)
            if fall:
                write_jsonl(fall, records)
            n_conv += 1
            n_group += len(records)
    finally:
        if fall:
            fall.close()
    if write_all:
        os.replace(tmp_all, OUTPUT_ALL_JSONL)

    print(f"{n_conv} 会話 / {n_group} グループ"
          + (f" / IC 数超過で除外 {dropped}" if dropped else "")
          + f" → {OUTPUT_ALL_JSONL if write_all else OUTPUT_DIR}")


if __name__ == "__main__":
    main()
