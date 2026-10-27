#!/usr/bin/env python3
"""
ic_sot_all.jsonl から「発話している IC が 1 人または 2 人」の発話グループを取り出し、
評価用の CSV(test.csv)に保存する。実データは少ないので train / dev には分けず、全部 test にする。

入力:
  out/ic_sot_all.jsonl   ic_sot_jsonl.py の出力
  out/audio_list.csv     audio_list.py の出力(1 話者グループのもう片方の CH の音声パスを引くのに使う)  ★

1 話者グループの扱い:  ★
  同じ会話のもう片方の CH を 2 本目のチャンネルにし、その CH の segments は空 [] にする
  (正解ラベルは "text<sc>" または "<sc>text" になる)。

除外するグループ:
  - num_IC が NUM_IC 以外  ★
  - どちらかの話者の音声が無い
  - MAX_DURATION 秒を超える(None なら除外しない)

出力(OUTPUT_DIR/test.csv、1 行 = 1 グループ):
  conversationID, groupID, startTime, endTime   グループの区間(元音声上の絶対秒)
  IC1, IC2                                      IC 番号(番号の若い順)
  IC1_audio_path, IC2_audio_path                IC1 / IC2 の音声(wav 全体のパス)
  IC1_segments, IC2_segments                    JSON 文字列 [{"start", "end", "text"}, ...]
                                                start / end はグループ先頭を 0 とした相対秒
                                                話していない側は []  ★

使い方:
  python make_2spk.py
"""
import csv
import json
import os
import random
from collections import defaultdict

# ===== 設定 =====
INPUT_JSONL = "out/ic_sot_all.jsonl"
OUTPUT_DIR = "out/1and2spk_timestamp"   # ★ 既存の 2spk_timestamp を上書きしないよう別名に
MAX_DURATION = None   # 例: 30.0 にすると 30 秒超のグループを CSV に入れない
AUDIO_LIST_CSV = "out/audio_list.csv"   # ★ 追加
NUM_IC = (1, 2)                         # ★ 追加: (2,) で従来どおり、(1,) で 1 話者だけ
SEED_1SPK = 0                           # ★ 追加
# ================

FIELDS = ["conversationID", "groupID", "startTime", "endTime",
          "IC1", "IC2",
          "IC1_audio_path", "IC2_audio_path",
          "IC1_segments", "IC2_segments"]


def read_audio_list(path):  # ★ 追加
    """{会話ID: {"IC1": wav パス, "IC2": wav パス}}"""
    audio = defaultdict(dict)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["audio_path"] and r["ic"].startswith("IC"):
                audio[r["conversationID"]][r["ic"]] = r["audio_path"]
    return audio


def main():
    audio = read_audio_list(AUDIO_LIST_CSV)   # ★ 追加
    rng = random.Random(SEED_1SPK)            # ★ 追加

    rows = []
    n_spk = defaultdict(int)                  # ★ 集計用
    skip = defaultdict(int)
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["num_IC"] not in NUM_IC:                       # ★ != 2 から変更
                skip[f"num_IC が {NUM_IC} 以外"] += 1
                continue
            # ★ 1 話者グループ: もう片方の CH を「何も話していない」チャンネルとして足す
            if r["num_IC"] == 1:
                spk_id = r["ics"][0]["id"]
                others = sorted(ic for ic in audio[r["conversationID"]] if ic != spk_id)
                if not others:
                    skip["1話者: 相手 CH なし"] += 1
                    continue
                partner = rng.choice(others)
                r["ics"].append({"id": partner,
                                 "audio_path": audio[r["conversationID"]][partner],
                                 "utterances": []})
            if not all(x["audio_path"] for x in r["ics"]):
                skip["IC 音声なし"] += 1
                continue
            # IC 番号の若い順に IC1 / IC2 とする
            s1, s2 = sorted(r["ics"], key=lambda x: int(x["id"].lstrip("IC")))
            dur = r["endTime"] - r["startTime"]
            if MAX_DURATION is not None and dur > MAX_DURATION:
                skip[f"{MAX_DURATION} 秒超"] += 1
                continue
            rows.append({
                "conversationID": r["conversationID"],
                "groupID": r["groupID"],
                "startTime": r["startTime"],
                "endTime": r["endTime"],
                "IC1": s1["id"],
                "IC2": s2["id"],
                "IC1_audio_path": s1["audio_path"],
                "IC2_audio_path": s2["audio_path"],
                "IC1_segments": json.dumps(s1["utterances"], ensure_ascii=False),
                "IC2_segments": json.dumps(s2["utterances"], ensure_ascii=False),
            })
            n_spk[r["num_IC"]] += 1                             # ★

    # 実データは少ないので全部 test にする
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "test.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    hours = sum(x["endTime"] - x["startTime"] for x in rows) / 3600
    n30 = sum(x["endTime"] - x["startTime"] > 30.0 for x in rows)
    print(f"test : {len(rows):6d} グループ(1話者 {n_spk[1]} / 2話者 {n_spk[2]})"   # ★
          f" / {hours:5.1f} 時間 / 30 秒超 {n30} → {path}")
    print("除外: " + ", ".join(f"{k} {v}" for k, v in skip.items()))


if __name__ == "__main__":
    main()