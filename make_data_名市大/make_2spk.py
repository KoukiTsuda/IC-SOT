#!/usr/bin/env python3
"""
ic_sot_all.jsonl から「発話している IC がちょうど 2 人」の発話グループだけを取り出し、
評価用の CSV(test.csv)に保存する。実データは少ないので train / dev には分けず、全部 test にする。

入力:
  out/ic_sot_all.jsonl   ic_sot_jsonl.py の出力

除外するグループ:
  - num_IC != 2(看護師だけ / 患者だけが話しているグループ)
  - どちらかの話者の音声が無い
  - MAX_DURATION 秒を超える(None なら除外しない)

出力(OUTPUT_DIR/test.csv、1 行 = 1 グループ):
  conversationID, groupID, startTime, endTime   グループの区間(元音声上の絶対秒)
  IC1, IC2                                      IC 番号(番号の若い順)
  IC1_audio_path, IC2_audio_path                IC1 / IC2 の音声(wav 全体のパス)
  IC1_segments, IC2_segments                    JSON 文字列 [{"start", "end", "text"}, ...]
                                                start / end はグループ先頭を 0 とした相対秒

使い方:
  python make_2spk.py
"""
import csv
import json
import os
from collections import defaultdict

# ===== 設定 =====
INPUT_JSONL = "out/ic_sot_all.jsonl"
OUTPUT_DIR = "out/2spk_timestamp"
MAX_DURATION = None   # 例: 30.0 にすると 30 秒超のグループを CSV に入れない
# ================

FIELDS = ["conversationID", "groupID", "startTime", "endTime",
          "IC1", "IC2",
          "IC1_audio_path", "IC2_audio_path",
          "IC1_segments", "IC2_segments"]


def main():
    rows = []
    skip = defaultdict(int)
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["num_IC"] != 2:
                skip["num_IC != 2"] += 1
                continue
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

    # 実データは少ないので全部 test にする
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "test.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    hours = sum(x["endTime"] - x["startTime"] for x in rows) / 3600
    n30 = sum(x["endTime"] - x["startTime"] > 30.0 for x in rows)
    print(f"test : {len(rows):6d} グループ / {hours:5.1f} 時間 / 30 秒超 {n30} → {path}")
    print("除外: " + ", ".join(f"{k} {v}" for k, v in skip.items()))


if __name__ == "__main__":
    main()