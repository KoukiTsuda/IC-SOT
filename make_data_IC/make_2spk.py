#!/usr/bin/env python3
"""
ic_sot_all.jsonl から「発話している IC がちょうど 2 人」の発話グループだけを取り出し、
協力者単位で train / dev / test に分けて CSV に保存する(学習スクリプト用)。
 
入力:
  out/ic_sot_all.jsonl   ic_sot_jsonl.py の出力
  out/split.csv          split.py の出力(協力者ID → train/dev/test)
                         無い場合は out/groups の会話長から協力者ごとの時間を見積もり、
                         split.py と同じ方法(assign)で分割して out/split_from_groups.csv に保存する
 
除外するグループ:
  - num_IC != 2
  - どちらかの話者の IC 音声が無い(N10A など IC レコーダを付けていない話者)
  - MAX_DURATION 秒を超える(None なら除外しない。学習側でも 30 秒フィルタあり)
 
出力(OUTPUT_DIR/{train,dev,test}.csv、1 行 = 1 グループ):
  conversationID, groupID, startTime, endTime   グループの区間(元音声上の絶対秒)
  speaker1_id, speaker2_id                      IC 番号(先に話し始めた方が speaker1 = SOT の FIFO 順)
  ch1_mic_audio_path, ch2_mic_audio_path        speaker1 / speaker2 の IC 音声(wav 全体のパス)
  speaker1_segments, speaker2_segments          JSON 文字列 [{"start", "end", "text"}, ...]
                                                start / end はグループ先頭を 0 とした相対秒
 
使い方:
  python make_2spk_csv.py
"""
import csv
import glob
import json
import os
import random
from collections import defaultdict
 
import split as split_cfg  # split.py の設定値と assign() を流用する
 
# ===== 設定 =====
INPUT_JSONL = "out/ic_sot_all.jsonl"
SPLIT_CSV = "out/split.csv"
GROUPS_DIR = "out/groups"
FALLBACK_SPLIT_CSV = "out/split_from_groups.csv"
OUTPUT_DIR = "out/2spk_timestamp"
MAX_DURATION = None   # 例: 30.0 にすると 30 秒超のグループを CSV に入れない
# ================
 
FIELDS = ["conversationID", "groupID", "startTime", "endTime",
          "IC1", "IC2",
          "IC1_audio_path", "IC2_audio_path",
          "IC1_segments", "IC2_segments"]
 
 
def read_split(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["informantID"]: r["split"] for r in csv.DictReader(f)}
 
 
def make_split_from_groups():
    """conversation.csv が無い環境用: groups の最後の endTime を会話長とみなして split.py と同じ手順で分割"""
    minutes = defaultdict(float)
    for p in glob.glob(os.path.join(GROUPS_DIR, "groups_*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            ends = [float(r["endTime"]) for r in csv.DictReader(f)]
        if ends:
            conv_id = os.path.basename(p)[len("groups_"):-len(".csv")]
            minutes[conv_id[:4]] += max(ends) / 60
 
    by_prefix = defaultdict(list)
    for i in sorted(minutes):
        by_prefix[i[0]].append(i)
 
    rng = random.Random(split_cfg.SEED)
    split = {}
    for prefix, ids in sorted(by_prefix.items()):
        if prefix in split_cfg.TRAIN_ONLY_PREFIXES:
            split.update({i: "train" for i in ids})
            continue
        total = sum(minutes[i] for i in ids)
        t_test, t_dev = total * split_cfg.TEST_RATIO, total * split_cfg.DEV_RATIO
        best, best_cost = None, float("inf")
        for _ in range(split_cfg.N_TRIALS):
            s, cost = split_cfg.assign(ids, minutes, t_test, t_dev, rng)
            if cost < best_cost:
                best, best_cost = s, cost
        split.update(best)
 
    with open(FALLBACK_SPLIT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["informantID", "split", "minutes"])
        w.writeheader()
        for i in sorted(split):
            w.writerow({"informantID": i, "split": split[i], "minutes": round(minutes[i], 1)})
    print(f"[注意] {SPLIT_CSV} が無いので会話長を groups から見積もって分割しました → {FALLBACK_SPLIT_CSV}")
    return split
 
 
def main():
    split = read_split(SPLIT_CSV) if os.path.exists(SPLIT_CSV) else make_split_from_groups()
 
    rows = {"train": [], "dev": [], "test": []}
    skip = defaultdict(int)
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["num_IC"] != 2:
                skip["num_IC != 2"] += 1
                continue
            # IC 音声が無い話者(id が Z10A / N10A など IC 番号でない)を含むグループは除外
            if not all(x["audio_path"] for x in r["ics"]):
                skip["IC 音声なし"] += 1
                continue
            # IC 番号の若い順に IC1 / IC2 とする(IC01 < IC03 < IC05 ...)
            s1, s2 = sorted(r["ics"], key=lambda x: int(x["id"].lstrip("IC")))
            if not s1["audio_path"] or not s2["audio_path"]:
                skip["IC 音声なし"] += 1
                continue
            dur = r["endTime"] - r["startTime"]
            if MAX_DURATION is not None and dur > MAX_DURATION:
                skip[f"{MAX_DURATION} 秒超"] += 1
                continue
            sp = split.get(r["conversationID"][:4])
            if sp is None:
                skip["split 不明"] += 1
                continue
            rows[sp].append({
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
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sp, rs in rows.items():
        path = os.path.join(OUTPUT_DIR, f"{sp}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rs)
        hours = sum(x["endTime"] - x["startTime"] for x in rs) / 3600
        n30 = sum(x["endTime"] - x["startTime"] > 30.0 for x in rs)
        n_inf = len({x["conversationID"][:4] for x in rs})
        print(f"{sp:5s}: {len(rs):6d} グループ / {hours:5.1f} 時間 / 協力者 {n_inf:2d} 人 / 30 秒超 {n30} → {path}")
    print("除外: " + ", ".join(f"{k} {v}" for k, v in skip.items()))
 
 
if __name__ == "__main__":
    main()