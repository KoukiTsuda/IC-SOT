#!/usr/bin/env python3
"""
ic_sot_all.jsonl から「発話している IC が 1 人または 2 人」の発話グループを取り出し、
協力者単位で train / dev / test に分けて CSV に保存する(学習スクリプト用)。

入力:
  out/ic_sot_all.jsonl   ic_sot_jsonl.py の出力
  out/split.csv          split.py の出力(協力者ID → train/dev/test)
                         無い場合は out/groups の会話長から協力者ごとの時間を見積もり、
                         split.py と同じ方法(assign)で分割して out/split_from_groups.csv に保存する
  out/audio_list.csv     audio_list.py の出力(1 話者グループの相手 IC の音声パスを引くのに使う)  ★

1 話者グループの扱い:  ★
  同じ会話の別の IC(音声あり)を 1 つランダムに選んで 2 本目のチャンネルにし、
  その IC の segments は空 [] にする(学習ラベルは "text<sc>" または "<sc>text" になる)。
  IC が 1 本しかない会話のグループは除外。

除外するグループ:
  - num_IC が 1, 2 以外  ★
  - どちらかの話者の IC 音声が無い(N10A など IC レコーダを付けていない話者)
  - MAX_DURATION 秒を超える(None なら除外しない。学習側でも 30 秒フィルタあり)

出力(OUTPUT_DIR/{train,dev,test}.csv、1 行 = 1 グループ):
  conversationID, groupID, startTime, endTime   グループの区間(元音声上の絶対秒)
  IC1, IC2                                      IC 番号(番号の若い方が IC1)
  IC1_audio_path, IC2_audio_path                IC1 / IC2 の IC 音声(wav 全体のパス)
  IC1_segments, IC2_segments                    JSON 文字列 [{"start", "end", "text"}, ...]
                                                start / end はグループ先頭を 0 とした相対秒
                                                話していない側は []  ★

使い方:
  python make_2spk.py
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
OUTPUT_DIR = "out/1and2spk_timestamp"   # ★ 既存の 2spk_timestamp を上書きしないよう別名に
MAX_DURATION = None   # 例: 30.0 にすると 30 秒超のグループを CSV に入れない
AUDIO_LIST_CSV = "out/audio_list.csv"   # ★ 追加
NUM_IC = (1, 2)                         # ★ 追加: 使うグループの話者数。(2,) にすれば従来どおり
SEED_1SPK = 0                           # ★ 追加: 1 話者グループの相手 IC を選ぶシード
# ================

FIELDS = ["conversationID", "groupID", "startTime", "endTime",
          "IC1", "IC2",
          "IC1_audio_path", "IC2_audio_path",
          "IC1_segments", "IC2_segments"]


def read_split(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["informantID"]: r["split"] for r in csv.DictReader(f)}


def read_audio_list(path):  # ★ 追加
    """{会話ID: {"IC01": wav パス, ...}}(IC 番号でない話者・音声なしは除く)"""
    audio = defaultdict(dict)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["audio_path"] and r["ic"].startswith("IC"):
                audio[r["conversationID"]][r["ic"]] = r["audio_path"]
    return audio


def make_split_from_groups():
    # (変更なし: 今のコードのまま)
    ...


def main():
    split = read_split(SPLIT_CSV) if os.path.exists(SPLIT_CSV) else make_split_from_groups()
    audio = read_audio_list(AUDIO_LIST_CSV)   # ★ 追加
    rng = random.Random(SEED_1SPK)            # ★ 追加

    rows = {"train": [], "dev": [], "test": []}
    n_spk = {"train": defaultdict(int), "dev": defaultdict(int), "test": defaultdict(int)}  # ★ 集計用
    skip = defaultdict(int)
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["num_IC"] not in NUM_IC:                       # ★ != 2 から変更
                skip[f"num_IC が {NUM_IC} 以外"] += 1
                continue
            # ★ 1 話者グループ: 同じ会話の別 IC を「何も話していない」チャンネルとして足す
            if r["num_IC"] == 1:
                spk_id = r["ics"][0]["id"]
                others = sorted(ic for ic in audio[r["conversationID"]] if ic != spk_id)
                if not others:
                    skip["1話者: 相手 IC なし"] += 1
                    continue
                partner = rng.choice(others)
                r["ics"].append({"id": partner,
                                 "audio_path": audio[r["conversationID"]][partner],
                                 "utterances": []})
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
            n_spk[sp][r["num_IC"]] += 1                        # ★

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
        print(f"{sp:5s}: {len(rs):6d} グループ(1話者 {n_spk[sp][1]} / 2話者 {n_spk[sp][2]})"   # ★
              f" / {hours:5.1f} 時間 / 協力者 {n_inf:2d} 人 / 30 秒超 {n30} → {path}")
    print("除外: " + ", ".join(f"{k} {v}" for k, v in skip.items()))


if __name__ == "__main__":
    main()