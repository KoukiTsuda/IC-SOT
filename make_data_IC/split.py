#!/usr/bin/env python3
"""
CEJC を協力者ID単位で train / dev / test に分割し、split.csv に保存する。

協力者ID:
  会話IDの先頭 4 文字(例: K003_002a → K003)。
  話者は複数の協力者グループにまたがらない(participant.csv で確認済み)ため、
  協力者単位で分ければ train と dev / test で話者が重複しない。

選び方:
  接頭辞(T / K / W ...)ごとに、会話時間の合計が TEST_RATIO / DEV_RATIO になるよう協力者を選ぶ。
  協力者ごとの時間はばらつきが大きいので、ランダムな順番で貪欲に割り当てる試行を N_TRIALS 回行い、
  目標時間との差が最も小さい分割を採用する(SEED を固定しているので毎回同じ結果になる)。
  TRAIN_ONLY_PREFIXES の接頭辞(協力者が少ないもの)は全て train に入れる。

出力(1 行 = 1 協力者):
  informantID   協力者ID
  split         train / dev / test
  numConversations, minutes, numCore   会話数、会話時間の合計(分)、コア会話数

使い方:
  下の設定値を書き換えてから  python make_split.py  で実行する。
  マニフェスト作成時は split.csv を読み、会話IDの先頭 4 文字で split を引く。
"""
import csv
import os
import random
from collections import defaultdict

# ===== 設定 =====
CEJC_ROOT = "/home/tsuda/dataset/CEJC"
OUTPUT_CSV = "out/split.csv"
TEST_RATIO = 0.10                   # 接頭辞ごとの会話時間に対する test の割合
DEV_RATIO = 0.05                    # 同 dev の割合
TRAIN_ONLY_PREFIXES = ["C", "S"]    # 協力者が少ないので全て train に入れる接頭辞
N_TRIALS = 10000                    # 割り当てを試す回数
SEED = 0
# ================

#/home/tsuda/dataset/CEJC/metaInfo/結合用/conversation.csv
CONVERSATION_META = os.path.join(CEJC_ROOT, "metaInfo", "結合用", "conversation.csv")


def read_informants(path):
    """conversation.csv から {協力者ID: {"numConversations", "minutes", "numCore"}} を返す"""
    info = defaultdict(lambda: {"numConversations": 0, "minutes": 0, "numCore": 0})
    with open(path, encoding="cp932", newline="") as f:
        for r in csv.DictReader(f):
            i = info[r["会話ID"][:4]]
            i["numConversations"] += 1
            i["minutes"] += int(r["会話時間"])
            i["numCore"] += int(r["コア"] == "1")
    return dict(info)


def assign(ids, minutes, target_test, target_dev, rng):
    """
    ids をランダムな順番に並べ、test → dev の順に目標時間に届くまで割り当てる(残りは train)。
    戻り値: ({協力者ID: split}, 目標時間との差の合計)
    """
    ids = ids[:]
    rng.shuffle(ids)
    split, total = {}, {"test": 0, "dev": 0}
    for i in ids:
        if total["test"] < target_test:
            s = "test"
        elif total["dev"] < target_dev:
            s = "dev"
        else:
            s = "train"
        split[i] = s
        if s != "train":
            total[s] += minutes[i]
    cost = abs(total["test"] - target_test) + abs(total["dev"] - target_dev)
    return split, cost


def main():
    info = read_informants(CONVERSATION_META)
    minutes = {i: v["minutes"] for i, v in info.items()}

    # 接頭辞ごとに協力者をまとめる  {"T": ["T001", ...], "K": [...], ...}
    by_prefix = defaultdict(list)
    for i in sorted(info):
        by_prefix[i[0]].append(i)

    rng = random.Random(SEED)
    split = {}
    for prefix, ids in sorted(by_prefix.items()):
        if prefix in TRAIN_ONLY_PREFIXES:
            split.update({i: "train" for i in ids})
            continue
        total = sum(minutes[i] for i in ids)
        target_test, target_dev = total * TEST_RATIO, total * DEV_RATIO
        # N_TRIALS 回試して目標時間に最も近い分割を採用する
        best, best_cost = None, float("inf")
        for _ in range(N_TRIALS):
            s, cost = assign(ids, minutes, target_test, target_dev, rng)
            if cost < best_cost:
                best, best_cost = s, cost
        split.update(best)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["informantID", "split", "numConversations", "minutes", "numCore"])
        w.writeheader()
        for i in sorted(info):
            w.writerow({"informantID": i, "split": split[i], **info[i]})

    # 集計を表示
    all_min = sum(minutes.values())
    for s in ["train", "dev", "test"]:
        ids = [i for i in sorted(info) if split[i] == s]
        m = sum(minutes[i] for i in ids)
        core = sum(info[i]["numCore"] for i in ids)
        print(f"{s:5s}: 協力者 {len(ids):2d} 人 / {m / 60:6.1f} 時間 ({100 * m / all_min:4.1f}%) / コア {core:2d} 会話"
              f"  {' '.join(ids) if s != 'train' else ''}")
    print(f"→ {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
