#!/usr/bin/env python3
"""
テスト用プログラムが出力した test_predictions_timestamp.csv から、
グループごと・ファイル(会話)ごとの CER を計算する。テスト用プログラムは変更しない。

テスト用プログラムの出力には会話ID が無いので、入力に使った test.csv を
同じ条件(30 秒以下)で絞り込み、行の順番(idx)で会話ID / groupID を対応させる。

入力:
  PRED_CSV   テスト用プログラムの出力(idx, ref, pred)
  TEST_CSV   テスト用プログラムに入力した test.csv(make_2spk.py の出力)

出力:
  OUT_GROUP_CSV  1 行 = 1 グループ(会話ID, groupID, 全体 / IC1 / IC2 の正解文字数・誤り数・CER, ref, pred)
  OUT_FILE_CSV   1 行 = 1 会話(グループ数, 全体 / IC1 / IC2 の正解文字数・誤り数・CER)

話者ごとの CER:
  ref / pred を最初の [SC] で 2 つに分け、前を IC1、後ろを IC2 として別々に編集距離を数える
  (学習ラベルは「IC1 の発話 → [SC] → IC2 の発話」の順)。
  pred に [SC] が無いグループは、pred 全体を IC1、IC2 を空として数える(件数を表示する)。

CER の計算:
  テスト用プログラムの compute_metrics と同じく、タイムスタンプ・[SC] を除いて normalize() をかけ、
  文字単位の編集距離を数える。ファイルごとの CER は「誤り数の合計 / 正解文字数の合計」
  (evaluate の cer と同じ計算方法なので、全体 CER は test_cer と一致するはず)。

使い方:
  下の設定値を書き換えてから  python cer_per_file.py  で実行する。
  ※ テスト用プログラムを DEBUG=1(先頭 8 件だけ)で実行した結果には使えない。
"""
import csv
import re
import sys
from collections import defaultdict

# ===== 設定 =====
PRED_CSV = "test_predictions_timestamp.csv"
TEST_CSV = "/home/tsuda/M1_research/IC-SOT/make_data_名市大/out/2spk_timestamp/test.csv"
NORMALIZE_DIR = "/home/tsuda/M1_research/IC-SOT/make_data_名市大"   # cejc_normalize.py のあるフォルダ
MAX_DURATION = 30.0                     # テスト用プログラムの filter と同じ値にする
OUT_GROUP_CSV = "test_cer_per_group.csv"
OUT_FILE_CSV = "test_cer_per_file.csv"
# ================

sys.path.append(NORMALIZE_DIR)
from cejc_normalize import normalize  # noqa: E402

# <|0.00|> などのタイムスタンプトークン
TIMESTAMP_RE = re.compile(r"<\|\d+\.\d+\|>")


def to_eval_text(s):
    """to_text() で作った文字列から、compute_metrics と同じ評価用テキストを作る"""
    s = TIMESTAMP_RE.sub("", s).replace("[SC]", "")
    return normalize(s)


def split_speakers(s):
    """最初の [SC] で IC1 / IC2 に分け、それぞれ評価用テキストにする。[SC] が無ければ IC2 は空"""
    ic1, _, ic2 = s.partition("[SC]")
    return to_eval_text(ic1), to_eval_text(ic2)


def cer_str(errors, ref_len):
    return round(100 * errors / ref_len, 1) if ref_len else ""


def edit_distance(ref, hyp):
    """文字単位の編集距離(置換・脱落・挿入の合計)"""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1,              # 脱落
                         cur[j - 1] + 1,           # 挿入
                         prev[j - 1] + (r != h))   # 置換(一致なら 0)
        prev = cur
    return prev[-1]


def main():
    # テスト用プログラムと同じ条件で絞り込んだ test.csv の行(idx の順番と一致する)
    with open(TEST_CSV, encoding="utf-8", newline="") as f:
        meta = [r for r in csv.DictReader(f)
                if float(r["endTime"]) - float(r["startTime"]) <= MAX_DURATION]
    with open(PRED_CSV, encoding="utf-8-sig", newline="") as f:
        preds = list(csv.DictReader(f))

    if len(meta) != len(preds):
        sys.exit(f"行数が合いません: test.csv(絞り込み後) {len(meta)} 行 / 推論結果 {len(preds)} 行\n"
                 "TEST_CSV と MAX_DURATION がテスト用プログラムと同じか確認してください")

    group_rows = []
    keys = ["all", "ic1", "ic2"]
    per_file = defaultdict(lambda: {"groups": 0, **{f"{k}_{x}": 0 for k in keys for x in ("ref", "err")}})
    no_sc = 0
    for p in preds:
        m = meta[int(p["idx"])]
        # 全体: compute_metrics と同じく [SC] を無視してつなげたテキスト
        ref, hyp = to_eval_text(p["ref"]), to_eval_text(p["pred"])
        # 話者ごと: [SC] の前後で分けたテキスト
        ref1, ref2 = split_speakers(p["ref"])
        hyp1, hyp2 = split_speakers(p["pred"])
        if "[SC]" not in p["pred"]:
            no_sc += 1
        counts = {
            "all": (len(ref), edit_distance(ref, hyp)),
            "ic1": (len(ref1), edit_distance(ref1, hyp1)),
            "ic2": (len(ref2), edit_distance(ref2, hyp2)),
        }
        row = {"idx": p["idx"], "conversationID": m["conversationID"], "groupID": m["groupID"]}
        for k, (n, e) in counts.items():
            row.update({f"{k}_ref_len": n, f"{k}_errors": e, f"{k}_cer": cer_str(e, n)})
        row.update({"ref": p["ref"], "pred": p["pred"]})
        group_rows.append(row)

        s = per_file[m["conversationID"]]
        s["groups"] += 1
        for k, (n, e) in counts.items():
            s[f"{k}_ref"] += n
            s[f"{k}_err"] += e

    with open(OUT_GROUP_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        w.writeheader()
        w.writerows(group_rows)

    with open(OUT_FILE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conversationID", "groups"]
                   + [f"{k}_{x}" for k in keys for x in ("ref_chars", "errors", "cer")])
        for conv_id, s in sorted(per_file.items()):
            w.writerow([conv_id, s["groups"]]
                       + [v for k in keys for v in (s[f"{k}_ref"], s[f"{k}_err"],
                                                     cer_str(s[f"{k}_err"], s[f"{k}_ref"]))])

    def line(name, s):
        return (f"{name}: 全体 {cer_str(s['all_err'], s['all_ref']):>5}% / "
                f"IC1 {cer_str(s['ic1_err'], s['ic1_ref']):>5}% / "
                f"IC2 {cer_str(s['ic2_err'], s['ic2_ref']):>5}%")

    for conv_id, s in sorted(per_file.items()):
        print(line(f"{conv_id} ({s['groups']} グループ)", s))
    total = {k: sum(s[k] for s in per_file.values()) for k in next(iter(per_file.values()))}
    print(line("合計", total))
    if no_sc:
        print(f"[注意] pred に [SC] が無いグループ: {no_sc} 件(pred 全体を IC1 として数えた)")
    print(f"→ {OUT_FILE_CSV} / {OUT_GROUP_CSV}")


if __name__ == "__main__":
    main()