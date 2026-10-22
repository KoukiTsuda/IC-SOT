#!/usr/bin/env python3
"""
発話区間CSVから「発話グループ」を切り出し、グループごとの startTime / endTime を出力する。
luu_list.csv に載っている全会話の luu.csv を順に処理し、会話ごとに CSV を作る。

発話グループの定義:
  重複発話によって繋がっている一連の発話集合。
  無音位置、または重複のない発話境界(どの発話にもまたがれていない時刻)で分割する。

アルゴリズム:
  発話を startTime 順に並べ、現在のグループの最大 endTime より
  次の発話の startTime が前(= 重複している)ならば同じグループに入れる。
  そうでなければ(無音 or ちょうど接している境界)新しいグループを開始する。

出力(OUTPUT_DIR の下に会話ごと):
  groups/groups_{会話ID}.csv    1 行 = 1 グループ
  labeled/labeled_{会話ID}.csv  1 行 = 1 発話(groupID 付き)

使い方:
  下の設定値を書き換えてから  python utterance_groups.py  で実行する。

転記タグの定義
https://www2.ninjal.ac.jp/conversation/cejc/transcript.html
"""
import csv
import os

# ===== 設定 =====
LUU_LIST_CSV = "out/luu_list.csv"   # luu.csv のパス一覧(luu_list.py で作成)
OUTPUT_DIR = "out"       # 出力先フォルダ
GAP = 0.0                       # この秒数未満の無音はつなげる(0 = 重複時のみ連結)
# ================


def read_luu_list(path):
    """luu_list.csv から luu.csv のパスのリストを返す"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["luu_csv"] for r in csv.DictReader(f)]

# アノテーション CSV にはヘッダーが無いので、列名をこちらで付ける
input_fields = ["IC", "speaker", "startClock", "startTime", "endClock", "endTime",
          "durationClock", "duration", "text"]

def read_utterances(path):
    # CEJC の luu.csv は Shift-JIS(cp932)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, fieldnames=input_fields) 
        rows = []
        for r in reader:
            r["startTime"] = float(r["startTime"])
            r["endTime"] = float(r["endTime"])
            r["IC"] = r["IC"]
            r["speakerName"] = r["speaker"]  # 患者, 看護師 …
            rows.append(r)
    return rows


def group_utterances(rows, gap=0.0):
    """
    gap: 次の発話の開始が「現在のグループ終了 + gap」より前なら同じグループ。
         gap=0 のとき、厳密に重複している場合のみ連結(接しているだけの境界では分割)。
    戻り値: グループのリスト。各グループは発話行のリスト。
    """
    # 発話を startTime 順にソートする。startTime が同じ場合は endTime, IC の順でソート。
    rows = sorted(rows, key=lambda r: (r["startTime"], r["endTime"], r["IC"]))
    # 完成した発話グループを入れる
    groups = []
    # 作り途中の発話グループと、そのグループの最大 endTime
    cur, cur_end = [], None
    for r in rows:
        #startTime が現在のグループの最大終了時間 + gap より前なら同じグループに入れる
        if cur and r["startTime"] < cur_end + gap:
            #行ごと追加する
            cur.append(r)
            #現在のグループの最大終了時間を更新する
            cur_end = max(cur_end, r["endTime"]) #max() 大きい引数を返す
        else:
            #curが空でなければ、
            if cur:
                groups.append(cur)
            cur, cur_end = [r], r["endTime"]
    if cur:
        groups.append(cur)
    return groups


def summarize(groups, conv_id):
    """グループごとの区間の一覧を返す(どの発話がどのグループかは labeled CSV に書く)"""
    out = []
    for gid, g in enumerate(groups, start=1):
        out.append({
            "conversationID": conv_id,
            "groupID": gid,
            "startTime": min(r["startTime"] for r in g),
            "endTime": max(r["endTime"] for r in g),
        })
    return out


def process_one(luu_csv):
    """1 会話分の luu.csv をグループ分けして CSV を保存し、グループ数を返す"""
    # 拡張子の前後で分ける "2025-10-02_23.15.21.csv" → "2025-10-02_23.15.21"
    conv_id = os.path.splitext(os.path.basename(luu_csv))[0]

    #luu.csvのstartTimeとendTimeを文字列型からfloat型に変換して返す
    rows = read_utterances(luu_csv)
    # 中身が空の luu.csv(展開に失敗した 0 バイトのファイル等)は飛ばす
    if not rows:
        print(f"[警告] 発話がありません(スキップ): {luu_csv}")
        return conv_id, 0
    #発話グループの分割を行う
    groups = group_utterances(rows, gap=GAP)
    #グループ一覧の作成
    summary = summarize(groups, conv_id)

    # グループ一覧をCSVに保存
    fields = ["conversationID", "groupID", "startTime", "endTime"]
    group_csv = os.path.join(OUTPUT_DIR, "groups", f"groups_{conv_id}.csv")
    with open(group_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    
    # 各発話に groupID を付けたCSVを保存
    fields = ["conversationID", "groupID", "luuID", "startTime", "endTime", "IC", "text"]
    labeled_csv = os.path.join(OUTPUT_DIR, "labeled", f"labeled_{conv_id}.csv")
    with open(labeled_csv, "w", encoding="utf-8-sig", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=fields, extrasaction="ignore")  # ← lf、ignore必須
        w.writeheader()
        for gid, g in enumerate(groups, start=1):          # ← summaryではなくgroupsを走査
            for r in g:
                w.writerow({"conversationID": conv_id, "groupID": gid, **r})

    return conv_id, len(groups)


def main():
    luu_paths = read_luu_list(LUU_LIST_CSV)
    os.makedirs(os.path.join(OUTPUT_DIR, "groups"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labeled"), exist_ok=True)

    total = 0
    for i, luu_csv in enumerate(luu_paths, start=1):
        conv_id, n = process_one(luu_csv)
        total += n
        print(f"[{i}/{len(luu_paths)}] {conv_id}: {n} グループ")

    print(f"{len(luu_paths)} 会話 / 合計 {total} グループ → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
