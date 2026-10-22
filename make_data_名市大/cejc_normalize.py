#!/usr/bin/env python3
"""
CEJC の長い発話単位(luu)CSV の転記テキストから転記タグを処理し、学習用テキストに変換する。
空白は削除し、句読点は残す(評価時は normalize() を正解・認識結果の両方にかける)。

処理順:
  1. @以降(転記コメント)を削除
  2. ＜咳＞ などの ＜＞ / <> で囲まれた部分を削除
  3. 括弧タグを一番内側から順に処理(下の TAG 規則)
  4. : % + を削除
  5. 空白を削除
  6. 記号のみになった発話は出力しない(DROP_EMPTY)
     判定には正規化(PUNCTUATION_RE / WHITESPACE_RE)を使う。特殊トークン <unknown> <sing> <beep> は記号扱いしない

タグ規則:
  (W 誤|正) (K 仮名|漢字) (Y 読み|表記) → 右を残す
  (G 口語|発音)                         → 左を残す
  (D ..) (F ..) (I ..) (B ..) (時間) (.) → もろとも削除
  (T ..) (C ..) (M ..)                   → 中身を残す(単独は削除)
  (L ..)                                 → 中身を残す、単独は削除、＃が混ざる場合 <unknown>
  (S ..)                                 → 中身を残す、単独は削除、(S ＊) は <sing>
  (U 候補1/候補2..)                      → 第一候補(空なら次の空でない候補)、＃が混ざる場合 <unknown>
  (X ..) (O ..)                          → 中身を出力、＃が混ざる場合 <unknown>
  (R ..)                                 → <beep>

使い方:
  utterance_groups.py を実行した後、下の設定値を確認して  python cejc_normalize.py  で実行する。
  groups_out/labeled/labeled_{会話ID}.csv を読み、groups_out/normalized/normalized_{会話ID}.csv を出力する。
  他のスクリプトからは  from cejc_normalize import convert, normalize  で 1 発話ずつ変換できる。
    convert(text)   : タグ処理後のテキスト(学習用)
    normalize(text) : 句読点・空白を削除したテキスト(評価用)

転記タグの定義
https://www2.ninjal.ac.jp/conversation/cejc/transcript.html
"""
import csv
import glob
import os
import re
import sys

# ===== 設定 =====
INPUT_DIR = "out/labeled"        # utterance_groups.py が出力した labeled_{会話ID}.csv のフォルダ
OUTPUT_DIR = "out/normalized"    # 変換結果 normalized_{会話ID}.csv の出力先
INPUT_ENCODING = "utf-8-sig"            # labeled_*.csv は utf-8(BOM付き)
DROP_EMPTY = True                       # 記号のみになった発話を出力しない
# ================


UNKNOWN, SING, BEEP = "<unknown>", "<sing>", "<beep>"
SPECIAL_TOKENS = [UNKNOWN, SING, BEEP]

PUNCTUATION_RE = re.compile(r'[、。,.．!?！？…‥";:；「」『』（）()［］\[\]{}<>＜＞／/]')
WHITESPACE_RE = re.compile(r"\s+")

# 一番内側のタグ: "(" + タグ名 or ポーズ長 + [" " + 括弧を含まない中身] + ")"
INNER_TAG_RE = re.compile(r"\(([A-Z]|[\d.]*)(?: ([^()]*))?\)")
ANGLE_RE = re.compile(r"＜[^＞]*＞|<[^>]*>")
SHARP = "＃"

# 正規化で <> が消えないよう、特殊トークンを私用領域の文字に退避する
_PROTECT = {tok: chr(0xE000 + i) for i, tok in enumerate(SPECIAL_TOKENS)}
_RESTORE = {v: k for k, v in _PROTECT.items()}



# 聞き取れない部分の記号。CEJC は全角 ＃、名市大のアノテーションは半角 #
# 場当たり的な対処なので考えとく
def _has_sharp(s):
    return "＃" in s or "#" in s


def _apply_tag(tag, content, warn):
    """一番内側のタグ 1 個を規則に従って置き換えた文字列を返す"""
    # ポーズ (0.736) / (.)
    if re.fullmatch(r"[\d.]*", tag):
        return ""
    if tag in "DFIB":
        return ""
    if tag in "TCM":
        return content
    if tag == "L":
        return UNKNOWN if _has_sharp(content) else content
    if tag == "S":
        return SING if content == "＊" else content
    if tag == "R":
        return BEEP
    if tag in "XO":
        return UNKNOWN if _has_sharp(content) else content
    if tag == "U":
        cands = content.split("/")
        # 第一候補が空(内側の (D ..) 等が消えた)なら次の空でない候補を使う
        chosen = next((c for c in cands if c.strip()), "")
        return UNKNOWN if _has_sharp(chosen) else chosen
    if tag in "WKYG":
        parts = content.split("|")
        if len(parts) != 2:
            warn(f"({tag} ..) の | が 1 個ではない: {content}")
            return content
        return parts[0] if tag == "G" else parts[1]
    warn(f"未定義のタグ ({tag} ..)")
    return content


def strip_tags(text, warn=lambda msg: None):
    """転記タグを処理した文字列を返す(句読点・空白はまだ残っている)"""
    text = text.split("@", 1)[0]
    text = ANGLE_RE.sub("", text)
    # 内側のタグから 1 個ずつ置き換える
    while True:
        m = INNER_TAG_RE.search(text)
        if not m:
            break
        text = text[:m.start()] + _apply_tag(m.group(1), m.group(2) or "", warn) + text[m.end():]
    if "(" in text or ")" in text:
        warn(f"括弧の対応が取れていない: {text}")
    return text.replace(":", "").replace("%", "").replace("+", "")


def normalize(text):
    """句読点と空白を削除する(特殊トークンは保護)。評価時と、空発話の判定に使う"""
    for tok, ch in _PROTECT.items():
        text = text.replace(tok, ch)
    text = PUNCTUATION_RE.sub("", text)
    text = WHITESPACE_RE.sub("", text)
    for ch, tok in _RESTORE.items():
        text = text.replace(ch, tok)
    return text


def convert(text, warn=lambda msg: None):
    """転記テキスト 1 発話を学習用テキストに変換する(空白は削除、句読点は残す)"""
    return WHITESPACE_RE.sub("", strip_tags(text, warn))


def process_one(in_csv, out_csv):
    """labeled_{会話ID}.csv 1 個を変換して保存し、(入力発話数, 出力発話数, 警告数) を返す"""
    fields = ["conversationID", "groupID", "luuID", "startTime", "endTime", "IC", "text"]
    n_in = n_out = n_warn = 0
    with open(in_csv, encoding=INPUT_ENCODING, newline="") as rf, \
         open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in csv.DictReader(rf):
            n_in += 1

            def warn(msg, r=r):
                nonlocal n_warn
                n_warn += 1
                print(f"[警告] {r['conversationID']}:{r['luuID']} {msg}", file=sys.stderr)

            text = convert(r["text"], warn)
            # 記号のみになった発話は除く(groupID は utterance_groups.py で全発話から決めてあるので影響しない)
            if DROP_EMPTY and not normalize(text):
                continue
            w.writerow({
                "conversationID": r["conversationID"],
                "groupID": r["groupID"],
                "luuID": r["luuID"],
                "startTime": r["startTime"],
                "endTime": r["endTime"],
                "IC": r["IC"],
                "text": text,
            })
            n_out += 1
    return n_in, n_out, n_warn


def main():
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "labeled_*.csv")))
    if not paths:
        sys.exit(f"labeled CSV が見つかりません: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_in = total_out = total_warn = 0
    for path in paths:
        # labeled_C001_001.csv → normalized_C001_001.csv
        conv_id = os.path.basename(path).removeprefix("labeled_").removesuffix(".csv")
        out_csv = os.path.join(OUTPUT_DIR, f"normalized_{conv_id}.csv")
        n_in, n_out, n_warn = process_one(path, out_csv)
        total_in += n_in
        total_out += n_out
        total_warn += n_warn

    print(f"{len(paths)} ファイル / 入力 {total_in} 発話 → 出力 {total_out} 発話 "
          f"(記号のみで除外 {total_in - total_out}, 警告 {total_warn}) → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
