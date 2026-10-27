import json
import glob
import os

WEIGHT_DIR = "/home/tsuda/M1_research/IC-SOT/IC-SOT-weight"

# 一番ステップ数が大きいチェックポイントの trainer_state.json を読む
ckpts = sorted(glob.glob(f"{WEIGHT_DIR}/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
with open(os.path.join(ckpts[-1], "trainer_state.json")) as f:
    state = json.load(f)

print("best_model_checkpoint:", state["best_model_checkpoint"])
print("best_metric (CER):", state["best_metric"])

# 参考: エポックごとの dev CER
for log in state["log_history"]:
    if "eval_cer" in log:
        print(f"epoch {log['epoch']:.0f}  step {log['step']}  eval_cer {log['eval_cer']:.2f}")