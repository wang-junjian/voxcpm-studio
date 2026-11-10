"""ASR 回环验证：用 FunASR paraformer-zh 把生成音频转回文字，计算字错率 CER。"""
import os, re, json, glob

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

OUT_DIR = os.path.abspath("outputs")
report = json.load(open(os.path.join(OUT_DIR, "report.json"), encoding="utf-8"))

PUNC = re.compile(r"[\s，。！？、；：\u201c\u201d\u2018\u2019（）()《》,.!?;:\"'\-—…·]")

def norm(s: str) -> str:
    return PUNC.sub("", s)

def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

print("加载 FunASR paraformer-zh ...", flush=True)
from funasr import AutoModel
asr = AutoModel(model="paraformer-zh", device="cpu", disable_update=True)
print("ASR 就绪", flush=True)

# 参考文本：场景 A/C 用原文本；场景 B 去掉括号内的音色描述（那段不该被念出来）
expect = {
    "A_basic_zh": report["cases"]["A_basic_zh"]["text"],
    "B_voice_design": re.sub(r"^\(.*?\)", "", report["cases"]["B_voice_design"]["text"]),
    "C_long_zh": report["cases"]["C_long_zh"]["text"],
}

rows = []
for name, ref in expect.items():
    wav = os.path.join(OUT_DIR, f"{name}.wav")
    if not os.path.exists(wav):
        continue
    res = asr.generate(input=wav, batch_size_s=300)
    hyp = res[0]["text"] if res else ""
    r, h = norm(ref), norm(hyp)
    d = edit_distance(r, h)
    cer = d / max(len(r), 1)
    rows.append({"case": name, "ref": ref, "hyp": hyp, "cer": round(cer, 4),
                 "correct": d == 0, "ref_len": len(r)})
    print(f"\n[{name}] CER={cer:.2%}")
    print(f"  参考: {ref}")
    print(f"  识别: {hyp}", flush=True)

summary = {
    "asr_model": "paraformer-zh",
    "cases": rows,
    "avg_cer": round(float(np.mean([r["cer"] for r in rows])), 4) if rows else None,
    "all_exact": all(r["correct"] for r in rows) if rows else False,
}
json.dump(summary, open(os.path.join(OUT_DIR, "asr_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n平均 CER: %.2f%% | 全部逐字正确: %s" % (summary["avg_cer"] * 100, summary["all_exact"]), flush=True)
print("ASR_DONE", flush=True)
