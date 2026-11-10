"""对单个音频做 ASR 回环，输出 CER。用法：python asr_one.py <wav> <参考文本>"""
import os
import re
import sys
import tempfile

# 先钉死临时目录：沙箱里 tempfile 的探测（写后删）会失败，torch 因此无法导入
_HERE = os.path.dirname(os.path.abspath(__file__))
tempfile.tempdir = os.path.join(_HERE, "tmpdir")
os.makedirs(tempfile.tempdir, exist_ok=True)

from funasr import AutoModel  # noqa: E402

P = re.compile(r"[\s，。！？、；：（）()《》,.!?;:\"'\u2018\u2019\u201c\u201d\-—…·]")


def lev(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i]
        for j in range(1, n + 1):
            cur.append(min(dp[j] + 1, cur[j - 1] + 1, dp[j - 1] + (a[i - 1] != b[j - 1])))
        dp = cur
    return dp[n]


def main():
    wav, ref = sys.argv[1], sys.argv[2]
    asr = AutoModel(model="paraformer-zh", device="cpu", disable_update=True)
    hyp = asr.generate(input=wav, batch_size_s=300)[0]["text"]
    a, b = P.sub("", ref), P.sub("", hyp)
    d = lev(a, b)
    print(f"文件: {wav}")
    print(f"参考: {ref}")
    print(f"识别: {hyp}")
    print(f"字数={len(a)} 编辑距离={d} CER={d / len(a):.2%}")


if __name__ == "__main__":
    main()
