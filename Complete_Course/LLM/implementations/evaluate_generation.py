"""
Simple generation evaluation helpers: exact match, token-F1, and a naive ROUGE-L.
This is for quick checks; use human evaluation for final quality.
"""
import re
from collections import Counter


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    return text.strip()


def tokens(text):
    return normalize(text).split()


def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)


def token_f1(pred, gold):
    p_tokens = tokens(pred)
    g_tokens = tokens(gold)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_tokens)
    rec = num_same / len(g_tokens)
    return 2 * (prec * rec) / (prec + rec)


def rouge_l(pred, gold):
    # very naive longest common subsequence length / reference length
    a = tokens(pred)
    b = tokens(gold)
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    if m == 0:
        return 0.0
    return lcs / m


if __name__ == '__main__':
    gold = "The cat sat on the mat."
    pred = "A cat was sitting on the mat"
    print('Exact match:', exact_match(pred, gold))
    print('Token F1:', token_f1(pred, gold))
    print('Rouge-L:', rouge_l(pred, gold))
