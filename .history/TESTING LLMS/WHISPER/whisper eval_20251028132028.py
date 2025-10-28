import re
import numpy as np
from typing import List, Tuple

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for comparison:  
    – Remove diacritics  
    – Replace various Alef forms with a standard Alef  
    – Remove tatweel (ـ)  
    – Remove punctuation  
    – Normalize spaces and lowercase (if relevant)  
    """
    # remove tashkeel (diacritics)  
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
    # unify Alef variants  
    text = re.sub(r'[إأآا]', 'ا', text)
    # remove tatweel
    text = text.replace('ـ', '')
    # remove punctuation (you may expand this set)
    text = re.sub(r'[!?،؛:…\.,\"«»\(\)\[\]{}]', ' ', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def levenshtein(a: List[str], b: List[str]) -> Tuple[int, int, int]:
    """
    Compute insertions (I), deletions (D), substitutions (S)
    between list of tokens a (reference) and b (hypothesis)
    using dynamic programming.
    Returns (S, D, I)
    """
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=int)
    for i in range(1, n+1):
        dp[i,0] = i
    for j in range(1, m+1):
        dp[0,j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i,j] = dp[i-1,j-1]
            else:
                dp[i,j] = min(
                    dp[i-1,j] + 1,    # deletion
                    dp[i,j-1] + 1,    # insertion
                    dp[i-1,j-1] + 1   # substitution
                )
    # back-trace to compute S, D, I  
    i, j = n, m
    S = D = I = 0
    while i>0 or j>0:
        if i>0 and j>0 and a[i-1] == b[j-1]:
            i, j = i-1, j-1
        elif j>0 and (i==0 or dp[i,j-1] < dp[i-1,j] and dp[i,j-1] <= dp[i-1,j-1]):
            I += 1
            j -= 1
        elif i>0 and (j==0 or dp[i-1,j] < dp[i,j-1] and dp[i-1,j] <= dp[i-1,j-1]):
            D += 1
            i -= 1
        else:
            S += 1
            i -= 1
            j -= 1
    return S, D, I

def compute_wer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_arabic(reference).split()
    hyp_norm = normalize_arabic(hypothesis).split()
    S, D, I = levenshtein(ref_norm, hyp_norm)
    N = len(ref_norm)
    return (S + D + I) / N if N>0 else float('nan')

def compute_cer(reference: str, hypothesis: str) -> float:
    # character‐level
    ref_norm = list(normalize_arabic(reference).replace(' ', ''))
    hyp_norm = list(normalize_arabic(hypothesis).replace(' ', ''))
    S, D, I = levenshtein(ref_norm, hyp_norm)
    N = len(ref_norm)
    return (S + D + I) / N if N>0 else float('nan')

def compute_ser(references: List[str], hypotheses: List[str]) -> float:
    """
    Sentence Error Rate = proportion of sentences where hypothesis != reference
    after normalization.
    """
    assert len(references)==len(hypotheses)
    num_errors = sum(1 for r,h in zip(references, hypotheses)
                     if normalize_arabic(r).strip() != normalize_arabic(h).strip())
    return num_errors / len(references) if references else float('nan')

# Example usage:
if __name__ == "__main__":
    ref = "مريم : هاي رنا. عفوا تأخرت لأوصل، بس طولت للقيت سرفيس."
    hyp = "هاي رنا عفوًا تأخرت لوصل بس طولت لقيت سرفيس."
    print("WER:", compute_wer(ref, hyp))
    print("CER:", compute_cer(ref, hyp))
    print("SER:", compute_ser([ref], [hyp]))
