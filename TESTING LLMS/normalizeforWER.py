import re
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
    normalize_ligature_ar
)

def normalize_arabic(text):
    """
    Normalize Arabic text using both custom and CAMeL tools:
    - Remove diacritics/tatweel
    - Normalize Alef/Hamza forms, ى->ي, ة->ه
    """
    text = text.strip()
    # Remove Arabic diacritics and tatweel
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]', '', text)
    # Apply CAMeL normalizers
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    text = normalize_ligature_ar(text)
    # Normalize Hamza variants manually
    text = text.replace('ؤ', 'و').replace('ئ', 'ي')
    return ' '.join(text.split())

def is_variant(word1, word2):
    """
    Return True if word1 and word2 are orthographic variants after normalization.
    For example, 'راح' vs 'رح' or 'بكرة' vs 'بكرا' should be True.
    """
    w1 = normalize_arabic(word1)
    w2 = normalize_arabic(word2)
    if w1 == w2:
        return True
    # Allow dropping a single Alef anywhere
    if w1.replace('ا', '', 1) == w2 or w2.replace('ا', '', 1) == w1:
        return True
    # Allow dropping final alef or heh
    def strip_end(w):
        if len(w) > 1 and w[-1] in ('ا', 'ه'):
            return w[:-1]
        return w
    if strip_end(w1) == strip_end(w2):
        return True
    return False

def fair_wer(ref, hyp):
    """
    Compute a 'fair' WER between reference and hypothesis strings.
    Returns WER = (sub + ins + del) / len(ref_words).
    """
    r_words = ref.split()
    h_words = hyp.split()
    n = len(r_words)
    # DP table: (n+1) x (m+1)
    dp = [[0]*(len(h_words)+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, len(h_words)+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, len(h_words)+1):
            cost = 0 if is_variant(r_words[i-1], h_words[j-1]) else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,            # deletion
                dp[i][j-1] + 1,            # insertion
                dp[i-1][j-1] + cost        # substitution (0 if variant match)
            )
    wer = dp[n][len(h_words)] / float(n) if n > 0 else float('nan')
    return wer

# Example usage:
if __name__ == "__main__":
    ref = "راح شوفك بكرة بالبيت"
    hyp = "رح شوفك بكرا بالبيت"
    wer_val = fair_wer(ref, hyp)
    print(f"Ref: {ref}")
    print(f"Hyp: {hyp}")
    print(f"Fair WER: {wer_val:.2f}")
