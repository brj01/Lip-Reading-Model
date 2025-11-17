ALEF = "\u0627"
HEH = "\u0647"


def is_variant(word1, word2):
    """
    Return True if word1 and word2 are considered orthographic variants.
    This implementation keeps the simple Alef/Heh heuristics that were used previously,
    but without performing any CAMeL-based normalization.
    """
    w1 = " ".join(word1.split())
    w2 = " ".join(word2.split())
    if w1 == w2:
        return True
    if w1.replace(ALEF, "", 1) == w2 or w2.replace(ALEF, "", 1) == w1:
        return True

    def strip_end(token: str) -> str:
        if len(token) > 1 and token[-1] in (ALEF, HEH):
            return token[:-1]
        return token

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
    ops = align_words(r_words, h_words)
    subs = sum(op == "sub" for op, *_ in ops)
    deletions = sum(op == "del" for op, *_ in ops)
    insertions = sum(op == "ins" for op, *_ in ops)
    denominator = len(r_words)
    wer = (subs + deletions + insertions) / float(denominator) if denominator else float("nan")
    return wer


def align_words(ref_words, hyp_words):
    """Dynamic programming alignment with backtrace."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution_cost = 0 if is_variant(ref_words[i - 1], hyp_words[j - 1]) else 1
            best_cost = dp[i - 1][j - 1] + substitution_cost
            best_prev = (i - 1, j - 1, "ok" if substitution_cost == 0 else "sub")

            delete_cost = dp[i - 1][j] + 1
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_prev = (i - 1, j, "del")

            insert_cost = dp[i][j - 1] + 1
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_prev = (i, j - 1, "ins")

            dp[i][j] = best_cost
            back[i][j] = best_prev

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        prev_i, prev_j, op = back[i][j]
        ref_word = ref_words[i - 1] if i > 0 else ""
        hyp_word = hyp_words[j - 1] if j > 0 else ""
        if op == "del":
            hyp_word = ""
        elif op == "ins":
            ref_word = ""
        ops.append((op, ref_word, hyp_word))
        i, j = prev_i, prev_j

    return list(reversed(ops))


def format_alignment(ref, hyp):
    """Return a textual visualization of the alignment."""
    ops = align_words(ref.split(), hyp.split())
    ref_line = []
    hyp_line = []
    mark_line = []
    symbol_map = {"ok": "✓", "sub": "✗", "del": "−", "ins": "+"}
    for op, ref_word, hyp_word in ops:
        ref_line.append(ref_word or "—")
        hyp_line.append(hyp_word or "—")
        mark_line.append(symbol_map.get(op, "?"))

    spacer = " | "
    return (
        "REF: " + spacer.join(ref_line) + "\n"
        "HYP: " + spacer.join(hyp_line) + "\n"
        "OP : " + spacer.join(mark_line)
    )


if __name__ == "__main__":
    ref = "راح شوفك بكرة بالبيت"
    hyp = "رح شوفك بكرا بالبيت"
    wer_val = fair_wer(ref, hyp)
    print(f"Ref: {ref}")
    print(f"Hyp: {hyp}")
    print(f"Fair WER: {wer_val:.2f}")
    print()
    print(format_alignment(ref, hyp))
