import json
from collections import Counter

PATH = r"c:\Users\user\Documents\GitHub\Lip-Reading-Model\ALIGNMENT\cleaned_no_edits.json"
TARGET = 30.0

with open(PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

episodes = data.get('results', [])
num_episodes = len(episodes)
num_chunks = 0
num_sentences = 0
chunks_over = 0
sent_missing_ts = 0
approx_sentences = 0
oversized_by_episode = Counter()

for ep in episodes:
    title = ep.get('title', '<untitled>')
    chunks = ep.get('chunks', [])
    num_chunks += len(chunks)
    for ch in chunks:
        start = ch.get('start')
        end = ch.get('end')
        if start is None or end is None:
            # treat as oversized if durations unknown
            chunks_over += 1
            oversized_by_episode[title] += 1
        else:
            dur = end - start
            if dur > TARGET:
                chunks_over += 1
                oversized_by_episode[title] += 1
        sents = ch.get('sentences', [])
        num_sentences += len(sents)
        for s in sents:
            if s.get('start') is None or s.get('end') is None:
                sent_missing_ts += 1
            if s.get('approximate'):
                approx_sentences += 1

# top episodes with most oversized chunks
top_offenders = oversized_by_episode.most_common(10)

summary = {
    'episodes': num_episodes,
    'chunks': num_chunks,
    'sentences': num_sentences,
    'chunks_exceeding_{:.1f}s'.format(TARGET): chunks_over,
    'sentences_missing_timestamps': sent_missing_ts,
    'approximate_sentences_flagged': approx_sentences,
    'top_oversized_episodes': top_offenders
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
