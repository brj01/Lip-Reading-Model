from tqdm import tqdm
import json
from nemo.collections.asr.metrics.wer import word_error_rate
from tqdm import tqdm
import re

def calculate_wer(output_manifest):
    """
    Arguments
    ---------
    output_manifest: str
        path to the output manifest of the model inference

    Output
    ---------
    WER/CER
    """
    predictions = []
    target_transcripts = []
    with open(output_manifest, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)

    wer = word_error_rate(predictions, target_transcripts)
    print("wer : ", wer)
    cer = word_error_rate(predictions, target_transcripts, use_cer=True)
    print("cer : ", cer)
    return wer, cer
