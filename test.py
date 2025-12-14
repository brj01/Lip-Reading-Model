import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import os

# CSV and column name
csv_file = r"C:\Users\user\Documents\GitHub\Lip-Reading-Model\metadata_part2.csv"
audio_column = "path"
base_folder = r"C:\Users\user\Desktop\audio\audio"

df = pd.read_csv(csv_file)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
    audio_path = row[audio_column]

    # Append folder path at runtime if path is relative
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(base_folder, audio_path)

    if not os.path.exists(audio_path):
        print(f"File not found, skipping: {audio_path}")
        continue

    # Determine output WAV path
    wav_path = os.path.splitext(audio_path)[0] + ".wav"

    # Skip conversion if WAV already exists
    if not os.path.exists(wav_path):
        # Load audio with pydub
        audio = AudioSegment.from_file(audio_path)
        # Convert to mono and 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        # Export as WAV
        audio.export(wav_path, format="wav")

    # Update CSV path (optional: keep original filename, here we keep the WAV path)
    df.at[idx, audio_column] = wav_path

# Save updated CSV
df.to_csv(csv_file, index=False)

print("Audio processing completed!")
