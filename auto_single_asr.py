import multiprocessing
from functools import partial
import torch.multiprocessing as mp
import argparse
import whisper
import glob
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def find_videos(dataset_dir):
    video_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for mp4_file in Path(dataset_dir).rglob('*.mp4'):
            video_list.append(str(mp4_file))
    return video_list

def process_clip_dir(clip_dir):
    try:
        mp4_files = glob.glob(f'{clip_dir}/*.mp4', recursive=True)
        return mp4_files[0] if mp4_files else None
    except Exception as e:
        print(f"Error processing {clip_dir}: {e}")
        return None

def process_audio_file(args):
    data, save_dir, asr_model = args
    video_name = (data).split('/')[-1].replace('.wav', '.json')
    asr_json_path = os.path.join(save_dir, video_name)
    if os.path.exists(asr_json_path):
        return
    try:
        result = asr_model.transcribe(data)
        with open(asr_json_path, 'w') as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        print(f"Error processing {data}: {e}")
        return
    
def check_and_prepare_task(i_data):
    i, data = i_data
    json_path = os.path.join(save_dir, os.path.basename(data).replace('.wav', '.json'))
    if not os.path.exists(json_path) and '_000_' not in data:
        return (data, save_dir, asr_models[i % num_processes])
    return None

def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--total", type=int, default=12
    )
    parser.add_argument(
        "--index", type=int, default=0
    )
    parser.add_argument("--audio_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/asr/wav_0709.txt")
    parser.add_argument("--save_dir", type=str, default="./asr_result")
    args = parser.parse_args()
    return args
if __name__=='__main__':
    mp.set_start_method('spawn')
    opt = get_parse()
    num_worker = opt.total
    index = opt.index
    audio_list_path = opt.audio_list_path
    save_dir = opt.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    wav_list = []
    with open(audio_list_path, "r") as f:
        wav_list = f.read().splitlines()
    wav_list =  wav_list[index :: num_worker]
    print("wav_list",len(wav_list))
    num_processes = 4  
    asr_models = [whisper.load_model("large") for _ in range(num_processes)]
    print('model loaded')
    num_threads = 16
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tasks = executor.map(check_and_prepare_task, enumerate(wav_list))
        tasks_with_models = [t for t in tasks if t is not None]
    print('task to do',len(tasks_with_models))
    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap(process_audio_file, tasks_with_models), total=len(tasks_with_models), desc="Processing"):
            pass  
