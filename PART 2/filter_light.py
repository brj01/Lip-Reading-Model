import cv2
import numpy as np
import os
import argparse
from multiprocessing import Pool, Lock, Manager
from tqdm import tqdm

lock = Lock()

def calculate_luminance_score(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("need RGB")
    
    R = frame[:, :, 0].astype(np.float32)
    G = frame[:, :, 1].astype(np.float32)
    B = frame[:, :, 2].astype(np.float32)
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.mean(luminance)

def has_over_bright_or_dark_sequence(video_path, threshold_low=10, threshold_high=210, max_consecutive=15):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path, "Error"

        consecutive_abnormal = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            luminance = calculate_luminance_score(frame_rgb)
            if luminance < threshold_low or luminance > threshold_high:
                consecutive_abnormal += 1
                if consecutive_abnormal > max_consecutive:
                    cap.release()
                    return video_path, "Abnormal"
            else:
                consecutive_abnormal = 0
        cap.release()
        return video_path, "Normal"
    except Exception as e:
        return video_path, "Error"

def process_and_write(args):
    video_path, output_path = args
    result = has_over_bright_or_dark_sequence(video_path)
    with lock:
        with open(output_path, "a") as fout:
            fout.write(f"{result[0]}\t{result[1]}\n")
    return result

def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--total", type=int, default=12)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=40, help="The number of videos processed in parallel")
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")
    parser.add_argument("--save_dir", type=str, default="./light_result")
    return parser.parse_args()

def main():
    opt = get_parse()
    num_worker = opt.total
    index = opt.index
    num_proc = opt.num_proc
    video_list_path = opt.video_list_path
    save_dir = opt.save_dir

    os.makedirs(save_dir, exist_ok=True)

    with open(video_list_path, "r") as f:
        video_list = f.read().splitlines()

    output_path = os.path.join(save_dir, f"video_list0707_{index}_{num_worker}.txt")

    processed_set = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                path = line.strip().split('\t')[0]
                processed_set.add(path)

    sub_video_list = [v for i, v in enumerate(video_list) if i % num_worker == index and v not in processed_set]
    print(f"The number of videos to be processed: {len(sub_video_list)}")

    args_list = [(video, output_path) for video in sub_video_list]

    with Pool(processes=num_proc) as pool:
        for _ in tqdm(pool.imap_unordered(process_and_write, args_list), total=len(args_list)):
            pass  #

if __name__ == "__main__":
    main()
