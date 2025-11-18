# --- (Previous extract_key_frame and find_group_representative functions remain unchanged) ---
import cv2
import os
from deepface import DeepFace
from collections import defaultdict
import os
os.environ['DEEPFACE_HOME'] = '../weights'
import shutil

# --- 1. Function to extract key frames from a video (Code remains unchanged) ---
def extract_key_frame(video_path, output_dir="temp_frames"):
    """
    Extract a frame from the middle of the video and save it as an image.
    Returns the path of the extracted image.
    If the video cannot be opened or has no frames, returns None.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video file {video_path} has no frames.")
        cap.release()
        return None
        
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    ret, frame = cap.read()
    cap.release()

    if ret:
        base_name = os.path.basename(video_path)
        file_name, _ = os.path.splitext(base_name)
        frame_path = os.path.join(output_dir, f"{file_name}_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
    else:
        print(f"Warning: Unable to read frame from the middle of video {video_path}.")
        return None

# --- 2. Function to automatically find the representative face (Code remains unchanged) ---
import random
def find_group_representative(video_list, frame_paths_map, sample_size = 40):
    """
    Find the most representative video for the group using a "democratic voting" mechanism.
    Returns the path of the most representative video.
    """
    # If the video list exceeds the sample_size, perform sampling
    if len(video_list) > sample_size:
        video_list = random.sample(video_list, sample_size)
    scores = defaultdict(int)
    current_frame_paths = [frame_paths_map[vp] for vp in video_list if vp in frame_paths_map]

    for i in range(len(current_frame_paths)):
        for j in range(i, len(current_frame_paths)):
            img1_path = current_frame_paths[i]
            img2_path = current_frame_paths[j]
            
            try:
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name='ArcFace',
                    enforce_detection=False,
                    detector_backend='yolov8'
                )
                if result['verified']:
                    scores[img1_path] += 1
                    if i != j:
                        scores[img2_path] += 1
            except Exception as e:
                print(f"DeepFace comparison failed: {img1_path} vs {img2_path}. Error: {e}")

    if not scores:
        print("Warning: Unable to compute scores for any frame, all comparisons may have failed.")
        return None

    representative_frame_path = max(scores, key=scores.get)
    
    for video_path, frame_path in frame_paths_map.items():
        if frame_path == representative_frame_path:
            return video_path
    
    return None

import os
def organize_videos_from_txt(file_path: str) -> dict:
    """
    Reorganize a txt file containing mp4 file paths based on video ID and Person ID.

    Args:
        file_path (str): Path to the input txt file.

    Returns:
        dict: A dictionary with video IDs as keys. 
              Each video ID has another dictionary containing 'A' and 'B' keys, 
              which have lists of filenames corresponding to each Person ID.
              Example:
              {
                  'cnabn_cknaj': {
                      'A': ['cnabn_cknaj_full_video_1080x1920_001_A_01.mp4'],
                      'B': ['cnabn_cknaj_full_video_1920x1080_001_B_01.mp4']
                  },
                  ...
              }
    """
    video_data = {}

    try:
        # Open the file using 'utf-8' encoding to handle various path characters
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove trailing newlines and leading/trailing spaces
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Get the filename from the full path
                filename = os.path.basename(line)

                # 1. Split using '_full_video_' to get the video ID
                if '_full_video_' not in filename:
                    print(f"Warning: Filename '{filename}' is incorrectly formatted, skipping.")
                    continue
                
                parts = filename.split('_full_video_')
                video_id = parts[0]
                
                # 2. Parse Person ID from the second part of the filename
                # Example: 1080×1920_001_A_01.mp4 -> ['1080×1920', '001', 'A', '01.mp4']
                suffix_parts = parts[1].split('_')
                if len(suffix_parts) < 3:
                    print(f"Warning: Filename '{filename}' has an incorrect format, skipping.")
                    continue
                
                # Person ID is the third part
                person_id = suffix_parts[2]

                # 3. If the video ID is encountered for the first time, initialize it in the dictionary
                if video_id not in video_data:
                    video_data[video_id] = {'A': [], 'B': []}

                # 4. Add the filename to the corresponding Person ID list
                # Only 'A' and 'B' are considered; 'C', 'D', etc., will be ignored.
                if person_id in video_data[video_id]:
                    video_data[video_id][person_id].append(filename)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Unexpected error while processing file: {e}")
        return {}

    return video_data

# --- 3. Main process (Optimized) ---
def refine_ID(list_A, list_B, temp_frame_dir):
    # ------------------ Step 1: Extract key frames from all videos ------------------
    print("Step 1: Extracting key frames from all videos...")
    all_videos = list(set(list_A + list_B))
    video_to_frame_map = {}
    for video_path in all_videos:
        frame_path = extract_key_frame(video_path, temp_frame_dir)
        if frame_path:
            video_to_frame_map[video_path] = frame_path
    print("Key frame extraction completed.\n")

    # ------------------ Step 2: Automatically find the representative faces ------------------
    print("Step 2: Finding the representative face for list A...")
    rep_video_A_path = find_group_representative(list_A, video_to_frame_map)
    if not rep_video_A_path:
        print("Error: Could not find the representative face for list A. Program terminating.")
        return
    rep_frame_A_path = video_to_frame_map[rep_video_A_path]
    print(f"Representative face for list A found: {os.path.basename(rep_frame_A_path)}\n")

    print("Step 2: Finding the representative face for list B...")
    rep_video_B_path = find_group_representative(list_B, video_to_frame_map)
    if not rep_video_B_path:
        print("Error: Could not find the representative face for list B. Program terminating.")
        return
    rep_frame_B_path = video_to_frame_map[rep_video_B_path]
    print(f"Representative face for list B found: {os.path.basename(rep_frame_B_path)}\n")

    # ------------------ Step 3: Identify and separate incorrect videos (Added cross-validation) ------------------
    # Using more descriptive variable names
    found_B_in_list_A = []
    found_A_in_list_B = []
    unverified_videos = []  # For storing videos that do not match either A or B

    print("Step 3: Identifying videos in list A...")
    for video_path in list_A:
        if video_path not in video_to_frame_map:
            continue
        
        current_frame_path = video_to_frame_map[video_path]
        try:
            # Step 1: Compare with its own representative face (A)
            is_A = DeepFace.verify(
                img1_path=current_frame_path,
                img2_path=rep_frame_A_path,
                model_name='ArcFace', enforce_detection=False, detector_backend='yolov8'
            )['verified']

            # If not A, perform cross-validation
            if not is_A:
                print(f"  - Video {os.path.basename(video_path)} does not match representative A, cross-validating if it is B...")
                # Step 2: Compare with the representative face of B
                is_B = DeepFace.verify(
                    img1_path=current_frame_path,
                    img2_path=rep_frame_B_path,
                    model_name='ArcFace', enforce_detection=False, detector_backend='yolov8'
                )['verified']

                if is_B:
                    # Confirm it is B, add to the incorrect list
                    print(f"    -> Confirmed as B!")
                    found_B_in_list_A.append(video_path)
                else:
                    # Does not match A or B, might be a third person or recognition issue
                    print(f"    -> Does not match B, marked for verification.")
                    unverified_videos.append(video_path)

        except Exception as e:
            print(f"DeepFace verification failed (A group identification): {current_frame_path}. Error: {e}")
            unverified_videos.append(video_path)  # Add to unverified list if verification fails

    print("\nStep 3: Identifying videos in list B...")
    for video_path in list_B:
        if video_path not in video_to_frame_map:
            continue

        current_frame_path = video_to_frame_map[video_path]
        try:
            # Step 1: Compare with its own representative face (B)
            is_B = DeepFace.verify(
                img1_path=current_frame_path,
                img2_path=rep_frame_B_path,
                model_name='ArcFace', enforce_detection=False, detector_backend='yolov8'
            )['verified']

            # If not B, perform cross-validation
            if not is_B:
                print(f"  - Video {os.path.basename(video_path)} does not match representative B, cross-validating if it is A...")
                # Step 2: Compare with the representative face of A
                is_A = DeepFace.verify(
                    img1_path=current_frame_path,
                    img2_path=rep_frame_A_path,
                    model_name='ArcFace', enforce_detection=False, detector_backend='yolov8'
                )['verified']

                if is_A:
                    # Confirm it is A, add to the incorrect list
                    print(f"    -> Confirmed as A!")
                    found_A_in_list_B.append(video_path)
                else:
                    # Does not match A or B
                    print(f"    -> Does not match A, marked for verification.")
                    unverified_videos.append(video_path)

        except Exception as e:
            print(f"DeepFace verification failed (B group identification): {current_frame_path}. Error: {e}")
            unverified_videos.append(video_path)

    # ------------------ Step 4: Report results ------------------
    print("\n--- Final Identification Results ---")
    print(f"\nVideos found in list A and confirmed as belonging to B ({len(found_B_in_list_A)} videos):")
    for path in found_B_in_list_A:
        print(f" - {path}")

    print(f"\nVideos found in list B and confirmed as belonging to A ({len(found_A_in_list_B)} videos):")
    for path in found_A_in_list_B:
        print(f" - {path}")
        
    # Remove duplicates and print
    unique_unverified = list(set(unverified_videos))
    print(f"\nVideos that could not be identified (do not match A or B, or verification failed) ({len(unique_unverified)} videos):")
    for path in unique_unverified:
        print(f" - {path}")
        
    # You can clean the temporary folder if needed
    shutil.rmtree(temp_frame_dir)
    return found_B_in_list_A, found_A_in_list_B, unique_unverified
import argparse
def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--index", type=int, default=24)
    parser.add_argument("--num_Total", type=int, default=96)
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")
    parser.add_argument("--output", type=str, default="./ID_result")
    return parser.parse_args()

def split_dict(d, n):
    # Convert dictionary items into a list
    items = list(d.items())
    # Calculate the size of each chunk
    chunk_size = math.ceil(len(items) / n)
    # Split the dictionary into n chunks
    chunks = [dict(items[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]
    return chunks

import math
import json
import multiprocessing
import concurrent.futures

def process_video_chunk(video_id, AB, opt):
    try:
        list_A, list_B = AB['A'], AB['B']
        temp_frame_dir = "temp_frames_for_ID_correction"
        found_B_in_list_A, found_A_in_list_B, unique_unverified = refine_ID(list_A, list_B, temp_frame_dir)

        results = {
            "video_id": video_id,
            "found_B_in_list_A": found_B_in_list_A,
            "found_A_in_list_B": found_A_in_list_B,
            "unique_unverified": unique_unverified
        }

        save_dir = os.path.join(opt.output, f"{video_id}.json")
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")

if __name__ == '__main__':
    opt = get_parse()
    video_list_path = opt.video_list_path
    video_dict = organize_videos_from_txt(video_list_path)
    
    if not video_dict:
        print("Error: No video data found. Exiting...")
        exit(1)
    
    chunks = split_dict(video_dict, opt.num_Total)

    os.makedirs(opt.output, exist_ok=True)

    # Using ProcessPoolExecutor to process each chunk of data
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for video_id, AB in chunks[opt.index].items():
            futures.append(executor.submit(process_video_chunk, video_id, AB, opt))

        # Wait for all future tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions from the execution
            except Exception as e:
                print(f"Error in processing a future task: {e}")
