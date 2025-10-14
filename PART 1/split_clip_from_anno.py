import os
def organize_videos_from_txt(file_path: str) -> dict:
    video_data = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 

                filename = os.path.basename(line)

                if '_full_video_' not in filename:
                    print(f"Warning: The file name '{filename}' is not in the correct format, and it has been skipped.")
                    continue
                
                parts = filename.split('_full_video_')
                video_id = parts[0]
                
                # 1080×1920_001_A.json -> ['1080×1920', '001', 'A', '.json']
                suffix_parts = parts[1].split('_')
                if len(suffix_parts) < 3:
                    continue
                if video_id not in video_data:
                    video_data[video_id] = []
                video_data[video_id].append(filename)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while handling files:{e}")
        return {}

    return video_data

def organize_videos_from_folder(folder_path: str) -> dict:
    video_data = {}

    try:
        # Get all files in the given folder
        files = os.listdir(folder_path)
        
        for file in files:
            file_path = os.path.join(folder_path, file)

            # Only process JSON files
            if file.endswith(".json"):
                filename = os.path.basename(file_path)

                if '_full_video_' not in filename:
                    print(f"Warning: The file name '{filename}' is not in the correct format, and it has been skipped.")
                    continue
                
                # Split the filename to get the video_id
                parts = filename.split('_full_video_')
                video_id = parts[0]

                # Split suffix to ensure we have valid file naming format
                suffix_parts = parts[1].split('_')
                if len(suffix_parts) < 3:
                    continue
                
                # Organize video files by video_id
                if video_id not in video_data:
                    video_data[video_id] = []
                video_data[video_id].append(file_path)

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while handling files: {e}")
        return {}

    return video_data

import ffmpeg
import json
import subprocess

if __name__ == '__main__':
    max_video_res = 1920
    clip_save_dir = "path you want to save clips"
    org_video_dir = "org_video_download_from_youtube"
    merged_anno_dir = "path of merged_anno download from huggingface"
    json_data = organize_videos_from_folder(merged_anno_dir)
    for video_id, json_list in json_data.items():
        video_name = video_id + '.mp4'
        org_video_path = os.path.join(org_video_dir,video_name)
        if not os.path.exists(org_video_path):
            continue

        for json_file in json_list:
            clip_info = json.load(open(json_file,'r'))
            box = clip_info['bbox']
            height = clip_info['raw_video_height']
            width = clip_info['raw_video_width']
            box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            start_time = clip_info['org_start_seconds']
            duration = clip_info['video_total_duration']
            json_filename = json_file.split('/')[-1].split('.')[0]

            # Calculate crop parameters for FFmpeg's 'crop=w:h:x:y' filter
            crop_w = int(box[2] - box[0])
            crop_h = int(box[3] - box[1])
            crop_x = int(box[0])
            crop_y = int(box[1])
            if crop_w % 2 != 0:
                crop_w -= 1
            if crop_h % 2 != 0:
                crop_h -= 1
            if crop_x % 2 != 0:
                crop_x -= 1
            if crop_y % 2 != 0:
                crop_y -= 1

            # Define the output directory and file paths
            output_subdirectory = os.path.join(clip_save_dir, video_name.split('.')[0])
            os.makedirs(output_subdirectory, exist_ok=True)
            
            output_video_path = os.path.join(output_subdirectory, json_filename + '.mp4')
            output_audio_path = os.path.join(output_subdirectory, json_filename + '.wav') # Changed to .wav


            print(f" {video_name} : {json_filename}")
            
            max_scale = max(crop_w,crop_h)
            if max_scale > max_video_res:
                scale_w = int(crop_w / max_scale * max_video_res)
                scale_h = int(crop_h / max_scale * max_video_res)
            else:
                scale_w = crop_w
                scale_h = crop_h
            print(scale_w,scale_h)
            if scale_w % 2 != 0:
                scale_w -= 1
            if scale_h % 2 != 0:
                scale_h -= 1
            try:
                command = [
                    'ffmpeg',
                    '-y',                          
                    '-ss', str(start_time),        
                    '-i', org_video_path,          
                    '-t', str(duration),           
                    '-vf', f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={scale_w}:{scale_h}', 
                    '-an',                         # an = No Audio
                    '-q:v', '2',
                    '-r', '25',
                    # '-f', 'image2pipe',
                    # '-pix_fmt', 'bgr24',
                    '-vcodec', 'libx264', 
                    output_video_path,             #
                ]

                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"  The video clip was successfully created: {output_video_path}")
                command = [
                    'ffmpeg',
                    '-y',                         
                    '-i', org_video_path,          
                    '-ss', str(start_time),        
                    '-t', str(duration),           
                    
                    '-vn',                         # vn = No Video 
                    '-acodec', 'pcm_s16le',        #  WAV (16-bit)
                    '-ar', '16000',                #  16kHz
                    '-ac', '1',                    #  (mono)
                    output_audio_path              
                ]

                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"  Successfully extracted audio: {output_audio_path}")

            except subprocess.CalledProcessError as e:
                print(f"  An error occurred during processing {json_filename}")
                print(f"  FFmpeg:\n{' '.join(command)}\n")
                print(f"  FFmpeg Standard error output:\n{e.stderr}")


            
        