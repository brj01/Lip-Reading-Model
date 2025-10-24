import os
import io
import cv2
import base64
import json
import torch
import ffmpeg
import numpy as np
from PIL import Image
import subprocess
import soundfile as sf
import threading
import time
import shutil
import tarfile
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys
from scenedetect import detect as scene_detect
from scenedetect import ContentDetector, HashDetector, HistogramDetector
from utils.box_utils import box_postporcess, complete_anyway, compute_center_point
from utils.crop_utils import compute_union, compute_crop_bounding_box, extend_size, video_crop, video_crop_all_frame
from utils.time_utils import time_format_diff, recompute_start, recompute_duration, scene_list_process
from utils.video_utils import ffmpeg_to_ndarray, ffmpeg_to_ndarray_only, ffmpeg_to_audio_binary, \
    save_ndarray_to_video, pyav_ndarray_to_binary, ffmpeg_to_full_audio_binary
import torchaudio
from pathlib import Path
import sys
import av
from speaker.speakerlab.bin.infer_diarization import Diarization3Dspeaker
from syncnet.syncnet import SyncNet_detector
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except:
        print("error")
        

def read_binary_video_data(binary_data):
    container = av.open(io.BytesIO(binary_data))

    frames = []
    for stream in container.streams.video:
        for frame in container.decode(stream):
            img = np.array(frame.to_image())
            frames.append(img)

    return np.array(frames)

def save_json(data, file_path, indent=4):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)
    except Exception as e:
        print(f"error on saving: {e}")

def filter(data):
    speaker_duration = {}
    speaker_count = {}
    for key, value in data.items():
        speaker = value["speaker"]
        duration = value["stop"] - value["start"]
        if speaker in speaker_duration:
            speaker_duration[speaker] += duration
        else:
            speaker_duration[speaker] = duration
        if speaker in speaker_count:
            speaker_count[speaker] += 1
        else:
            speaker_count[speaker] = 1

    print(speaker_duration,speaker_count)
    top_speakers = sorted(speaker_duration, key=speaker_duration.get, reverse=True)[:2]
    top_speakers_count = sorted(speaker_count, key=speaker_count.get, reverse=True)[:2]
    if len(top_speakers) == 2:
        if not sorted(top_speakers) == sorted(top_speakers_count):
        # if (top_speakers[0] != top_speakers_count[0]) or (top_speakers[1] != top_speakers_count[1]):
            print(top_speakers,top_speakers_count)
            return None
    first_start_speaker = None
    first_start_time = float('inf')
    for key, value in data.items():
        speaker = value["speaker"]
        if speaker in top_speakers and value["start"] < first_start_time:
            first_start_time = value["start"]
            first_start_speaker = speaker
    for key, value in data.items():
        speaker = value["speaker"]
        if speaker == first_start_speaker:
            value["speaker"] = "A"
        elif speaker in top_speakers:
            value["speaker"] = "B"
        else:
            value["speaker"] = chr(ord('A') + value["speaker"])
    return data

def extract_intervals(data):
    A = []
    speaker = []
    for key, value in data.items():
        A.append([value['start'], value['stop']])
        speaker.append(value['speaker'])
    return A,speaker

def find_overlapping_intervals(A,speaker, start, end):
    result = []
    result_speaker = []
    for i in range(len(A)):
        a, b = A[i]
        s = speaker[i]
        if a <= end and b >= start:
            overlap_start = max(a, start)
            overlap_end = min(b, end)
            result.append([overlap_start, overlap_end])
            result_speaker.append(s)
    return result,result_speaker
from pydub import AudioSegment
from collections import defaultdict
from scipy.io import wavfile
import torchaudio
import torch
from ultralytics import YOLO
def from_torchaudio_to_AudioSegment(wav_data, fs):
    if wav_data.dtype == torch.float32:  
        wav_data = torch.round(wav_data * 32767).clamp(-32768, 32767).short()
    audio = AudioSegment(
        wav_data.numpy().tobytes(),
        frame_rate=fs,
        sample_width=2,  # 16-bit
        channels=wav_data.shape[0]
    )
    return audio
def from_AudioSegment_to_torchaudio(audio):
    samples = audio.get_array_of_samples()  # 
    wav_data = torch.FloatTensor(samples).view(-1, audio.channels).T  
    wav_data = wav_data / 32767
    fs = audio.frame_rate
    return wav_data, fs
def split_audio_by_speakers(wav,fs, start, speaker_times):
    # audio = AudioSegment.from_file(input_file)
    extracted_audio = from_torchaudio_to_AudioSegment(wav, fs)
    silence = AudioSegment.silent(duration=len(extracted_audio))
    speaker_audio = silence
    for start_speak, end_speak in speaker_times:
        start_speak_ms = (start_speak - start) * 1000
        end_speak_ms = (end_speak - start) * 1000
        speaker_audio = speaker_audio[:start_speak_ms] + extracted_audio[start_speak_ms:end_speak_ms] + speaker_audio[end_speak_ms:]
    wav_from_AudioSegment,fs_from_AudioSegment = from_AudioSegment_to_torchaudio(speaker_audio)
    return wav_from_AudioSegment
def match_video_audio(video_data, audio_ids, threshold):
    result = {}
    result_score = {}
    matched_audios = set()
    
    candidates = []
    for vid, scores in video_data.items():
        for i, score in enumerate(scores):
            if threshold is not None:
                if score >= threshold:
                    candidates.append((score, vid, audio_ids[i]))
            else: 
                candidates.append((score, vid, audio_ids[i]))
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    for score, vid, aid in candidates:
        if vid not in result and aid not in matched_audios:
            result[vid] = aid
            result_score[vid] = score
            matched_audios.add(aid)
    
    
    for vid in video_data:
        if vid not in result:
            result[vid] = None
            result_score[vid] = 0
    
    cur_auid = audio_ids[0]
    if cur_auid == 'A':
        next_auid = 'B'
    elif cur_auid == 'B':
        next_auid = 'A'
    else:
        next_auid = None
    if len(video_data.keys()) == 2 and len(audio_ids) == 1 and next_auid is not None:
        for vid in video_data:
            if result[vid] == None:
                result[vid] = next_auid + '_None'
            
    return result, result_score

def interpolate_missing_values(list_A):
    if not list_A:
        return list_A
    
    valid_indices = [i for i, sublist in enumerate(list_A) if sublist != [-1, -1, -1, -1]]
    
    if not valid_indices:
        return list_A  
    
    for i in range(len(list_A)):
        if list_A[i] == [-1, -1, -1, -1]:
            nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
            list_A[i] = list_A[nearest_idx].copy()  
    
    return list_A
def resize_aspect_ratio(pil_img, target_long_side=640):
    
    width, height = pil_img.size
    
    if width > height:
        ratio = target_long_side / width
        new_width = target_long_side
        new_height = int(height * ratio)
    else:
        ratio = target_long_side / height
        new_height = target_long_side
        new_width = int(width * ratio)
    new_height = int(np.round(new_height / 32) * 32)  
    new_width = int(np.round(new_width / 32) * 32)  
    resized_img = pil_img.resize((new_width, new_height))
    
    return resized_img
def process_frame(img, transform):
    img_pil = Image.fromarray(img[:, :, [2, 1, 0]])
    img_tensor = transform(img_pil)
    return img_tensor
from torchvision import transforms
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
class Detector:
    def __init__(
        self,
        root_dir,
        record_dir,
        record_name,
        label_name,
        save_oss=None,
        save_name=None,
        syncnet_model_path = "../weights/syncnet_v2.model",
        s3fd_model_path = "../weights/sfd_face.pth",
        yolo_model_path = "../weights/yolo11x.pt",
        diarization_model_path = "../weights/speaker"
    ):
        if save_oss is not None and save_name is not None:
            self.save_oss_root = os.path.join(save_oss, save_name)

        self.root_dir = root_dir
        self.record_dir = record_dir
        self.record_name = record_name
        self.label_name = label_name

        os.makedirs(os.path.join(self.record_dir, label_name), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, "process_records"), exist_ok=True)

        self.record_path = os.path.join(os.path.join(self.record_dir, "process_records"), record_name)
        self.records = []
        if os.path.exists(self.record_path):
            with open(self.record_path, 'r') as f:
                for line in f:
                    self.records.append(line.strip())
        
        self.fps = 25
        self.diarization = Diarization3Dspeaker('cpu',model_cache_dir=diarization_model_path)
        self.yolo_model = YOLO(yolo_model_path)
        self.syncnet_model = SyncNet_detector(syncnet_model_path, s3fd_model_path, 'cuda')
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: resize_aspect_ratio(img, target_long_side=640)),
            ])
    

    def tar_data(self, tar_path, data_dict):
        with tarfile.open(tar_path, 'w') as tar:
            for key in data_dict:
                tarinfo = tarfile.TarInfo(name=key)
                tarinfo.size = len(data_dict[key])
                tar.addfile(tarinfo, io.BytesIO(data_dict[key]))


    def detect(self, vid_name, video_path, audio_path, clip_save_dir):
            
        if not os.path.exists(video_path):
            return
        ##### get basic information of video
        probe_res = ffmpeg.probe(video_path)
        video_info = next(s for s in probe_res['streams'] if s['codec_type'] == "video")
        raw_fps = eval(video_info['avg_frame_rate'])
        height = int(video_info['height'])
        width = int(video_info['width'])
        bitrate = int(probe_res['format']['bit_rate'])
        duration = float(probe_res['format']['duration'])
        print(vid_name, raw_fps, height, width, bitrate, duration, bitrate / np.sqrt(height*width))

        if audio_path is None:
            audio_binary_io, full_audio_binary = ffmpeg_to_full_audio_binary(video_path)
        else:
            audio_binary_io, full_audio_binary = ffmpeg_to_full_audio_binary(audio_path)
        wav_data, fs = torchaudio.load(audio_binary_io)  

        
        output_field_labels = self.diarization(wav_data)
        out_json = {}
        for seg in output_field_labels:
            seg_st, seg_ed, cluster_id = seg
            item = {
                'start': seg_st,
                'stop': seg_ed,
                'speaker': cluster_id,
            }
            segid = str(round(seg_st, 3))+'_'+str(round(seg_ed, 3))
            out_json[segid] = item

        speaker_json = filter(out_json)
        if speaker_json is None:
            print('flitered speaker is None')
            return
        A, speaker = extract_intervals(speaker_json)


        

        scene_list = scene_detect(video_path, HistogramDetector()) # HistogramDetector, ContentDetector
        scene_list = scene_list_process(scene_list)
        for idx, scene in enumerate(scene_list):
            print(scene)
            start = scene[0].get_timecode()
            end = scene[1].get_timecode()
            start_seconds = scene[0].get_seconds()
            end_seconds = scene[1].get_seconds()

            start_sample = int(start_seconds * fs)
            end_sample = int(start_sample * fs)
            if end_sample > wav_data.shape[1]:
                end_sample = wav_data.shape[1]
            clip_wav_data = wav_data[:, start_sample:end_sample]

            duration = end_seconds - start_seconds
            clip_video_duration = end_seconds - start_seconds
            overlapping_intervals, result_speaker = find_overlapping_intervals(A, speaker, start_seconds, end_seconds)
            if len(result_speaker) == 0:
                continue

            final_duration = {}
            for i in range(len(overlapping_intervals)):
                speak_id = result_speaker[i]
                final_duration[speak_id] = 0
            for i in range(len(overlapping_intervals)):
                speak_id = result_speaker[i]
                duration = overlapping_intervals[i][1] - overlapping_intervals[i][0]
                final_duration[speak_id] += duration
            
            clip_speakers = defaultdict(list)
            for i in range(len(overlapping_intervals)):
                clip_speakers[result_speaker[i]].append(overlapping_intervals[i])
            

            diff, dur_str = time_format_diff(start, end)

            if diff < 3:
                ##### NOTE: clip short than 3s
                continue

            print(idx, start, end, dur_str)

            frames = ffmpeg_to_ndarray_only(video_path, height, width, start, dur_str)

            ###
            batch_frames = frames  
            batch_images = [None] * len(batch_frames)  
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = []
                for idx_frame, img in enumerate(batch_frames):
                    future = executor.submit(process_frame, img, self.transform)
                    futures.append((idx_frame, future))
                for idx_frame, future in futures:
                    batch_images[idx_frame] = future.result()
            resized_frame = np.array([np.array(img) for img in batch_images])
            resized_height, resized_width = resized_frame.shape[-3], resized_frame.shape[-2]
            frames_vid_binary = pyav_ndarray_to_binary(resized_frame, [resized_height, resized_width])
            

            # video = base64.b64encode(frames_vid_binary).decode("utf-8")
            # video = base64.b64decode(video)
            video_frames = read_binary_video_data(frames_vid_binary)
            batch_size = 32
            boxes_list = []

            for i in range(0, len(video_frames), batch_size):
                batch_frames = video_frames[i:i + batch_size]  
                batch_tensor = torch.from_numpy(batch_frames).to('cuda')   # b, h, w, 3
                batch_tensor = batch_tensor.permute(0, 3, 1, 2).float() / 255.0  # b,3, h, w
                try:
                    body_results = self.yolo_model(batch_tensor, classes=[0], conf=0.4, iou=0.3, verbose=False)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")  
                resized_h , resized_w = batch_tensor.shape[-2], batch_tensor.shape[-1]
                for body_result in body_results:
                    boxes = body_result.boxes.xywh.cpu()
                    boxes_list.append(boxes/torch.Tensor([resized_w, resized_h, resized_w, resized_h]))
    
            raw_yolo_boxes_list = [list(boxes.cpu().numpy().tolist()) for boxes in boxes_list]
            phrases_list = raw_yolo_boxes_list

            win_size = 5
            blank_start_frame_num = 0
            blank_end_frame_num = 0

            print("yolo postprocessing...")
            ##### NOTE: compute blank box from start
            for i in range(len(raw_yolo_boxes_list)-win_size):
                flag = True
                for w in range(win_size):
                    if len(raw_yolo_boxes_list[i + w]) == 0:
                        flag = False
                        break
                if flag:
                    blank_start_frame_num = i
                    break
            if blank_start_frame_num == len(frames):
                ##### NOTE: no human or face in video
                print("return, because no human or face in video")
                continue

            ##### NOTE: compute blank box from end
            for i in range(len(raw_yolo_boxes_list)-1, win_size-2, -1):
                flag = True
                for w in range(win_size):
                    if len(raw_yolo_boxes_list[i - w]) == 0:
                        flag = False
                        break
                if flag:
                    blank_end_frame_num = len(raw_yolo_boxes_list) - i
                    break

            blank_frame_num = blank_start_frame_num + blank_end_frame_num

            left_frames_num = len(raw_yolo_boxes_list) - blank_frame_num

            boxes_list = raw_yolo_boxes_list[blank_start_frame_num:len(raw_yolo_boxes_list)-blank_end_frame_num]
            boxes_list = box_postporcess(boxes_list, width, height)

            all_res_box_list = []
            for boxes in boxes_list:
                res_box_list = []
                for box in boxes:
                    res_box_list.append(list(box))
                all_res_box_list.append(res_box_list)
            boxes_list = all_res_box_list
            ####

            win_size = 5
            blank_start_frame_num = 0
            blank_end_frame_num = 0

            print("gdino postprocessing...")
            ##### NOTE: compute blank box from start
            for i in range(len(phrases_list)-win_size):
                flag = True
                for w in range(win_size):
                    if len(phrases_list[i + w]) == 0:
                        flag = False
                        break
                if flag:
                    blank_start_frame_num = i
                    break
            if blank_start_frame_num == frames.shape[0]:
                ##### NOTE: no human or face in video
                print("continue, because no human or face in video")
                continue

            ##### NOTE: compute blank box from end
            for i in range(len(phrases_list)-1, win_size-2, -1):
                flag = True
                for w in range(win_size):
                    if len(phrases_list[i - w]) == 0:
                        flag = False
                        break
                if flag:
                    blank_end_frame_num = len(phrases_list) - i
                    break

            blank_frame_num = blank_start_frame_num + blank_end_frame_num

            left_frames_num = len(phrases_list) - blank_frame_num

            if left_frames_num < 75:
                ##### NOTE: clip short than 3s
                print("continue, because valid video length is less than 3s")
                continue
            
            for person_num in range(len(boxes_list)):
                boxes_list[person_num] = interpolate_missing_values(boxes_list[person_num])
            boxes_list, boxes_list_raw, start_end_list, size_list = compute_union(boxes_list)
            boxes_list, boxes_list_raw, start_end_list = extend_size(boxes_list, boxes_list_raw, start_end_list, size_list, height, width)
            
            bad_box = False
            for box in boxes_list:
                if -1 in box:
                    bad_box = True
            if bad_box:
                continue
            print(boxes_list,start_end_list)
            print("yolo postprocess end, total: {} boxes left (including possible union of multi boxes)".format(len(boxes_list)))
            print('___________________')

            if len(boxes_list) == 0:
                continue
            clip_audios = []
            audio_video_rate = int(fs / self.fps)
            for item in overlapping_intervals:
                start_sample = int(item[0] * fs / audio_video_rate) * audio_video_rate
                end_sample = int(item[1] * fs / audio_video_rate) * audio_video_rate
                if end_sample > wav_data.shape[1]:
                    end_sample = wav_data.shape[1]
                clip_audios.append(wav_data[:, start_sample:end_sample])
            
            conf_map = {}
            video_array_data = {}
            audio_wav_data = {}
            video_box_data = {}
            sync_score_map = {}
            wav_binary_all = [[io.BytesIO() for i in range(len(overlapping_intervals))] for j in range(len(boxes_list))]
            for pidx, data_item in enumerate(zip(boxes_list, start_end_list)):
                box, startend = data_item[0], data_item[1]
                conf_map[pidx] = []
                frames_of_curbox, expand_res_frames, res_size = video_crop_all_frame(frames, box)
                video_array_data[pidx] = (expand_res_frames, res_size)
                video_box_data[pidx] = box
                sync_score_map[pidx] = []
                wav_binary_ios = wav_binary_all[pidx] #[io.BytesIO() for i in range(len(overlapping_intervals))]
                for i in range(len(overlapping_intervals)):
                    start_in_clip, end_in_clip = overlapping_intervals[i][0] - start_seconds, overlapping_intervals[i][1] - start_seconds
                    if end_in_clip - start_in_clip < 0.5:
                        continue

                    start_sample = int(overlapping_intervals[i][0] * fs / audio_video_rate) * audio_video_rate
                    end_sample = int(overlapping_intervals[i][1] * fs / audio_video_rate) * audio_video_rate
                    if end_sample > wav_data.shape[1]:
                        end_sample = wav_data.shape[1]
                    clip_audio = wav_data[:, start_sample:end_sample]

                    audio_np = clip_audio.numpy().squeeze().T  
                    audio_np = np.round(audio_np * 32767).astype(np.int16)
                    
                    wav_binary_ios[i].truncate(0)  
                    wavfile.write(wav_binary_ios[i], fs, audio_np.squeeze())  
                    wav_binary_ios[i].seek(0)  
                    clip_audio_binary = wav_binary_ios[i].read()

                    start_frame = int(start_in_clip * fs / audio_video_rate)
                    end_frame = int(end_in_clip * fs / audio_video_rate)
                    if end_frame > frames.shape[0]:
                        end_frame = frames.shape[0]
                    startend = (start_frame,end_frame)
                    frames_of_box_x_and_clip_y = np.array(frames_of_curbox[start_frame:end_frame])
                    clip_audio_data, sr = sf.read(io.BytesIO(clip_audio_binary))
                    avoff, conf, dist = self.syncnet_model.inference(frames_of_box_x_and_clip_y, clip_audio_data, sr)


                    # for server
                    # conf, dist, avoff = syncnet_call_func(box_x_and_clip_y_vid_binary, clip_audio_binary)
                    # if conf is None:
                    #     for ii in range(20):
                    #         conf, dist, avoff = syncnet_call_func(box_x_and_clip_y_vid_binary, clip_audio_binary)
                    #         time.sleep(4)
                    #         if conf is not None:
                    #             break


                    if conf is None:
                        conf_map[pidx].append(0)
                        sync_score_map[pidx].append((0, 0, 0))
                    else:
                        conf_map[pidx].append(conf)
                        sync_score_map[pidx].append((conf, dist, avoff))
            # you can add a syncnet threshold there
            # threshold = 3
            threshold = None
            print(sync_score_map,'sync_score_map')
            print(overlapping_intervals)
            matches, sync_result_score = match_video_audio(conf_map, result_speaker, threshold)
            blank_start_sample = blank_start_frame_num * audio_video_rate
            blank_end_sample = (len(phrases_list)-blank_end_frame_num) * audio_video_rate
            clip_speaker_num = 0
            for video_id, audio_id in matches.items():
                if audio_id is not None:
                    clip_speaker_num += 1
            if clip_speaker_num == 0:
                continue
            for video_id, audio_id in matches.items():
                clip_wav_data_of_cur_speaker = split_audio_by_speakers(clip_wav_data,fs, start_seconds, clip_speakers[audio_id])
                video_name = vid_name.split(".mp4")[0] + "_{:03d}_{}_{:02d}.mp4".format(idx, audio_id, video_id)
                label_name = vid_name.split(".mp4")[0] + "_{:03d}_{}.json".format(idx, audio_id)
                label_path = os.path.join(self.record_dir, self.label_name, label_name)
                audio_name = vid_name.split(".mp4")[0] + "_{:03d}_{}.wav".format(idx, audio_id)
                tar_name = vid_name.split(".mp4")[0] + "_{:03d}_{}.tar".format(idx, audio_id)
                tar_path = os.path.join(clip_save_dir, tar_name)


                cur_video_array_data = video_array_data[video_id][0][blank_start_frame_num:len(phrases_list)-blank_end_frame_num]
                res_size = video_array_data[video_id][1]
                cur_box_start, cur_box_end = start_end_list[video_id][0] , start_end_list[video_id][1]
                cur_video_array_data = cur_video_array_data[cur_box_start:cur_box_end]
                cur_video_binary_data = pyav_ndarray_to_binary(cur_video_array_data, res_size)


                cur_audio_wav_data = clip_wav_data_of_cur_speaker[:, blank_start_sample:blank_end_sample]
                cur_audio_wav_data = cur_audio_wav_data[:,cur_box_start * audio_video_rate:cur_box_end * audio_video_rate]
                audio_np = cur_audio_wav_data.numpy().squeeze().T 
                audio_np = np.round(audio_np * 32767).astype(np.int16)
                wav_binary_io = io.BytesIO()
                wav_binary_io.seek(0)  
                wavfile.write(wav_binary_io, fs, audio_np.squeeze())  
                wav_binary_io.seek(0)  
                cur_audio_binary_data = wav_binary_io.read()


                
                start_new = start_seconds + (blank_start_frame_num + start_end_list[video_id][0]) / self.fps #recompute_start(start, (blank_start_frame_num + startend[0]) / self.fps)
                dur_str_new = (start_end_list[video_id][1] - start_end_list[video_id][0]) / self.fps #recompute_duration(dur_str, (blank_frame_num + len(frames) - startend[1]) / self.fps)
                box = video_box_data[video_id]
                label_dict = {
                    "num_frames": len(cur_video_array_data),
                    "is_talking": 1,
                    "clip_speaker_num": clip_speaker_num,
                    "clear": bitrate / np.sqrt(height*width),
                    "video_name": vid_name.split(".mp4")[0],
                    "video_total_duration": clip_video_duration,
                    "start_new": start_new,
                    "org_start_seconds": start_seconds,
                    "duration_new": dur_str_new,
                    "bbox": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
                    "conf": sync_result_score[video_id],
                    "sync": sync_score_map,
                    "speaker": result_speaker,
                    "raw_video_height": height,
                    "raw_video_width": width,
                    "clip_video_height": res_size[0],
                    "clip_video_width": res_size[1],
                }
                with open(label_path, 'w') as f:
                    json.dump(label_dict, f, indent=4, ensure_ascii=False)

                data_dict = {
                    video_name: cur_video_binary_data,
                    audio_name: cur_audio_binary_data,
                    label_name: json.dumps(label_dict).encode(),
                }

                self.tar_data(tar_path, data_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/video_json/batch2_video0/batch2_video0_0.json")
parser.add_argument("--batch_num", type=str, default="batch2")
args = parser.parse_args()
file_path = args.file_path
batch_num = args.batch_num
file_name = file_path.split("/")[-1].replace(".json", "")
record_name = file_name + ".txt"

root_dir = f"../output/{batch_num}/save_dir/"
record_dir = f"../output/{batch_num}/records/"
label_name = "video_labels"


detector = Detector(
    root_dir=root_dir,
    record_dir=record_dir,
    record_name=record_name,
    label_name=label_name,
)


with open(file_path, 'r') as f:
    data_list = json.load(f)
for data in data_list:
    video_path = data
    if video_path in detector.records:
        continue
    temp_raw_dir = os.path.join(detector.root_dir, "raw_video")
    clip_save_dir = os.path.join(detector.root_dir, "processed_clip")
    os.makedirs(temp_raw_dir, exist_ok=True)
    os.makedirs(clip_save_dir, exist_ok=True)
    video_uid = video_path.split('/')[-1].split('.mp4')[0]
    vid_name_raw = vid_name = video_uid + ".mp4"
    # video_path = os.path.join(temp_raw_dir, vid_name_raw)
    audio_path = None
    print(video_path)

    detector.detect(vid_name, video_path, audio_path, clip_save_dir)
    if os.path.exists(video_path):
        os.remove(video_path)
    if not audio_path is None:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    write_cmd = "echo '{}' >> {}".format(video_path, detector.record_path)
    os.system(write_cmd)
    detector.records.append(video_path)
    breakpoint()

    try:
        detector.detect(vid_name, video_path, audio_path, clip_save_dir)

        if os.path.exists(video_path):
            os.remove(video_path)
        if not audio_path is None:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        write_cmd = "echo '{}' >> {}".format(video_path, detector.record_path)
        os.system(write_cmd)
        detector.records.append(video_path)

    except Exception as e:
        print("Error start: " + "##########" * 5)
        print(e)
        print("Error end: " + "##########" * 5)

        if os.path.exists(video_path):
            os.remove(video_path)
        if not audio_path is None:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # write_cmd = "echo '{}' >> {}".format(oss_path, detector.record_path)
        # os.system(write_cmd)
        # detector.records.append(oss_path)
        continue
