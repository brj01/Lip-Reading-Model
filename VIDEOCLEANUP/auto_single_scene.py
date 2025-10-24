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
# target_path = str(Path(__file__).parent.parent)  # 根据实际路径调整
# sys.path.insert(0, target_path)
from speaker.speakerlab.bin.infer_diarization import Diarization3Dspeaker
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 格式错误")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def save_json(data, file_path, indent=4):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

def filter(data):
    # 计算每个speaker的总说话时间
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
    # 找到说话时间最长的两个speaker
    top_speakers = sorted(speaker_duration, key=speaker_duration.get, reverse=True)[:2]
    top_speakers_count = sorted(speaker_count, key=speaker_count.get, reverse=True)[:2]
    if len(top_speakers) == 2:
        if not sorted(top_speakers) == sorted(top_speakers_count):
        # if (top_speakers[0] != top_speakers_count[0]) or (top_speakers[1] != top_speakers_count[1]):
            print(top_speakers,top_speakers_count)
            return None
    # 找到最先开始说话的speaker
    first_start_speaker = None
    first_start_time = float('inf')
    for key, value in data.items():
        speaker = value["speaker"]
        if speaker in top_speakers and value["start"] < first_start_time:
            first_start_time = value["start"]
            first_start_speaker = speaker
    # 将最先开始说话的speaker设置为A，另一个设置为B
    for key, value in data.items():
        speaker = value["speaker"]
        if speaker == first_start_speaker:
            value["speaker"] = "A"
        elif speaker in top_speakers:
            value["speaker"] = "B"
        else:
            value["speaker"] = chr(ord('A') + value["speaker"])
    # 输出修改后的JSON数据
    # print(json.dumps(data, indent=4))
    return data

def extract_intervals(data):
    """从JSON数据中提取时间区间，生成A数组"""
    A = []
    speaker = []
    for key, value in data.items():
        A.append([value['start'], value['stop']])
        speaker.append(value['speaker'])
    return A,speaker

def find_overlapping_intervals(A,speaker, start, end):
    """找到与[start, end]范围有重叠的A数组中的元素"""
    result = []
    result_speaker = []
    for i in range(len(A)):
        a, b = A[i]
        s = speaker[i]
        # 检查是否有重叠
        if a <= end and b >= start:
            # 计算重叠部分
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
    # 转换为pydub对象
    if wav_data.dtype == torch.float32:  # 处理归一化数据
        # wav_data = (wav_data * 32767).clamp(-32768, 32767).short()
        wav_data = torch.round(wav_data * 32767).clamp(-32768, 32767).short()
    audio = AudioSegment(
        wav_data.numpy().tobytes(),
        frame_rate=fs,
        sample_width=2,  # 16-bit
        channels=wav_data.shape[0]
    )
    return audio
def from_AudioSegment_to_torchaudio(audio):
    # 转换为pydub对象
    samples = audio.get_array_of_samples()  # 获取numpy数组
    wav_data = torch.FloatTensor(samples).view(-1, audio.channels).T  # 转换为(channel, samples)
    wav_data = wav_data / 32767
    fs = audio.frame_rate
    return wav_data, fs
def split_audio_by_speakers(wav,fs, start, speaker_times):
    # 加载音频文件
    # audio = AudioSegment.from_file(input_file)
    extracted_audio = from_torchaudio_to_AudioSegment(wav, fs)
    silence = AudioSegment.silent(duration=len(extracted_audio))
    speaker_audio = silence
    for start_speak, end_speak in speaker_times:
        start_speak_ms = (start_speak - start) * 1000
        end_speak_ms = (end_speak - start) * 1000
        speaker_audio = speaker_audio[:start_speak_ms] + extracted_audio[start_speak_ms:end_speak_ms] + speaker_audio[end_speak_ms:]
    # # 导出两段音频
    # speaker1_audio.export(output_file1, format="wav")
    # speaker2_audio.export(output_file2, format="wav")
    wav_from_AudioSegment,fs_from_AudioSegment = from_AudioSegment_to_torchaudio(speaker_audio)
    return wav_from_AudioSegment
def match_video_audio(video_data, audio_ids, threshold):
    """
    为每个视频ID匹配最佳音频ID，确保一一对应且匹配度大于阈值
    
    参数:
        video_data: {视频ID: [匹配度1, 匹配度2, ...]}
        audio_ids: [音频ID1, 音频ID2, ...] 对应每个匹配度的音频ID
        threshold: 匹配度阈值
    
    返回:
        {视频ID: 最佳匹配音频ID} 的字典，未匹配的返回None
    """
    # 初始化结果字典和已匹配音频集合
    result = {}
    result_score = {}
    matched_audios = set()
    
    # 创建所有候选匹配对，格式为(匹配度, 视频ID, 音频ID)
    candidates = []
    for vid, scores in video_data.items():
        for i, score in enumerate(scores):
            # if score >= threshold:
            candidates.append((score, vid, audio_ids[i]))
    
    # 按匹配度降序排序
    candidates.sort(reverse=True, key=lambda x: x[0])
    # 贪心算法分配最佳匹配
    for score, vid, aid in candidates:
        if vid not in result and aid not in matched_audios:
            result[vid] = aid
            result_score[vid] = score
            matched_audios.add(aid)
    
    
    # 填充未匹配的视频ID
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
    """
    使用最邻近插值替换list A中的[-1, -1, -1, -1]元素。
    
    参数:
        list_A (list): 包含多个子列表的列表，子列表包含4个float值，部分子列表为[-1, -1, -1, -1]。
    
    返回:
        list: 处理后的列表，所有[-1, -1, -1, -1]被替换为最邻近的非-1值。
    """
    if not list_A:
        return list_A
    
    # 遍历列表，找到所有非-1的索引
    valid_indices = [i for i, sublist in enumerate(list_A) if sublist != [-1, -1, -1, -1]]
    
    if not valid_indices:
        return list_A  # 如果全部是-1，无法插值，直接返回
    
    # 遍历列表，替换-1元素
    for i in range(len(list_A)):
        if list_A[i] == [-1, -1, -1, -1]:
            # 找到最邻近的非-1索引
            nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
            list_A[i] = list_A[nearest_idx].copy()  # 避免引用问题
    
    return list_A
def resize_aspect_ratio(pil_img, target_long_side=640):
    """
    将图像的长边调整为指定大小，短边按比例缩放。
    """
    # 转换为 PIL 图像
    
    # 获取原始图像的尺寸
    width, height = pil_img.size
    
    # 计算长边的比例
    if width > height:
        ratio = target_long_side / width
        new_width = target_long_side
        new_height = int(height * ratio)
    else:
        ratio = target_long_side / height
        new_height = target_long_side
        new_width = int(width * ratio)
    new_height = int(np.round(new_height / 32) * 32)  # 对短边进行四舍五入到32的倍数
    new_width = int(np.round(new_width / 32) * 32)  # 对宽边进行四舍五入到32的倍数
    # 调整图像大小
    resized_img = pil_img.resize((new_width, new_height))
    
    return resized_img
def process_frame(img, transform):
    # BGR转RGB并应用变换
    img_pil = Image.fromarray(img[:, :, [2, 1, 0]])
    img_tensor = transform(img_pil)
    return img_tensor
from torchvision import transforms
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
class Detector:
    def __init__(
        self,
        temp_root_dir,
        record_dir,
        record_name,
        label_name,
        save_oss="s3://qy-hcvm-data/tar_data/",
        save_name="test2"
    ):

        self.save_oss_root = os.path.join(save_oss, save_name)

        self.temp_root_dir = temp_root_dir
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
        self.diarization = Diarization3Dspeaker('cpu',model_cache_dir='/data/pipline/human_centric_vw_model/3D-Speaker/ckpt')
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: resize_aspect_ratio(img, target_long_side=640)),
            ])
    

    def tar_data(self, tar_path, data_dict):
        with tarfile.open(tar_path, 'w') as tar:
            for key in data_dict:
                tarinfo = tarfile.TarInfo(name=key)
                tarinfo.size = len(data_dict[key])
                tar.addfile(tarinfo, io.BytesIO(data_dict[key]))


    def detect(self, vid_name, video_path, audio_path, temp_raw_dir, temp_clip_dir,speaker_root, scene_root):

        speaker_json_name = vid_name.split(".mp4")[0] + ".json"
        speaker_json_path = os.path.join(speaker_root, speaker_json_name)
        
        scene_json_name = vid_name.split(".mp4")[0] + ".json"
        scene_json_path = os.path.join(scene_root, scene_json_name)

        if os.path.exists(speaker_json_path) and os.path.exists(scene_json_path):
            print(f"speaker_json_path: {speaker_json_path} and scene_json_path: {scene_json_path} already exists, skip")
            return
            
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
        wav_data, fs = torchaudio.load(audio_binary_io)  # 

        
        output_field_labels = self.diarization(wav_data)
        print("diarization done")
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

        speaker_ID_dict = {}
        speaker_ID_dict['spaeker'] = speaker
        speaker_ID_dict['A'] = A
        speaker_ID_dict['speaker_json'] = speaker_json
        speaker_ID_dict['org_out'] = out_json

        

        org_scene_list = scene_detect(video_path, HistogramDetector()) # HistogramDetector, ContentDetector
        scene_list = scene_list_process(org_scene_list)
        scene_dict = {}
        scene_dict['org_scene_list'] = [str(scene) for scene in org_scene_list]
        scene_dict['scene_list'] = [str(scene) for scene in scene_list]
        for idx, scene in enumerate(scene_list):
            cur_scene_dict = {}
            start = scene[0].get_timecode()
            end = scene[1].get_timecode()
            start_seconds = scene[0].get_seconds()
            end_seconds = scene[1].get_seconds()
            cur_scene_dict['start'] = start
            cur_scene_dict['end'] = end
            cur_scene_dict['start_seconds'] = start_seconds
            cur_scene_dict['end_seconds'] = end_seconds
            scene_dict["scene_{:03d}".format(idx)] = cur_scene_dict


        with open(speaker_json_path, "w", encoding="utf-8") as f:
            json.dump(speaker_ID_dict, f, ensure_ascii=False, indent=4)
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, ensure_ascii=False, indent=4)


def split_list_get_idx_part(data_list, num, idx):
    assert 0 <= idx < num, "idx must be within the range of [0, num)"
    n = len(data_list)
    chunk_size = n // num
    remainder = n % num

    start = idx * chunk_size + min(idx, remainder)
    end = start + chunk_size + (1 if idx < remainder else 0)

    return data_list[start:end]


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default=None)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--num", type=int, default=150)
args = parser.parse_args()
file_path = args.file_path
idx = args.idx
num = args.num
file_name = file_path.split("/")[-1].replace(".json", "")
record_name = file_name + ".txt"
temp_root_dir = f"/cpfs/new_dialog/{str(num)}_scene_{str(idx)}/temp_dir/"
record_dir = f"/cpfs/new_dialog/{str(num)}_scene_{str(idx)}/records/"
label_name = "video_labels"
speaker_root = f"/cpfs/new_dialog/speaker_json/"
scene_root = f"/cpfs/new_dialog/scene_json/"
os.makedirs(speaker_root, exist_ok=True)
os.makedirs(scene_root, exist_ok=True)

detector = Detector(
    temp_root_dir=temp_root_dir,
    record_dir=record_dir,
    record_name=record_name,
    label_name=label_name,
    save_oss=f"s3://zhangyouliang-model/batch2_yolo/",
    save_name=f"video4",
)


with open(file_path, 'r') as f:
    data_list = json.load(f)
data_list = split_list_get_idx_part(data_list,num,idx)
last_idx = 0
data_list = data_list[::-1]
for data in data_list:
    video_path = data
    temp_raw_dir = os.path.join(detector.temp_root_dir, "raw_video")
    temp_clip_dir = os.path.join(detector.temp_root_dir, "raw_clip_{}".format(detector.record_name.replace(".txt", "")))
    os.makedirs(temp_raw_dir, exist_ok=True)
    os.makedirs(temp_clip_dir, exist_ok=True)
    video_uid = video_path.split('/')[-1].split('.mp4')[0]

    vid_name_raw = vid_name = video_uid + ".mp4"
    audio_path = None
    print(video_path)

    try:
        detector.detect(vid_name, video_path, audio_path, temp_raw_dir, temp_clip_dir,speaker_root, scene_root)
    except Exception as e:
        print("Error start: " + "##########" * 5)
        print(e)
        print("Error end: " + "##########" * 5)
        continue
