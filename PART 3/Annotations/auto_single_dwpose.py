 # Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import json
import copy
import torch
import random
import argparse
import shutil
import tempfile
import subprocess
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
import logging
from io import BytesIO
import os.path as osp
import multiprocessing as mp
import sys
from tqdm import tqdm
from dwpose import util
from dwpose.wholebody import Wholebody
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger


class DWposeDetector:
    def __init__(self,onnx_pose, onnx_det):
        self.pose_estimation = Wholebody(onnx_pose, onnx_det)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            # print(score.shape)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) # (bodyfoot_score[:,18]+bodyfoot_score[:,19])/2
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) # (bodyfoot_score[:,21]+bodyfoot_score[:,22])/2
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_body_and_foot(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas

def dw_func(_id, file_path, dwpose_model, save_dir = "/cpfs/dialog/dwpose"):
    video_name = (file_path).split('/')[-1].replace('.mp4', '.npy')
    dwpose_woface_path = os.path.join(save_dir, video_name)
    if os.path.exists(dwpose_woface_path):
        return
    frame_all = []
    videoCapture = cv2.VideoCapture(file_path)
    iiii = 0
    while videoCapture.isOpened():
        # get a frame
        ret, frame = videoCapture.read()
        iiii += 1
        if ret:
            
            frame_all.append(frame)
        else:
            break
    
    videoCapture.release()
    video_frame_all = {}
    dwpose_wface_list = []
    for i_index, frame in tqdm(enumerate(frame_all)):
        frame_name  = str(i_index).zfill(6)+".jpg"
        frame_h, frame_w, _ = frame.shape
        _, img_encode = cv2.imencode('.jpg', frame)
        img_bytes = img_encode.tobytes()
        video_frame_all[frame_name] = img_bytes
        dwpose_wface = dwpose_model(frame)
        dwpose_wface_list.append(dwpose_wface)
    np.save(dwpose_woface_path, dwpose_wface_list)

def mp_main(ii, dwpose_model, video_paths,onnx_pose, onnx_det,save_dir):
    dwpose_model = DWposeDetector(onnx_pose, onnx_det)  
    for i, file_path in enumerate(video_paths):
        logger.info(f"{i}/{len(video_paths)}, {file_path}")
        dw_func(i, file_path, dwpose_model,save_dir)

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
logger = get_logger('dw pose extraction')
def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--total", type=int, default=1
    )
    parser.add_argument(
        "--index", type=int, default=0
    )
    parser.add_argument(
        "--onnx_pose", type=str, default="../weights/dw-ll_ucoco_384.onnx"
    )
    parser.add_argument(
        "--onnx_det", type=str, default="../weights/yolox_l.onnx"
    )
    parser.add_argument(
        "--save_dir", type=str, default="../skeleton/dwpose_result"
    )
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    mp.set_start_method('spawn')
    opt = get_parse()
    num_worker = opt.total
    index = opt.index
    onnx_pose = opt.onnx_pose
    onnx_det = opt.onnx_det
    save_dir = opt.save_dir
    all_video_list_path = opt.video_list_path
    video_list = []
    with open(all_video_list_path, "r") as f:
        video_list = f.read().splitlines()
    video_list =  video_list[index :: num_worker]
    logger.info("There are {} videos for extracting poses".format(len(video_list)))
    logger.info('LOAD: DW Pose Model')
    dwpose_model = None
    tid = 0
    mp_main(tid, dwpose_model, video_list, onnx_pose, onnx_det, save_dir)
    
