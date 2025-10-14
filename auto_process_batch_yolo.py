import os
import tarfile
import numpy as np
import shutil
from tqdm import tqdm
import time
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=24)
parser.add_argument("--ind", type=int, default=-1)
args = parser.parse_args()

cpui = 0
batch_num = "batch21"
file_list_dir = f"/data/pipline/human_centric_vw_model/data_process/video_json/{batch_num}"

file_list = list(os.listdir(file_list_dir))
file_list.sort()
data_len = len(file_list)

if args.ind != -1:
    start = 24 * args.ind; end = 24 * (args.ind + 1)
else:
    start = args.start; end = args.end
os.makedirs("auto_process_step1",exist_ok=True)
print(start,end)
num_GPU = 8
for file_item in file_list[start:end]:
    gpui = cpui % num_GPU
    file_path = os.path.join(file_list_dir, file_item)
    print("processing file list {}".format(file_path))
    cmd = "CUDA_VISIBLE_DEVICES={} nohup python3 auto_single_yolo.py --file_path {} {} --batch_num {} >> auto_process_yolo2/{}_{}_autoprocess_log_{:03d}.log 2>&1 &".format(gpui, file_path,batch_num,batch_num, start, cpui)
    print(cmd)
    os.system(cmd)
    cpui += 1





