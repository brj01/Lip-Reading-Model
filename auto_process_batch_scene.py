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
parser.add_argument("--end", type=int, default=10)
parser.add_argument("--num", type=int, default=260)
parser.add_argument("--ind", type=int, default=0)

args = parser.parse_args()

cpui = 0
file_path = "/data/pipline/human_centric_vw_model/data_process/version2/all_0710_mp4_files.json"

if args.ind != -1:
    start = 10 * args.ind; end = 10 * (args.ind + 1)
else:
    start = args.start; end = args.end
num = args.num
print(start,end)
for idx in range(start, end):
    cmd = "nohup python3 auto_single_scene.py --file_path {} --idx {} --num {} >> auto_process_scene/{}_scene_{}_{}_autoprocess_log_{:03d}.log 2>&1 &".format(file_path,idx,num,idx , num, start, cpui)
    print(cmd)
    os.system(cmd)
    cpui += 1





