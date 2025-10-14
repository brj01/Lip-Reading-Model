import subprocess
from multiprocessing import Pool
import argparse
def run_script(args):
    index, total, video_list_path = args
    cmd = f"python filter_light.py --index {index} --total {total} --video_list_path {video_list_path}"
    print(cmd)
    subprocess.run(cmd, shell=True)

# find /path/to/folder -type f -name "*.json" | wc -l
# python filter_light.py --index 0 --total 16 --num_proc 1 12 20
# python para_filter.py
def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--cur_num", type=int, default=4)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--num_Total", type=int, default=28)
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")
    parser.add_argument("--save_dir", type=str, default="./light_result")
    return parser.parse_args()

def main():
    opt = get_parse()
    cur_num = opt.cur_num
    shift = opt.shift
    num_Total = opt.num_Total
    video_list_path = opt.video_list_path
    tasks = [(i + shift, num_Total, video_list_path) for i in range(cur_num)]
    with Pool(num_Total) as pool:
        pool.map(run_script, tasks)
