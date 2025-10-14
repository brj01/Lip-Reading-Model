import subprocess
from multiprocessing import Pool
import argparse
def run_script(args):
    index, gpu_id, total, video_list_path,output = args
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python refine_ID.py --index {index} --total {total} --video_list_path {video_list_path} --output {output}"
    print(cmd)
    subprocess.run(cmd, shell=True)

def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--GPU", type=int, default=8)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--num_cur", type=int, default=24)
    parser.add_argument("--num_Total", type=int, default=96)
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")
    parser.add_argument("--output", type=str, default="./ID_result")
    return parser.parse_args()

def main():
    opt = get_parse()
    num_GPU = opt.GPU
    shift = opt.shift
    num_Total = opt.num_Total
    video_list_path = opt.video_list_path
    output = opt.output
    num_cur = opt.num_cur
    tasks = [(i + shift, i % num_GPU, num_Total,video_list_path,output) for i in range(num_cur)] 
    with Pool(num_Total) as pool:
        pool.map(run_script, tasks)