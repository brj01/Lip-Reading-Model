import subprocess
from multiprocessing import Pool
import argparse
def run_script(args):
    index, gpu_id, total , audio_list_path, save_dir = args
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python process_local_asr.py --index {index} --total {total} --audio_list_path {audio_list_path} --save_dir {save_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True)

def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--GPU", type=int, default=8)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--num_cur", type=int, default=24)
    parser.add_argument("--num_Total", type=int, default=96)
    parser.add_argument("--audio_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/wav_0710.txt")
    parser.add_argument("--save_dir", type=str, default="./asr_result")
    return parser.parse_args()

def main():
    opt = get_parse()
    num_GPU = opt.GPU
    shift = opt.shift
    num_Total = opt.num_Total
    audio_list_path = opt.audio_list_path
    save_dir = opt.save_dir
    tasks = [(i + shift, i % num_GPU, num_Total, audio_list_path, save_dir) for i in range(num_Total)]
    with Pool(num_Total) as pool:
        pool.map(run_script, tasks)
