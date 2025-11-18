import os
import json

# === 修改为你的路径 ===
mp4_list_json_path = '/data/pipline/human_centric_vw_model/data_process/statistic/all_org_mp4_files.json'  # 存储 mp4 文件路径的 json 文件
json_folder_path = '/cpfs/new_dialog/scene_json/'    # json 文件夹（下一层有若干 .json 文件）
output_path = 'all_0710_mp4_files.json'     # 输出结果路径

# 1. 加载 mp4 列表
with open(mp4_list_json_path, 'r') as f:
    mp4_paths = json.load(f)

# 2. 获取 json 文件夹下一层所有文件名（不包含路径）
json_filenames = set(os.listdir(json_folder_path))

# 3. 提取 mp4 的文件名（带 .mp4）和对应的 .json 名字
missing_files = []
for mp4_path in mp4_paths:
    mp4_filename = os.path.basename(mp4_path)  # e.g. "-3qRRzepWjk_1920x1080_full_video.mp4"
    json_filename = mp4_filename.replace('.mp4', '.json')
    if json_filename not in json_filenames:
        missing_files.append(mp4_path)

# 4. 保存缺失项
with open(output_path, 'w') as f:
    json.dump(missing_files, f, indent=2)

print(f"共缺失 {len(missing_files)} 个 json 文件，已保存至 {output_path}")
