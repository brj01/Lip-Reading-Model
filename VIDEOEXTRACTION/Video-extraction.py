#!/usr/bin/env python3
import warnings, sys, os, subprocess, shutil, json, re
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve paths (always next to this script)
SCRIPT_DIR = Path(__file__).parent.resolve()
alamatparent = str(SCRIPT_DIR)
json_path = str(SCRIPT_DIR / "youtubevideos.json")

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return s or "video"

def process_subtitles(video_dir: str, safe_name: str):
    """Process any available subtitles and convert to clean text format"""
    # Look for any subtitle files
    subtitle_files = []
    for ext in ['srt', 'vtt', 'ass', 'lrc']:
        for lang_code in ['ar', 'en', 'fr', 'es', 'de']:  # Common languages
            sub_file = f"video.{lang_code}.{ext}"
            if os.path.exists(sub_file):
                subtitle_files.append((lang_code, ext, sub_file))
    
    # Also look for generic subtitle files
    for ext in ['srt', 'vtt']:
        sub_file = f"video.{ext}"
        if os.path.exists(sub_file):
            subtitle_files.append(('unknown', ext, sub_file))
    
    # Process each subtitle file found
    for lang_code, ext, sub_file in subtitle_files:
        try:
            output_file = os.path.join(video_dir, f"{safe_name}_{lang_code}.txt")
            
            if ext == 'srt':
                clean_srt_content(sub_file, output_file)
            elif ext == 'vtt':
                clean_vtt_content(sub_file, output_file)
            else:
                # For other formats, just copy the file
                shutil.copy2(sub_file, output_file)
            
            print(f"    → Processed {lang_code} subtitles")
            
            # Remove the original subtitle file
            os.remove(sub_file)
            
        except Exception as e:
            print(f"    → Warning: Failed to process {sub_file}: {e}")

def clean_srt_content(input_file: str, output_file: str):
    """Clean SRT subtitle content and extract text only"""
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    text_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip line numbers and timestamps
        if line.isdigit() or '-->' in line:
            i += 1
            continue
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Add text content
        text_lines.append(line)
        i += 1
    
    # Write cleaned text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

def clean_vtt_content(input_file: str, output_file: str):
    """Clean VTT subtitle content and extract text only"""
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    lines = content.split('\n')
    text_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip VTT header, timestamps, and style information
        if (line.startswith('WEBVTT') or 
            '-->' in line or 
            line.startswith('NOTE') or
            line.startswith('STYLE') or
            not line):
            continue
        
        # Add text content
        text_lines.append(line)
    
    # Write cleaned text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

def ensure_dirs():
    """Create videos directory structure"""
    videos_root = os.path.join(alamatparent, "videos")
    os.makedirs(videos_root, exist_ok=True)
    return videos_root

def load_json_items(path):
    print("Reading JSON from:", os.path.abspath(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    if os.stat(path).st_size == 0:
        raise ValueError(f"JSON file is empty: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of objects: [{name, type, video_link}, ...]")
    items = []
    for idx, obj in enumerate(data):
        if not isinstance(obj, dict): 
            print(f"Skipping item #{idx}: not an object"); continue
        link = obj.get("video_link") or obj.get("link") or obj.get("url")
        if not link:
            print(f"Skipping item #{idx}: missing video_link/link/url"); continue
        items.append({
            "name": obj.get("name") or "Untitled",
            "type": obj.get("type") or "video",
            "video_link": link,
            "description": obj.get("description", ""),
            "tags": obj.get("tags", [])
        })
    print(f"Loaded {len(items)} valid item(s).")
    return items

def download_from_json(items, videos_root):
    print(f"Starting downloads for {len(items)} item(s)...")
    
    for idx, item in enumerate(items, start=1):
        name = item["name"]
        link = item["video_link"]
        safe = slugify(name)
        item_dir = os.path.join(videos_root, safe)
        os.makedirs(item_dir, exist_ok=True)
        
        print(f"\n[{idx}] Downloading: {name}")
        print(f"    → Link: {link}")
        print(f"    → Folder: {item_dir}")

        # Clear cache
        subprocess.call([sys.executable, "-m", "yt_dlp", "--rm-cache-dir"])

        # Download video with subtitles
        cmd = [
            sys.executable, "-m", "yt_dlp",
            link,
            "-S", "ext:mp4:m4a",
            "-o", "video.mp4",
            "--socket-timeout", "5",
            "-4",
            "--write-sub",           # Write subtitle file
            "--write-auto-sub",      # Write auto-generated subtitle file
            "--sub-langs", "all",    # Download all available subtitles
            "--convert-subs", "srt", # Convert to SRT format
        ]
        
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[{idx}] yt-dlp returned code {rc} (skipped/failed).")
            continue

        # Move video if present
        mp4_src = os.path.join(alamatparent, "video.mp4")
        if os.path.exists(mp4_src):
            mp4_dst = os.path.join(item_dir, f"{safe}.mp4")
            shutil.move(mp4_src, mp4_dst)
            print(f"    → Saved video → {mp4_dst}")
        else:
            print(f"    → No video was saved (likely filtered/skipped).")

        # Process all available subtitles
        process_subtitles(item_dir, safe)

        # Write comprehensive metadata file
        meta_path = os.path.join(item_dir, "metadata.json")
        metadata = {
            "title": name,
            "type": item["type"],
            "video_link": link,
            "description": item.get("description", ""),
            "tags": item.get("tags", []),
            "download_date": subprocess.getoutput('date -Iseconds'),  # ISO format date
            "files": {
                "video": f"{safe}.mp4",
                "subtitles": [f for f in os.listdir(item_dir) if f.endswith('.txt')]
            }
        }
        
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)
        
        print(f"    → Saved metadata → {meta_path}")

def main():
    videos_root = ensure_dirs()
    try:
        items = load_json_items(json_path)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return
    if not items:
        print("No valid items in JSON.")
        return
    download_from_json(items, videos_root)
    print(f"\n✅ All downloads completed! Check the 'videos' folder.")

if __name__ == "__main__":
    main()