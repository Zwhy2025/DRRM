#!/usr/bin/env python3
"""
Convert LeRobot dataset from image mode to video mode.
Creates a new dataset with video files while preserving all other data.
"""

import json
import shutil
from pathlib import Path
import argparse
import subprocess
import sys
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

def encode_video_frames(input_dir, output_video, fps=30, vcodec='libx264', pix_fmt='yuv420p'):
    """Convert image sequence to video using ffmpeg"""
    input_dir = Path(input_dir).resolve()
    output_video = Path(output_video).resolve()
    
    output_video.parent.mkdir(parents=True, exist_ok=True)

    png_files = sorted(list(input_dir.glob('frame_*.png')))
    if not png_files:
        print(f"Error: No frame_*.png files found in {input_dir}")
        return False
    
    first_frame = png_files[0].name
    match = re.match(r'frame_(\d+)\.png', first_frame)
    if not match:
        print(f"Error: Unexpected frame filename format: {first_frame}")
        return False
    
    start_num = int(match.group(1))
    input_pattern = str(input_dir / 'frame_%06d.png')
    
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', vcodec,
        '-pix_fmt', pix_fmt,
        '-y',
        str(output_video)
    ]
    
    if start_num != 0:
        cmd.insert(-2, str(start_num))
        cmd.insert(-2, '-start_number')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting to video: {result.stderr}")
        return False
    return True

def update_info_json(info_path, output_path, new_height, new_width):
    """Update info.json with new video dimensions"""
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Update image/video feature dimensions
    for key, feature in info['features'].items():
        if feature['dtype'] in ['image', 'video']:
            # Update shape [C, H, W] -> [C, new_H, new_W]
            feature['shape'][1] = new_height
            feature['shape'][2] = new_width
            
            # Update video info if present
            if 'info' in feature:
                feature['info']['video.height'] = new_height
                feature['info']['video.width'] = new_width
    
    # Save updated info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Updated info.json: video dimensions set to {new_height}x{new_width}")

def convert_image_to_video(input_dir, output_dir, fps=30, target_height=240, target_width=320):
    """
    Convert all image sequences to video files and update metadata.
    
    Args:
        input_dir: Path to input dataset (e.g., datasets/ur12e/real_libero_spatial)
        output_dir: Path to output dataset (e.g., datasets/ur12e_video/real_libero_spatial)
        fps: Target frames per second for the video
        target_height: Target video height
        target_width: Target video width
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return False
    
    print(f"Converting dataset from {input_path} to {output_path}")
    print(f"Target video dimensions: {target_height}x{target_width}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy and update meta/info.json
    info_input = input_path / 'meta' / 'info.json'
    info_output = output_path / 'meta' / 'info.json'
    if info_input.exists():
        update_info_json(info_input, info_output, target_height, target_width)
    else:
        print(f"Warning: {info_input} not found")
        return False
    
    # Copy other meta files
    meta_files = ['tasks.jsonl', 'stats.json', 'episodes.jsonl', 'episodes_stats.jsonl']
    for meta_file in meta_files:
        src = input_path / 'meta' / meta_file
        dst = output_path / 'meta' / meta_file
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {meta_file}")
    
    # Convert image sequences to video
    images_input = input_path / 'images'
    videos_output = output_path / 'videos'
    
    if images_input.exists():
        image_dirs = [d for d in images_input.iterdir() if d.is_dir()]
        print(f"Found {len(image_dirs)} image sequences to convert to video")
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=convert_image_to_video.workers) as executor:
            futures = {}
            for image_dir in image_dirs:
                video_name = image_dir.name + '.mp4'
                output_video = videos_output / video_name
                futures[executor.submit(encode_video_frames, image_dir, output_video, fps)] = image_dir
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting images to videos"):
                if future.result():
                    success_count += 1
        
        print(f"Successfully converted {success_count}/{len(image_dirs)} image sequences to video")
    else:
        print("No image sequences found")
    
    print(f"\nDataset conversion complete!")
    print(f"Output saved to: {output_path}")
    return True

convert_image_to_video.workers = multiprocessing.cpu_count()

def main():
    parser = argparse.ArgumentParser(description='Convert LeRobot dataset from image mode to video mode')
    parser.add_argument('input_dir', type=str, help='Input dataset directory')
    parser.add_argument('output_dir', type=str, help='Output dataset directory')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video (default: 30)')
    parser.add_argument('--height', type=int, default=240, help='Target video height (default: 240)')
    parser.add_argument('--width', type=int, default=320, help='Target video width (default: 320)')
    parser.add_argument('--workers', type=int, default=16, help='Parallel workers for image processing')
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)
    
    convert_image_to_video.workers = max(1, args.workers)
    success = convert_image_to_video(args.input_dir, args.output_dir, args.fps, args.height, args.width)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
