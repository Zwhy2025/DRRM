#!/usr/bin/env python3
"""
Resize dataset images/videos to reduce training time.
Creates a new dataset with smaller images while preserving all other data.
"""

import json
import shutil
from pathlib import Path
import argparse
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image


def resize_image(input_image, output_image, target_height, target_width):
    """Resize image using PIL"""
    output_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(input_image)
        resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized.save(output_image)
        return True
    except Exception as e:
        print(f"Error resizing {input_image}: {e}")
        return False


def resize_videos_with_ffmpeg(input_video, output_video, target_height, target_width):
    """Resize video using ffmpeg"""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', str(input_video),
        '-vf', f'scale={target_width}:{target_height}',
        '-c:v', 'libx264',  # Use H.264 for better compatibility
        '-preset', 'medium',
        '-crf', '23',
        '-y',  # Overwrite output file
        str(output_video)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error resizing {input_video}: {result.stderr}")
        return False
    return True


def update_info_json(info_path, output_path, new_height, new_width):
    """Update info.json with new image dimensions"""
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
    
    print(f"Updated info.json: image dimensions set to {new_height}x{new_width}")


def resize_dataset(input_dir, output_dir, target_height=240, target_width=320):
    """
    Resize all videos in the dataset and update metadata.
    
    Args:
        input_dir: Path to input dataset (e.g., datasets/ur12e/real_libero_spatial)
        output_dir: Path to output dataset (e.g., datasets/ur12e_small/real_libero_spatial)
        target_height: Target image height
        target_width: Target image width
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return False
    
    print(f"Resizing dataset from {input_path} to {output_path}")
    print(f"Target dimensions: {target_height}x{target_width}")
    
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
    # todo 自动添加其他meta文件
    meta_files = ['tasks.jsonl', 'stats.json', 'episodes.jsonl', 'episodes_stats.jsonl']
    for meta_file in meta_files:
        src = input_path / 'meta' / meta_file
        dst = output_path / 'meta' / meta_file
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {meta_file}")
    
    # Copy data directory (parquet files - no need to modify)
    data_input = input_path / 'data'
    data_output = output_path / 'data'
    if data_input.exists():
        if data_output.exists():
            shutil.rmtree(data_output)
        shutil.copytree(data_input, data_output)
        print(f"Copied data directory")
    
    # Resize images (multi-process)
    images_input = input_path / 'images'
    images_output = output_path / 'images'
    
    if images_input.exists():
        exts = ['.png', '.jpg', '.jpeg']
        image_files = []
        for ext in exts:
            image_files.extend(images_input.rglob(f'*{ext}'))
            image_files.extend(images_input.rglob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to resize")
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=resize_dataset.workers) as executor:
            futures = {}
            for image_file in image_files:
                relative_path = image_file.relative_to(images_input)
                output_image = images_output / relative_path
                futures[executor.submit(resize_image, image_file, output_image, target_height, target_width)] = image_file
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing images"):
                if future.result():
                    success_count += 1
        
        print(f"Successfully resized {success_count}/{len(image_files)} images")
    else:
        print("No images directory found")
    
    # Resize videos
    videos_input = input_path / 'videos'
    videos_output = output_path / 'videos'
    
    if videos_input.exists():
        video_files = list(videos_input.rglob('*.mp4'))
        print(f"Found {len(video_files)} videos to resize")
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=resize_dataset.workers) as executor:
            futures = {}
            for video_file in video_files:
                relative_path = video_file.relative_to(videos_input)
                output_video = videos_output / relative_path
                futures[executor.submit(resize_videos_with_ffmpeg, video_file, output_video, target_height, target_width)] = video_file
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing videos"):
                if future.result():
                    success_count += 1
        
        print(f"Successfully resized {success_count}/{len(video_files)} videos")
    else:
        print("No videos directory found")
    
    print(f"\nDataset resizing complete!")
    print(f"Output saved to: {output_path}")
    return True


resize_dataset.workers = multiprocessing.cpu_count()


def main():
    parser = argparse.ArgumentParser(description='Resize dataset images/videos')
    parser.add_argument('input_dir', type=str, help='Input dataset directory')
    parser.add_argument('output_dir', type=str, help='Output dataset directory')
    parser.add_argument('--height', type=int, default=240, help='Target height (default: 240)')
    parser.add_argument('--width', type=int, default=320, help='Target width (default: 320)')
    parser.add_argument('--workers', type=int, default=16, help='Parallel workers for images')
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)
    
    resize_dataset.workers = max(1, args.workers)
    success = resize_dataset(args.input_dir, args.output_dir, args.height, args.width)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

