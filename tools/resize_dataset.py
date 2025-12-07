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
from tqdm import tqdm


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
            # Update shape [H, W, C] -> [new_H, new_W, C]
            feature['shape'][0] = new_height
            feature['shape'][1] = new_width
            
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
    
    # Resize videos
    videos_input = input_path / 'videos'
    videos_output = output_path / 'videos'
    
    if videos_input.exists():
        # Find all video files
        video_files = list(videos_input.rglob('*.mp4'))
        print(f"Found {len(video_files)} videos to resize")
        
        # Resize each video
        success_count = 0
        for video_file in tqdm(video_files, desc="Resizing videos"):
            # Preserve directory structure
            relative_path = video_file.relative_to(videos_input)
            output_video = videos_output / relative_path
            
            if resize_videos_with_ffmpeg(video_file, output_video, target_height, target_width):
                success_count += 1
            else:
                print(f"Failed to resize {video_file}")
        
        print(f"Successfully resized {success_count}/{len(video_files)} videos")
    else:
        print("No videos directory found")
    
    print(f"\nDataset resizing complete!")
    print(f"Output saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Resize dataset images/videos')
    parser.add_argument('input_dir', type=str, help='Input dataset directory')
    parser.add_argument('output_dir', type=str, help='Output dataset directory')
    parser.add_argument('--height', type=int, default=240, help='Target height (default: 240)')
    parser.add_argument('--width', type=int, default=320, help='Target width (default: 320)')
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)
    
    success = resize_dataset(args.input_dir, args.output_dir, args.height, args.width)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

