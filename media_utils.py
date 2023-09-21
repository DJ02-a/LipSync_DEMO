import os
import subprocess


def extract_frames(video_path, from_time, to_time, save_path, fps=25):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        command = f"ffmpeg -i {video_path} -ss {from_time} -to {to_time} -r {fps} -vb 20M -start_number {from_time*fps} {save_path}/%06d.png"
        subprocess.run([command], shell=False)
