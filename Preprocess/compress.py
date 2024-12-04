import subprocess
from glob import glob

def compress_video(inp_vid_path, out_vid_path, cp_rate = 23, preset = 'medium'):
    convert_cmd = f'ffmpeg -i "{inp_vid_path}" -c:v libx264 -preset {preset} -crf {cp_rate} -c:a copy "{out_vid_path}"'
    try:
        subprocess.run(convert_cmd, shell=True, check=True)
        print(f"Video compression completed, output file: {out_vid_path}")
    except subprocess.CalledProcessError as e:
        print(f"Video compression failed: {e}")

video_paths = '/home/thaibahung/ML_Project/Dataset/Example'
all_dirs = glob(video_paths + '/*')
out_dirs = '/home/thaibahung/ML_Project/Dataset/Compress_'

print("Hello")

for path in all_dirs:
    name = path.split('/')[-1]
    out_path = out_dirs + name
    print(name)
    compress_video(path, out_path, cp_rate=28)