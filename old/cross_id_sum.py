# 같은 오디오에 모든 인물 cross_id 결과를 visualization하는 코드입니다.
# cross_id.py가 선행되어야 합니다.
packages_path = './'
import os, cv2, glob
from tqdm import tqdm
import numpy as np
from innerverz import FaceAligner

def get_grad_mask(size=512):
    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr1 = np.sqrt(x_axis ** 4 + y_axis ** 4)

    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr2 = np.sqrt(x_axis ** 2 + y_axis ** 2)

    grad_mask = np.clip(1-(arr1/2+arr2/2), 0, 1)
    return grad_mask

FA_3D = FaceAligner(size=256, lmk_type='3D')

name_list = ['RD_Radio13_000', 'RD_Radio14_000', 'RD_Radio16_000', 'RD_Radio52_000']
fps = 25
save_dir = 'crossid_results_sync_130000'
root_dir = '/ssd2t/DATASET/HDTF-synced-video-25fps-frames'

min_frames = len(glob.glob(f'assets/inputs/{name_list[0]}'))
for s_name in name_list:
    output_dir = f'{save_dir}/{s_name}'
    video_dir = f'{save_dir}/{s_name}/video'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    s_img_dir = f'assets/inputs/{s_name}'
    s_img_paths = sorted(glob.glob(s_img_dir))
    
    # for case that the number of frames is different / find min frame length
    for t_name in name_list:
        t_img_dir = f'{save_dir}/{t_name}_{s_name}/*.*'
        t_img_paths = sorted(glob.glob(t_img_dir))
        if len(t_img_paths) < min_frames:
            min_frames = len(t_img_paths)
    
    print(s_name, min_frames)
    for i in tqdm(range(min_frames)):
        grid = []
        for t_name in name_list:
            # path setting
            orig_img_path = f'{root_dir}/{t_name}/frames/' + '%06d.png'%i
            aligned_img_path = f'{save_dir}/{t_name}_{s_name}/' + '%06d.png'%i
            param_path = f'{root_dir}/{t_name}/flame_params/' + '%06d.npy'%i

            # load data
            orig_imgs = cv2.imread(orig_img_path)
            aligned_imgs = cv2.imread(aligned_img_path)
            size = orig_imgs.shape[:-1]
            flame_param = np.load(param_path, allow_pickle=True).item()
            tfm_inv = FA_3D.get_face(orig_imgs)['tfm_inv']
            
            # get replaced image
            grad_mask = (get_grad_mask(256)*3).clip(0,1)
            _grad_mask = cv2.warpAffine(grad_mask, tfm_inv, (size[1], size[0]))[:,:,None]
            _reshape_face = cv2.warpAffine(aligned_imgs, tfm_inv, (size[1], size[0]))
            reshape_img = _grad_mask * _reshape_face + (1 - _grad_mask) * orig_imgs
            grid.append(cv2.resize(reshape_img, (512,512)))
            
        cat_img = np.concatenate(grid,axis=1)
        cv2.imwrite(f'{output_dir}/%06d.png'%i, cat_img)

    os.system(f'ffmpeg -y -r {fps} -start_number 0 -i {output_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p {video_dir}/{s_name}.mp4')
    os.system(f'ffmpeg -y -i {video_dir}/{s_name}.mp4 -i assets/audio/{s_name}.wav -c:v copy -map 0:v -map 1:a -y {video_dir}/{s_name}_all.mp4')
    os.system(f'rm {video_dir}/{s_name}.mp4')


    