packages_path = "./"
import glob
import os
import sys

import cv2
import torch

sys.path.append(packages_path)
import numpy as np
from innerverz import DECA
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from main import LipSyncer

# /data1/LipSync-dataset/HDTFs/HDTF-synced-video-25fps-frames 를 data삼아 training
# 앞에 250 frames만 사용함 / 전처리 250개보다 많이 되어있는 폴더가 있기 때문에 사용에 유의
ckpt_name = "sync_130000"
LS = LipSyncer(
    ckpt_path=f"{packages_path}ckpts/{ckpt_name}.pt", skip=False, ref_input=False
)
DC = DECA()

name_list = [
    "RD_Radio13_000",
    "RD_Radio14_000",
    "RD_Radio16_000",
    "RD_Radio52_000",
]  # RD_Radio52_000 is not in train dataset
fps = 25
filter_lmk = True

# t = target / s = source
t_name = name_list[0]
s_name = name_list[1]

for t_name in name_list:
    for s_name in name_list:
        # setting data path / it's for serv6 you should rematch data path
        t_img_dir = f"./assets/frames/{t_name}/*"
        s_img_dir = f"./assets/frames/{s_name}/*"
        input_dir = f"assets/inputs/{t_name}_{s_name}"
        output_dir = f"crossid_results_{ckpt_name}/{t_name}_{s_name}"
        concat_dir = f"crossid_results_{ckpt_name}/{t_name}_{s_name}/concat"
        video_dir = f"crossid_results_{ckpt_name}/{t_name}_{s_name}/video"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        t_img_paths = sorted(glob.glob(t_img_dir))
        s_img_paths = sorted(glob.glob(s_img_dir))
        path_length = min(len(t_img_paths), len(s_img_paths))
        import pdb

        print(t_name, " / ", s_name)
        # inference는 audio feature의 batch 때문에 5장씩 끊어서 clip형태로 들어갑니다.
        end_frame_index = list(range(5, path_length, 5))
        # video_clip_num = len(end_frame_index) # 5*video_clip_num <= path_length / 남는 부분은 버림
        video_clip_num = (
            50 if len(end_frame_index) > 50 else len(end_frame_index)
        )  # training data가 맨 앞 250 frames만 포함하고 있기 때문 -> 일부러 이렇게 하는 듯.
        # first_image = cv2.imread(t_img_paths[0])
        s_hubert_path = (
            os.path.dirname(s_img_paths[0].replace("aligned_imgs", "huberts_768"))
            + "/000000.npy"
        )
        s_hubert_features = torch.tensor(
            np.load(s_hubert_path, allow_pickle=True).transpose(1, 0)
        ).to(LS.device)
        for i in tqdm(range(video_clip_num)):
            t_img_path_clip = t_img_paths[end_frame_index[i] - 5 : end_frame_index[i]]
            s_img_path_clip = s_img_paths[end_frame_index[i] - 5 : end_frame_index[i]]
            t_ref_img_path_clip_list = (
                t_img_path_clip  # gt image와 같은 이미지를 reference로 넣어줌
            )

            frame_num = int(s_img_path_clip[0].split("/")[-1].split(".")[0])
            # 0.2sec hubert feature, 이거 왜 *2?
            s_hubert_feature = (
                s_hubert_features[:, frame_num * 2 : frame_num * 2 + 9]
                .unsqueeze(0)
                .unsqueeze(0)
            )

            (
                frame_names,
                t_img_clips,
                s_img_clips,
                t_ref_img_clips,
                t_lmk_clips,
                jaw_lmk_clips,
            ) = ([], [], [], [], [], [])
            for i in range(5):
                # image preprocess
                frame_names.append(os.path.basename(t_img_path_clip[i]))
                t_image = LS.pp_image(t_img_path_clip[i]).unsqueeze(0).to(LS.device)
                s_image = LS.pp_image(s_img_path_clip[i]).unsqueeze(0).to(LS.device)
                t_ref_image = (
                    LS.pp_image(t_ref_img_path_clip_list[i]).unsqueeze(0).to(LS.device)
                )

                # guide image를 위한 landmark 뽑는 과정
                t_param_path = (
                    os.path.splitext(
                        t_img_path_clip[i].replace("aligned_imgs", "flame_params")
                    )[0]
                    + ".npy"
                )
                s_param_path = (
                    os.path.splitext(
                        s_img_path_clip[i].replace("aligned_imgs", "flame_params")
                    )[0]
                    + ".npy"
                )
                t_flame_params = np.load(t_param_path, allow_pickle=True).item()
                s_flame_params = np.load(s_param_path, allow_pickle=True).item()
                jaw_flame_params = {}
                for key in t_flame_params.keys():
                    jaw_flame_params[key] = t_flame_params[key].copy()
                # target의 parmas에 source의 pose, exp를 넣어줌
                jaw_flame_params["pose"][3] = s_flame_params["pose"][3]
                jaw_flame_params["exp"] = s_flame_params["exp"]
                for key in jaw_flame_params.keys():
                    t_flame_params[key] = (
                        torch.from_numpy(t_flame_params[key]).unsqueeze(0).to(LS.device)
                    )
                    jaw_flame_params[key] = (
                        torch.from_numpy(jaw_flame_params[key])
                        .unsqueeze(0)
                        .to(LS.device)
                    )
                t_vis_dict = DC.decode(
                    t_flame_params,
                    original_image=t_image,
                    tform_invs=t_flame_params["tform_inv"],
                )
                jaw_vis_dict = DC.decode(
                    jaw_flame_params,
                    original_image=t_image,
                    tform_invs=jaw_flame_params["tform_inv"],
                )
                pdb.set_trace()

                # 5장씩 다시 모아줌
                t_img_clips.append(t_image)
                s_img_clips.append(s_image)
                t_ref_img_clips.append(t_ref_image)
                t_lmk_clips.append(t_vis_dict["landmarks2d_points"])
                jaw_lmk_clips.append(jaw_vis_dict["landmarks2d_points"])

            # input에 넣기 좋은 형태로 concat
            t_img_clips = torch.cat(t_img_clips, dim=0)
            s_img_clips = torch.cat(s_img_clips, dim=0)
            t_ref_img_clips = torch.cat(t_ref_img_clips, dim=0)
            t_lmk_clips = torch.cat(t_lmk_clips, dim=0)
            jaw_lmk_clips = torch.cat(jaw_lmk_clips, dim=0)

            ## 5장에 대해서만 smoothing하는 코드 / 아직 이부분 넣고 코드 돌려보지 않았습니다...
            ## 전체를 한번에 smoothing하려면 2단계로 나눠야함
            ## 1. params를 불러와서 landmark를 받고 smoothing
            ## 2. smoothing된 landmark를 slicing만 해서 input으로 넣어줌
            # if filter_lmk:
            #     t_lmk_clips = gaussian_filter1d(t_lmk_clips, sigma=1.0, axis=0)
            #     t_lmk_clips = np.reshape(t_lmk_clips, (-1, 68, 2))
            #     jaw_lmk_clips = gaussian_filter1d(jaw_lmk_clips, sigma=1.0, axis=0)
            #     jaw_lmk_clips = np.reshape(jaw_lmk_clips, (-1, 68, 2))

            input_clips, result_clips = LS(
                t_img_clips,
                t_ref_img_clips,
                s_hubert_feature,
                t_lmk_clips,
                jaw_lmks=jaw_lmk_clips,
                dil_iter_num=10,
            )

            for i in range(5):
                grid = []
                grid.append(
                    t_img_clips[i]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:, :, ::-1]
                    * 127.5
                    + 127.5
                )
                grid.append(
                    s_img_clips[i]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:, :, ::-1]
                    * 127.5
                    + 127.5
                )
                grid.append(
                    result_clips[i]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:, :, ::-1]
                    * 127.5
                    + 127.5
                )
                cv2.imwrite(
                    f"{input_dir}/{frame_names[i]}",
                    input_clips[i]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:, :, ::-1]
                    * 127.5
                    + 127.5,
                )
                cv2.imwrite(
                    f"{output_dir}/{frame_names[i]}",
                    result_clips[i]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:, :, ::-1]
                    * 127.5
                    + 127.5,
                )
                cv2.imwrite(
                    f"{concat_dir}/{frame_names[i]}", np.concatenate(grid, axis=1)
                )

        os.system(
            f"ffmpeg -y -r {fps} -start_number 0 -i {concat_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p {video_dir}/{t_name}_{s_name}_tmp.mp4"
        )
        os.system(
            f"ffmpeg -y -i {video_dir}/{t_name}_{s_name}_tmp.mp4 -i assets/audio/{s_name}.wav -c:v copy -map 0:v -map 1:a -y {video_dir}/{t_name}_{s_name}.mp4"
        )
        os.system(f"rm {video_dir}/{t_name}_{s_name}_tmp.mp4")
