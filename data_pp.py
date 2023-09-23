import argparse
import glob
import os
import subprocess

import cv2
import librosa
import numpy as np
import torch
from innerverz import DECA, FaceAligner
from tqdm import tqdm

from utils import util, util_infer

# def check_offset(opts):
#     return


def video_pp(opts, FA, DECA):
    command = f"ffmpeg -y -i {opts.video_path} -qscale:v 2 -threads 1 -f image2 -start_number 0 {opts.frame_save_path + '/%06d.png'}"
    subprocess.call(command, shell=True, stdout=None)

    # get face
    frame_list = sorted(glob.glob(os.path.join(opts.frame_save_path, "*.*")))

    face_bool_list, lmks_list, lmks_3D_list, tfm_inv_list = [], [], [], []
    for frame_path in frame_list:
        name = os.path.basename(frame_path)
        frame = cv2.imread(frame_path)
        FA_dict = FA.get_face(frame)
        lmks_3D = FA.detect_lmk(FA_dict["aligned_face"])["lmks_68"]
        face_bool_list.append(FA_dict["facebool"])
        lmks_list.append(FA_dict["aligned_lmks_68"])
        lmks_3D_list.append(lmks_3D)
        tfm_inv_list.append(FA_dict["tfm_inv"])
        cv2.imwrite(os.path.join(opts.face_save_path, name), FA_dict["aligned_face"])

    np.save(opts.face_bool_save_path, np.array(face_bool_list))
    np.save(opts.lmks_save_path, np.array(lmks_list))
    np.save(opts.tfm_inv_save_path, np.array(tfm_inv_list))

    face_path_list = sorted(glob.glob(os.path.join(opts.face_save_path, "*.*")))

    deca_code_dict_list = []
    for _, lmks, lmks_3D, face_path in zip(
        face_bool_list, lmks_list, lmks_3D_list, face_path_list
    ):
        face = cv2.imread(face_path)

        image_dict = DECA.data_preprocess(face, lmks)
        code_dict = DECA.encode(image_dict["image"])

        _, _, _, _, _, trans_landmarks3ds, _, _ = DC.get_lmks_from_params(
            code_dict, tform_invs=image_dict["tform_inv"]
        )
        lmks_ts = torch.tensor(lmks_3D[None, ...], device=trans_landmarks3ds.device)
        z_calibration_t = (lmks_ts[:, :, 2] - trans_landmarks3ds[:, :, 2]).mean(
            1, keepdim=True
        )
        z_calibration_s = torch.ones_like(z_calibration_t)
        code_dict["z_calibration"] = torch.cat(
            [z_calibration_t, z_calibration_s], dim=1
        )
        optimized_code_dict = DC.optimize_per_batch(
            code_dict, lmks_ts, image_dict["tform_inv"], num_iter=10, lmk_type="3D"
        )

        optimized_code_dict["tform_inv"] = image_dict["tform_inv"]
        deca_code_dict_list.append(optimized_code_dict)
    np.save(opts.deca_param_save_path, np.array(deca_code_dict_list))


def audio_pp(opts):
    audio, _ = librosa.load(opts.video_path, sr=opts.sr)
    mel = util.get_mel(audio)
    np.save(opts.mel_save_path, mel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path options
    parser.add_argument("--video_path", type=str, default="./assets/demo_crop_videos")
    parser.add_argument("--pp_save_root", type=str, default="./assets/demo_crop_pp")

    # video options
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=256)

    # audio options
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")

    args = parser.parse_args()

    DC = DECA()
    FA_3D = FaceAligner(size=args.image_size, lmk_type="3D")

    # 일단 offset은 맞춘 동영상으로 가정
    # TODO : workflow
    # sync offset -> video pp / audio pp -> crop?(optional)
    # if args.workflow["sync_offset"]:
    #     offset = check_offset(args)
    #     if offset:
    #         os.system(
    #             f"ffmpeg -y -i {fname} -itsoffset {offset/args.fps} -i {fname} -vb 20M -map 0:v -map 1:a -r {str(args.fps)} sample.mp4"
    #         )

    video_file_paths = sorted(glob.glob(args.video_path + "/*.*"))

    # video_file_paths = ["assets/test_videos/RD_Radio34_008.mp4"]
    for i, video_file_path in enumerate(video_file_paths):
        work = f"###  {os.path.basename(video_file_path)} ({str(i+1)}/{len(video_file_paths)})  ###"
        print("#" * len(work))
        print(work)
        print("#" * len(work))
        args.video_file_path = video_file_path
        args = util.setting_pp_init(args)
        video_pp(args, FA_3D, DC)
        audio_pp(args)
