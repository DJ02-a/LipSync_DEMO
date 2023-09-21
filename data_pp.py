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

from utils import infer_util, util

# def check_offset(opts):
#     return


def video_pp(opts, FA, DECA):
    if opts.workflow["frames"]:
        command = f"ffmpeg -y -i {opts.video_path} -qscale:v 2 -threads 1 -f image2 -start_number 0 {opts.frame_save_path + '/%06d.png'}"
        subprocess.call(command, shell=True, stdout=None)

    if opts.workflow["face"]:
        frame_list = sorted(glob.glob(os.path.join(opts.frame_save_path, "*.*")))

        face_bool_list, lmks_106_list, tfm_inv_list = [], [], []
        for frame_path in frame_list:
            name = os.path.basename(frame_path)
            frame = cv2.imread(frame_path)
            FA_dict = FA.get_face(frame)
            cv2.imwrite(
                os.path.join(opts.face_save_path, name), FA_dict["aligned_face"]
            )
            face_bool_list.append(FA_dict["facebool"])
            lmks_106_list.append(FA_dict["lmks_106"])
            tfm_inv_list.append(FA_dict["tfm_inv"])
        np.save(opts.face_bool_save_path, np.array(face_bool_list))
        np.save(opts.lmks_save_path, np.array(lmks_106_list))
        np.save(opts.tfm_inv_save_path, np.array(tfm_inv_list))

    if opts.workflow["deca_params"]:
        face_path_list = sorted(glob.glob(os.path.join(opts.face_save_path, "*.*")))

        deca_file_names, deca_tform_invs, deca_code_dict_list = [], [], []
        for _, lmks_106, face_path in zip(
            face_bool_list, lmks_106_list, face_path_list
        ):
            file_name = os.path.basename(face_path)
            face = cv2.imread(face_path)
            image_dict = DECA.data_preprocess(face, lmks_106)
            code_dict = DECA.encode(image_dict["image"])

            deca_file_names.append(file_name)
            deca_tform_invs.append(image_dict["tform_inv"])
            deca_code_dict_list.append(code_dict)
        np.save(opts.deca_param_save_path, np.array(deca_code_dict_list))

    if opts.workflow["flame_params"]:
        trans_landmarks2d_list = []
        for file_name, code_dict, tform_invs in zip(
            deca_file_names, deca_code_dict_list, deca_tform_invs
        ):
            _, _, _, _, trans_landmarks2d, _, _ = DECA.get_lmks_from_params(
                code_dict, tform_invs=tform_invs
            )
            trans_landmarks2d_list.append(trans_landmarks2d.squeeze().cpu().numpy())
        np.save(opts.trans_landmark2d_save_path, np.array(trans_landmarks2d_list))
    return


def audio_pp(opts):
    audio, _ = librosa.load(opts.video_path, sr=opts.sr)
    mel = util.get_mel(audio)
    np.save(opts.mel_save_path, mel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path options
    parser.add_argument("--type", type=str, default="driving")
    parser.add_argument("--video_path", type=str, default="./assets/demo_videos")
    parser.add_argument("--pp_save_root", type=str, default="./assets/demo_pp")

    # video options
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=256)

    # audio options
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")

    args = parser.parse_args()

    DC = DECA()
    FA_3D = FaceAligner(size=args.image_size)

    # 일단 offset은 맞춘 동영상으로 가정
    # TODO : workflow
    # sync offset -> video pp / audio pp -> crop?(optional)
    # if args.workflow["sync_offset"]:
    #     offset = check_offset(args)
    #     if offset:
    #         os.system(
    #             f"ffmpeg -y -i {fname} -itsoffset {offset/args.fps} -i {fname} -vb 20M -map 0:v -map 1:a -r {str(args.fps)} sample.mp4"
    #         )

    # video_file_paths = sorted(glob.glob(args.video_path + "/*.*"))
    video_file_paths = ["assets/demo_videos/Mrbang.mp4"]
    for video_file_path in video_file_paths:
        args.video_file_path = video_file_path
        args = util.setting_pp_init(args)
        video_pp(args, FA_3D, DC)
        if args.type == "driving":
            audio_pp(args)
