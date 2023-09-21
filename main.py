import argparse
import glob
import os

import cv2
import numpy as np
from innerverz import DECA
from tqdm import tqdm

import utils.util as util
from package.LipSync import LipSyncer


# TODO : 저장하는 코드
def vis(opts, results):
    return


# TODO : 5장 만큼 끊어서 generation 하는 코드 필요
def run(opts):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # inference options
    parser.add_argument(
        "--driving_clip_path", type=str, default="./assets/driving_clips"
    )
    parser.add_argument("--source_clip_path", type=str, default="./assets/source_clips")
    parser.add_argument("--save_path", type=str, default="./results/")

    # video options
    parser.add_argument("--sv_to", type=int, default=0)
    parser.add_argument("--sv_from", type=int, default=10)
    parser.add_argument("--dv_to", type=int, default=0)
    parser.add_argument("--dv_from", type=int, default=10)

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=256)

    # lipsync model options
    parser.add_argument("--frame_amount", type=int, default=5)
    parser.add_argument("--hubert_amount", type=int, default=9)
    parser.add_argument("--ckpt_file", type=str, default="sync_130000.pt")
    parser.add_argument("--skip_connection", type=bool, default=False)
    parser.add_argument("--ref_input", type=bool, default=False)

    args = parser.parse_args()

    LS = LipSyncer(
        ckpt_path=f"./Lipsync_inference/ckpts/{args.ckpt_file}",
        skip=args.skip_connection,
        ref_input=args.ref_input,
    )
    DC = DECA()

    # must file type : mp4
    driving_clip_names = sorted(os.listdir(args.driving_clip_path))
    source_clip_names = sorted(os.listdir(args.source_clip_path))
    for driving_clip_crop_name in driving_clip_names:
        for source_clip_crop_name in source_clip_names:
            args = util.setting_main_init(
                args,
                driving_clip_crop_name,
                source_clip_crop_name,
            )

            # 5 프레임씩 분할

            for (
                source_original_frames,
                source_faces,
                source_deca_params,
                driving_faces,
                driving_deca_params,
                driving_trans_landmark2ds,
            ) in zip(util.setting_batch_data(args)):
                print("test")

            results = run(args)
            # min 필요.
            vis(args, results)
