import argparse
import glob
import os

import cv2
import numpy as np
from innerverz import DECA
from package.LipSync import LipSyncer
from tqdm import tqdm
from utils import util, util_infer


def save(opts, results, inputs, tmp):
    os.makedirs(os.path.join(args.save_path, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "result_faces"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "result_frames"), exist_ok=True)

    tfm_invs = np.load(opts.source_tfm_inv_path)
    for frame_path, tfm_inv, result, _input, _tmp in zip(
        opts.source_frame_paths, tfm_invs, results, inputs, tmp
    ):
        frame_name = os.path.basename(frame_path)
        frame = cv2.imread(frame_path)
        blend_frame = util.get_vis(frame, result, tfm_inv)

        cv2.imwrite(os.path.join(opts.save_path, "inputs", frame_name), _input)
        cv2.imwrite(os.path.join(opts.save_path, "results", frame_name), _tmp)
        cv2.imwrite(os.path.join(opts.save_path, "result_faces", frame_name), result)

        cv2.imwrite(
            os.path.join(opts.save_path, "result_frames", frame_name), blend_frame
        )


def run(opts, generators, DC):
    result_all, input_all, tmp = [], [], []
    for sv_face_paths, sv_decas, lipsync_decas, mels in tqdm(zip(*generators)):
        sv_face_batch = util_infer.get_batch_image_from_path(opts, sv_face_paths)

        # get lmks
        sv_lmks = util_infer.get_deca_lmks(DC, sv_decas)
        lipsync_lmks = util_infer.get_deca_lmks(DC, lipsync_decas)

        # get convexhull
        sv_masks = util_infer.get_convexhull_mask(
            sv_lmks, dilate_iter=opts.dilate_iter, device="cpu"
        )
        lipsync_masks = util_infer.get_convexhull_mask(
            lipsync_lmks, dilate_iter=opts.dilate_iter, device="cpu"
        )
        mask = sv_masks | lipsync_masks
        dilate_mask = util.get_blend_mask(np.array(mask.permute([0, 2, 3, 1])))
        lipsync_lmks_vis = util_infer.get_lmk_imgs(
            lipsync_lmks, types="sparse", device="cpu"
        )
        input_images = sv_face_batch * (1 - mask) + mask * lipsync_lmks_vis

        _input_images = input_images.to("cuda")
        _mels = mels.to("cuda")
        _sv_face_batch = sv_face_batch.to("cuda")
        results = LS(_input_images, _sv_face_batch, _mels)
        _results = results.cpu() * dilate_mask + sv_face_batch * (1 - dilate_mask)

        results = (
            results.clone().detach().permute([0, 2, 3, 1]).cpu().numpy()[:, :, :, ::-1]
            * 127.5
            + 127.5
        )
        tmp.extend(results)

        _results = (
            _results.clone().detach().permute([0, 2, 3, 1]).cpu().numpy()[:, :, :, ::-1]
            * 127.5
            + 127.5
        )
        result_all.extend(_results)
        _results = (
            _input_images.clone()
            .detach()
            .permute([0, 2, 3, 1])
            .cpu()
            .numpy()[:, :, :, ::-1]
            * 127.5
            + 127.5
        )
        input_all.extend(_results)
    return result_all, input_all, tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # inference options
    parser.add_argument("--pp_path", type=str, default="./assets/demo_crop_pp")
    parser.add_argument("--video_path", type=str, default="./assets/demo_crop_videos")
    parser.add_argument("--save_root", type=str, default="./peronsal_4k/")

    # video options
    parser.add_argument("--sv_to", type=int, default=0)
    parser.add_argument("--sv_from", type=int, default=10)
    parser.add_argument("--dv_to", type=int, default=0)
    parser.add_argument("--dv_from", type=int, default=10)

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dilate_iter", type=int, default=5)

    # lipsync model options
    parser.add_argument("--frame_amount", type=int, default=5)
    parser.add_argument("--hubert_amount", type=int, default=9)
    parser.add_argument("--ckpt_file", type=str, default="Larissa_4k.pt")
    parser.add_argument("--skip_connection", type=bool, default=False)
    parser.add_argument("--ref_input", type=bool, default=False)

    args = parser.parse_args()

    LS = LipSyncer(
        ckpt_path=f"./package/LipSync/ckpts/{args.ckpt_file}",
        skip=args.skip_connection,
        ref_input=args.ref_input,
    )
    DC = DECA()

    # must file type : mp4
    # driving_clip_names = sorted(os.listdir(args.pp_path))
    driving_clip_names = [
        "Larissa_sync_1_crop",
        "Larissa_sync_2_crop",
        "Larissa_sync_3_crop",
    ]
    # source_clip_names = sorted(os.listdir(args.pp_path))
    source_clip_names = [
        "Larissa_sync_1_crop",
        "Larissa_sync_2_crop",
        "Larissa_sync_3_crop",
    ]
    for driving_clip_crop_name in driving_clip_names:
        for source_clip_crop_name in source_clip_names:
            if driving_clip_names == source_clip_crop_name:
                continue
            args.dv_name, args.sv_name = driving_clip_crop_name, source_clip_crop_name

            args.save_path = os.path.join(
                args.save_root, f"{source_clip_crop_name}_{driving_clip_crop_name}"
            )
            (
                args.source_frame_paths,
                args.source_face_paths,
                args.source_deca_path,
                args.source_face_bool_path,
                _,
                args.source_tfm_inv_path,
            ) = util.get_video_info(args, source_clip_crop_name)

            (
                args.driving_frame_paths,
                _,
                args.driving_deca_path,
                _,
                args.driving_mel_path,
                _,
            ) = util.get_video_info(args, driving_clip_crop_name)

            (
                source_deca_params,
                lipsync_deca_params,
            ) = util.get_lipsync_deca_param(args)

            min_duration = min(
                len(args.source_frame_paths), len(args.driving_frame_paths)
            )
            generators = util.set_generators(
                args, source_deca_params, lipsync_deca_params, min_duration
            )

            results, inputs, tmp = run(args, generators, DC)
            # min 필요.
            save(args, results, inputs, tmp)
            util.video_save(args)
