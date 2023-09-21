import glob
import math
import os

import cv2
import librosa
import numpy as np
import torch

from .infer_util import get_grad_mask


# SETTTINGS
def setting_pp_init(opts):
    opts.video_name = os.path.basename(opts.video_file_path).split(".")[0]
    opts.video_path = opts.video_file_path
    opts.pp_save_path = os.path.join(
        opts.pp_save_root,
        opts.type,
        f"{opts.video_name}",
    )

    if opts.type == "source":
        opts.workflow = {
            "sync_offset": False,
            "frames": True,
            "face": True,
            "deca_params": True,
            "flame_params": False,
        }
    else:
        opts.workflow = {
            "sync_offset": True,
            "frames": True,
            "face": True,
            "deca_params": True,
            "flame_params": True,
        }

    opts.frame_save_path = os.path.join(opts.pp_save_path, "frames")
    opts.face_save_path = os.path.join(opts.pp_save_path, "faces")
    opts.trans_landmark2d_save_path = os.path.join(
        opts.pp_save_path, "trans_landmark2d.npy"
    )
    opts.face_bool_save_path = os.path.join(opts.pp_save_path, "face_bool.npy")
    opts.lmks_save_path = os.path.join(opts.pp_save_path, "lmks_106.npy")
    opts.tfm_inv_save_path = os.path.join(opts.pp_save_path, "tfm_inv.npy")
    opts.deca_param_save_path = os.path.join(opts.pp_save_path, "deca_params.npy")

    opts.audio_save_path = os.path.join(opts.pp_save_path, "audio.wav")
    opts.mel_save_path = os.path.join(opts.pp_save_path, "mel.npy")

    os.makedirs(opts.pp_save_path, exist_ok=True)
    os.makedirs(opts.frame_save_path, exist_ok=True)
    os.makedirs(opts.face_save_path, exist_ok=True)

    return opts


# TODO : single mode
def get_file_list(
    opts,
    sv_path,
    dv_path,
    folder_name,
    sv_start_num,
    sv_end_num,
    dv_start_num,
    dv_end_num,
):
    sv_paths = sorted(glob.glob(os.path.join(sv_path, folder_name, "*.*")))
    dv_paths = sorted(glob.glob(os.path.join(dv_path, folder_name, "*.*")))
    _sv_paths, _dv_paths = (
        sv_paths[sv_start_num:sv_end_num],
        dv_paths[dv_start_num:dv_end_num],
    )
    path_length = min(len(_sv_paths), len(_dv_paths))

    _sv_paths, _dv_paths = _sv_paths[:path_length], _dv_paths[:path_length]
    opts.video_duration = path_length / opts.fps

    opts.sv_end_sec = int(opts.sv_start_sec + opts.video_duration)
    opts.dv_end_sec = int(opts.dv_start_sec + opts.video_duration)

    return _sv_paths, _dv_paths


def dict_to_device(dicts, device="cuda"):
    device_dicts = {}
    for key in dicts.keys():
        if device == "cuda":
            device_dicts[key] = torch.from_numpy(dicts[key]).unsqueeze(0).to(device)
        elif device == "cpu":
            device_dicts[key] = dicts[key].squeeze(0).clone().detach().numpy()

    return device_dicts


def setting_main_init(
    opts,
    driving_clip_crop_name,
    source_clip_crop_name,
):
    opts.driving_video_name = driving_clip_crop_name[:-8]
    opts.source_video_name = source_clip_crop_name[:-8]

    opts.driving_clip_crop_name, opts.source_clip_crop_name = (
        driving_clip_crop_name,
        source_clip_crop_name,
    )

    (
        opts.driving_video_start,
        opts.driving_video_end,
    ) = opts.driving_clip_crop_name.split("_")[-2:]
    opts.source_video_start, opts.source_video_end = opts.source_clip_crop_name.split(
        "_"
    )[-2:]
    driving_video_duration = abs(
        float(opts.driving_video_end) - float(opts.driving_video_start)
    )
    source_video_duration = abs(
        float(opts.source_video_end) - float(opts.source_video_start)
    )
    opts.min_duration = min(driving_video_duration, source_video_duration)

    # driving
    opts.driving_face_path = os.path.join(
        opts.driving_clip_path, opts.driving_clip_crop_name, "faces"
    )
    opts.driving_deca_params_path = os.path.join(
        opts.driving_clip_path, opts.driving_clip_crop_name, "deca_params.npy"
    )
    opts.driving_mel_path = os.path.join(
        opts.driving_clip_path, opts.driving_clip_crop_name, "mel.npy"
    )
    opts.driving_trans_landmark2d_path = os.path.join(
        opts.driving_clip_path, opts.driving_clip_crop_name, "trans_landmark2d.npy"
    )

    # source
    opts.source_deca_params_path = os.path.join(
        opts.source_clip_path, opts.source_clip_crop_name, "deca_params.npy"
    )
    opts.source_face_bool_path = os.path.join(
        opts.source_clip_path, opts.source_clip_crop_name, "face_bool.npy"
    )
    opts.source_tfm_inv_path = os.path.join(
        opts.source_clip_path, opts.source_clip_crop_name, "tfm_inv.npy"
    )
    opts.source_original_frame_path = os.path.join(
        opts.source_clip_path, opts.source_clip_crop_name, "frames"
    )
    opts.source_faces_path = os.path.join(
        opts.source_clip_path, opts.source_clip_crop_name, "faces"
    )
    # face, landmark2d_points, deca_params, flame_params
    # source video
    # face, tfm inv, original frame, deca_params, flame_params

    opts.workspace_path = os.path.join(
        opts.save_path,
        f"{opts.driving_video_name}_{str(opts.driving_video_start).zfill(3)}_{str(int(opts.driving_video_start) + int(opts.min_duration)).zfill(3)}_{opts.source_video_name}_{str(opts.source_video_start).zfill(3)}_{str(int(opts.source_video_start) + int(opts.min_duration)).zfill(3)}",
    )
    opts.result_faces_path = os.path.join(opts.workspace_path, "results")
    opts.result_frames_path = os.path.join(opts.workspace_path, "result_frames")

    os.makedirs(opts.workspace_path, exist_ok=True)
    os.makedirs(opts.result_faces_path, exist_ok=True)
    os.makedirs(opts.result_frames_path, exist_ok=True)

    return opts


# def setting_batch_data(opts):
#     # need
#     # source : original frame, faces, deca, tfm_inv
#     source_original_frame_list = sorted(
#         glob.glob(os.path.join(opts.source_original_frame_path, "*.*"))
#     )
#     source_face_list = sorted(glob.glob(os.path.join(opts.source_faces_path, "*.*")))
#     source_deca_list = np.load(opts.source_deca_params_path, allow_pickle=True)
#     source_tfm_inv_list = np.load(opts.source_tfm_inv_path, allow_pickle=True)

#     # driving : mel, faces, deca, landmarks2d_points
#     driving_face_list = sorted(glob.glob(os.path.join(opts.driving_face_path, "*.*")))
#     driving_deca_param_list = np.load(opts.driving_deca_params_path, allow_pickle=True)
#     driving_mel = np.load(opts.driving_mel_path, allow_pickle=True)
#     driving_trans_landmark2d_list = np.load(
#         opts.driving_trans_landmark2d_path, allow_pickle=True
#     )
#     import pdb

#     frame_len = int(opts.min_duration * opts.fps)

#     (
#         source_original_frame_batches,
#         source_faces_batches,
#         source_deca_batches,
#         source_tfm_inv_batches,
#     ) = ([], [], [], [])
#     (
#         driving_faces_batches,
#         driving_deca_param_batches,
#         driving_mel_batches,
#         driving_trans_landmark2d_batches,
#     ) = ([], [], [], [])
#     image_divide_indexes = list(range(0, frame_len, 5))
#     for image_start_index in image_divide_indexes:
#         image_end_index = image_start_index + opts.frame_amount
#         hubert_start_index = image_start_index * 2
#         hubert_end_index = hubert_start_index + opts.hubert_amount

#         # source
#         source_original_frame_batches.append(
#             source_original_frame_list[image_start_index:image_end_index]
#         )
#         source_faces_batches.append(source_face_list[image_start_index:image_end_index])
#         source_deca_batches.append(source_deca_list[image_start_index:image_end_index])
#         source_tfm_inv_batches.append(
#             source_tfm_inv_list[image_start_index:image_end_index]
#         )

#         # driving
#         driving_faces_batches.append(
#             driving_face_list[image_start_index:image_end_index]
#         )
#         driving_deca_param_batches.append(
#             driving_deca_param_list[image_start_index:image_end_index]
#         )
#         driving_trans_landmark2d_batches.append(
#             driving_trans_landmark2d_list[image_start_index:image_end_index]
#         )
#         driving_mel_batches.append(
#             driving_deca_param_list[hubert_start_index:hubert_end_index]
#         )
#         pdb.set_trace()
#         # TODO : 이거 5개씩 나눈거 체크 해야됨.
#     return


def get_vis(frame, face, lmk_img, mask_img, FA):
    size = frame.shape
    FA_dict = FA.get_face(frame)
    tfm_inv = FA_dict["tfm_inv"]
    grad_mask = (get_grad_mask(256) * 3).clip(0, 1)

    # vis
    lmk_vis = (face * (1 - lmk_img) + face * lmk_img * 0.1) + lmk_img * 255 * 0.9
    mask_vis = (face * (1 - mask_img) + face * mask_img * 0.8) + mask_img * 255 * 0.2

    warp_face = cv2.warpAffine(face, tfm_inv, (size[1], size[0]))
    warp_lmk_img = cv2.warpAffine(lmk_vis, tfm_inv, (size[1], size[0]))
    warp_mask_img = cv2.warpAffine(mask_vis, tfm_inv, (size[1], size[0]))
    warp_grad_mask = cv2.warpAffine(grad_mask, tfm_inv, (size[1], size[0]))[:, :, None]

    # get replaced image
    blend_face_img = warp_grad_mask * warp_face + (1 - warp_grad_mask) * frame
    blend_lmk_img = warp_grad_mask * warp_lmk_img + (1 - warp_grad_mask) * frame
    blend_mask_img = warp_grad_mask * warp_mask_img + (1 - warp_grad_mask) * frame
    return blend_face_img, blend_lmk_img, blend_mask_img


def video_save(opts):
    # os.system(
    #     # f"ffmpeg -y -i assets/sync_audio/{opts.audio_name}.wav -ss {opts.dv_start_sec} -to {opts.dv_end_sec} ./audio_tmp.wav"
    #     f"ffmpeg -y -i assets/sync_audio/{opts.dv_name}.wav -ss {opts.dv_start_sec} -to {opts.dv_end_sec} ./audio_tmp.wav"
    # )
    # Korean / DEMO dataset
    os.system(
        f"ffmpeg -y -i {opts.dv_dataset_path+'-audio'}/{opts.dv_name}.wav -ss {opts.dv_start_sec} -to {opts.dv_end_sec} ./{opts.ckpt_file_name}_audio_tmp.wav"
    )
    # lmk
    os.system(
        f"ffmpeg -y -i {opts.lmks_vis_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.vis_lmks_video_path}/{opts.sv_name}_{str(opts.sv_start_sec).zfill(2)}to{str(opts.sv_end_sec).zfill(2)}_{opts.dv_name}_{str(opts.dv_start_sec).zfill(2)}to{str(opts.dv_end_sec).zfill(2)}.mp4"
    )
    # mask
    os.system(
        f"ffmpeg -y -i {opts.mask_vis_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.vis_mask_video_path}/{opts.sv_name}_{str(opts.sv_start_sec).zfill(2)}to{str(opts.sv_end_sec).zfill(2)}_{opts.dv_name}_{str(opts.dv_start_sec).zfill(2)}to{str(opts.dv_end_sec).zfill(2)}.mp4"
    )
    # result
    os.system(
        f"ffmpeg -y -i {opts.grid_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.result_video_path}/{opts.sv_name}_{str(opts.sv_start_sec).zfill(2)}to{str(opts.sv_end_sec).zfill(2)}_{opts.dv_name}_{str(opts.dv_start_sec).zfill(2)}to{str(opts.dv_end_sec).zfill(2)}.mp4"
    )
    # test
    # os.system(f"ffmpeg -y -i {opts.test_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.test_video_path}/{opts.sv_name}_{opts.dv_name}.mp4")
    os.system(f"rm ./{opts.ckpt_file_name}_audio_tmp.wav")


def putText_lmk_dist(text, lmks_img, lmks):
    _lmks_img = lmks_img.copy()
    _lmks_img = cv2.putText(
        _lmks_img,
        f"{text}",
        (int(15), int(45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
    )
    # _lmks_img = cv2.putText(_lmks_img, f"x : {str(int(math.dist(lmks[54],lmks[48])))}", (int(15),int(125)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
    # _lmks_img = cv2.putText(_lmks_img, f"y : {str(int(math.dist(lmks[57],lmks[51])))}", (int(15),int(205)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    return _lmks_img


# audio pp
def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4) * ((S + 100) / 100) - 4, -4, 4)


def _linear_to_mel(spectogram, sr=16000):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(sr)
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis(sr=16000, n_fft=800, num_mels=80, fmin=55, fmax=7600):
    assert 7600 <= sr // 2
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )


def get_mel(audio, n_fft=800, sr=16000):
    D = librosa.stft(y=audio, n_fft=n_fft, hop_length=200, win_length=800)
    S = _amp_to_db(_linear_to_mel(np.abs(D), sr)) - 20
    mel = _normalize(S)
    return mel
