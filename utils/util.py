import glob
import math
import os

import cv2
import librosa
import numpy as np
import torch


def get_grad_mask(size=512):
    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr1 = np.sqrt(x_axis**4 + y_axis**4)

    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr2 = np.sqrt(x_axis**2 + y_axis**2)

    grad_mask = np.clip(1 - (arr1 / 2 + arr2 / 2), 0, 1)
    return grad_mask


# SETTTINGS
def setting_pp_init(opts):
    opts.video_name = os.path.basename(opts.video_file_path).split(".")[0]
    opts.video_path = opts.video_file_path
    opts.pp_save_path = os.path.join(
        opts.pp_save_root,
        f"{opts.video_name}",
    )

    opts.frame_save_path = os.path.join(opts.pp_save_path, "frames")
    opts.face_save_path = os.path.join(opts.pp_save_path, "faces")
    opts.trans_landmark2d_save_path = os.path.join(
        opts.pp_save_path, "trans_landmark2d.npy"
    )
    opts.face_bool_save_path = os.path.join(opts.pp_save_path, "face_bool.npy")
    opts.lmks_save_path = os.path.join(opts.pp_save_path, "lmks.npy")
    opts.tfm_inv_save_path = os.path.join(opts.pp_save_path, "tfm_inv.npy")
    opts.deca_param_save_path = os.path.join(opts.pp_save_path, "deca_params.npy")

    opts.audio_save_path = os.path.join(opts.pp_save_path, "audio.wav")
    opts.mel_save_path = os.path.join(opts.pp_save_path, "mel.npy")

    os.makedirs(opts.pp_save_path, exist_ok=True)
    os.makedirs(opts.frame_save_path, exist_ok=True)
    os.makedirs(opts.face_save_path, exist_ok=True)

    return opts


# TODO : single mode
def get_video_info(
    opts,
    folder_path,
    folder_name,
):
    frame_paths = sorted(
        glob.glob(os.path.join(folder_path, folder_name, "frames/*.*"))
    )
    face_paths = sorted(glob.glob(os.path.join(folder_path, folder_name, "faces/*.*")))
    deca_path = os.path.join(folder_path, folder_name, "deca_params.npy")
    face_bool_path = os.path.join(folder_path, folder_name, "face_bool.npy")
    mel_path = os.path.join(folder_path, folder_name, "mel.npy")
    tfm_inv_path = os.path.join(folder_path, folder_name, "tfm_inv.npy")
    return (
        frame_paths,
        face_paths,
        deca_path,
        face_bool_path,
        mel_path,
        tfm_inv_path,
    )


def dict_to_device(dicts, device="cuda"):
    device_dicts = {}
    for key in dicts.keys():
        if device == "cuda":
            device_dicts[key] = torch.from_numpy(dicts[key]).unsqueeze(0).to(device)
        elif device == "cpu":
            device_dicts[key] = dicts[key].squeeze(0).clone().detach().numpy()

    return device_dicts


def get_vis(frame, face, tfm_inv):
    size = frame.shape
    grad_mask = (get_grad_mask(256) * 3).clip(0, 1)

    warp_face = cv2.warpAffine(face.copy(), tfm_inv, (size[1], size[0]))
    warp_grad_mask = cv2.warpAffine(grad_mask, tfm_inv, (size[1], size[0]))[:, :, None]

    # get replaced image
    blend_face_img = warp_grad_mask * warp_face + (1 - warp_grad_mask) * frame
    return blend_face_img


def video_save(opts):
    os.makedirs(f"{opts.save_path}/../result_videos", exist_ok=True)
    # os.system(
    #     # f"ffmpeg -y -i assets/sync_audio/{opts.audio_name}.wav -ss {opts.dv_start_sec} -to {opts.dv_end_sec} ./audio_tmp.wav"
    #     f"ffmpeg -y -i assets/sync_audio/{opts.dv_name}.wav -ss {opts.dv_start_sec} -to {opts.dv_end_sec} ./audio_tmp.wav"
    # )
    # # lmk
    # os.system(
    #     f"ffmpeg -y -i {opts.lmks_vis_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.vis_lmks_video_path}/{opts.sv_name}_{str(opts.sv_start_sec).zfill(2)}to{str(opts.sv_end_sec).zfill(2)}_{opts.dv_name}_{str(opts.dv_start_sec).zfill(2)}to{str(opts.dv_end_sec).zfill(2)}.mp4"
    # )
    # # mask
    # os.system(
    #     f"ffmpeg -y -i {opts.mask_vis_save_path}/%06d.png -i ./{opts.ckpt_file_name}_audio_tmp.wav -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.vis_mask_video_path}/{opts.sv_name}_{str(opts.sv_start_sec).zfill(2)}to{str(opts.sv_end_sec).zfill(2)}_{opts.dv_name}_{str(opts.dv_start_sec).zfill(2)}to{str(opts.dv_end_sec).zfill(2)}.mp4"
    # )
    # result
    os.system(
        f"ffmpeg -y -i {os.path.join(opts.save_path, 'result_frames')}/%06d.png -i {os.path.join(opts.video_path, opts.dv_name+'.mp4')} -ss 0 -to {opts.min_duration/25} -r {opts.fps} -map 0:v -map 1:a -vb 20M -y {opts.save_path}/../result_videos/{opts.sv_name}_{opts.dv_name}.mp4"
    )
    # test


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


def get_lipsync_deca_param(opts):
    driving_deca_params = np.load(opts.driving_deca_path, allow_pickle=True)
    source_deca_params = np.load(opts.source_deca_path, allow_pickle=True)
    # load pp data
    lipsync_deca_params = []
    for driving_deca_param, source_deca_param in zip(
        driving_deca_params, source_deca_params
    ):
        lipsync_flame_params = transfer_lip_params(
            source_deca_param, driving_deca_param
        )
        lipsync_deca_params.append(lipsync_flame_params)

    return source_deca_params, lipsync_deca_params


def transfer_lip_params(s_deca_params, d_deca_params):
    lipysnc_deca_params = {}
    for key in s_deca_params.keys():
        lipysnc_deca_params[key] = s_deca_params[key].clone()

    lipysnc_deca_params["pose"][0][3] = d_deca_params["pose"][0][3]
    lipysnc_deca_params["exp"] = d_deca_params["exp"]
    return lipysnc_deca_params


# generator
def get_batch_size_data(data, batch_size=5):
    if len(data) % 5 != 0:
        data = data[: -(len(data) % 5)]
    for index in range(0, len(data), batch_size):  # 5
        yield data[index : index + batch_size]


# audio generator
# def get_batch_size_mel_data(data, mel_size=16):
#     for index in range(0, data.shape[1], mel_size):
#         yield data[:, index : index + mel_size]
def get_batch_size_mel_data(data_path, duration, mel_size=16):  ### 0.2s
    data = torch.tensor(np.load(data_path), dtype=torch.float)[:duration]
    data = torch.cat([data[:, :1], data], dim=-1)  # 처음 프레임 음성을 하나 복사함
    if data.shape[1] % 5 != 0:
        data = data[:, : -(data.shape[1] % 5)]
    for index in range(1, data.shape[1], mel_size):  # 10
        stack = []
        for a_index in range(5):
            segment_data = data[
                None, :, index - 1 + a_index : index - 1 + a_index + mel_size
            ]
            if segment_data.shape[-1] != 16:
                pad_amount = 16 - segment_data.shape[-1]
                pad_value = segment_data[:, :, -1:].repeat(1, 1, pad_amount)
                segment_data = torch.cat([segment_data, pad_value], dim=-1)
            stack.append(segment_data)
        yield torch.stack(stack, dim=0).reshape(-1, 1, 80, mel_size)  # 5 80 16


def set_generators(opts, source_deca_params, lipsync_deca_params, min_duration):
    sv_face_generator = get_batch_size_data(opts.source_face_paths[:min_duration])
    sv_deca_generator = get_batch_size_data(source_deca_params[:min_duration])
    lipsync_deca_generator = get_batch_size_data(lipsync_deca_params[:min_duration])
    mel_generator = get_batch_size_mel_data(
        opts.driving_mel_path, int(min_duration / 25 * 80)
    )
    return (
        sv_face_generator,
        sv_deca_generator,
        lipsync_deca_generator,
        mel_generator,
    )


def get_blend_mask(
    masks,
    dilate_iter=0,
    blur_size=5,
):
    kernel = np.ones((3, 3), np.uint8)
    blur_masks = []
    for mask in masks:
        _mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        _mask = cv2.blur(np.array(_mask).astype(np.float64), (blur_size, blur_size))
        blur_masks.append(_mask)

    blur_masks = np.stack(blur_masks, axis=0)
    blur_masks_ts = torch.tensor(blur_masks).permute([0, 3, 1, 2])
    return blur_masks_ts
