import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torchvision import transforms


def get_grad_mask(size=512):
    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr1 = np.sqrt(x_axis**4 + y_axis**4)

    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr2 = np.sqrt(x_axis**2 + y_axis**2)

    grad_mask = np.clip(1 - (arr1 / 2 + arr2 / 2), 0, 1)
    return grad_mask


def landmark_smoothing(landmarks):
    sm_landmarks = gaussian_filter1d(landmarks, sigma=1.0, axis=0)
    sm_landmarks = np.reshape(sm_landmarks, (-1, 68, 2))
    return sm_landmarks


def get_lmk_imgs(
    batch_lmk, color=(1, 1, 1), size=2, image_size=256, types="full", device="cuda"
):
    lmk_imgs = []
    for lmk in batch_lmk:
        canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        if types == "full":
            for lmk_ in lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        else:
            for lmk_ in lmk[:17]:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

            indexes = [
                27,
                28,
                29,
                30,
                31,
                33,
                35,
                49,
                51,
                53,
                61,
                63,
                67,
                65,
                55,
                57,
                59,
            ]
            _lmk = lmk[indexes, :]
            for lmk_ in _lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        lmk_imgs.append(np.expand_dims(canvas.transpose(2, 0, 1), axis=0))
    lmk_imgs = np.concatenate(lmk_imgs, axis=0)
    return torch.from_numpy(lmk_imgs).to(device)


def transfer_lip_params(s_flame_params, d_flame_params):
    lipysnc_flame_params = {}
    for key in s_flame_params.keys():
        lipysnc_flame_params[key] = s_flame_params[key].clone()

    lipysnc_flame_params["pose"][0][3] = d_flame_params["pose"][0][3]
    # lipysnc_flame_params["pose"][0][4] = d_flame_params["pose"][0][4]
    # lipysnc_flame_params["pose"][0][5] = d_flame_params["pose"][0][5]
    lipysnc_flame_params["exp"] = d_flame_params["exp"]
    # import pdb;pdb.set_trace()
    return lipysnc_flame_params


def get_convexhull_mask(batch_lmk, dilate_iter=15, image_size=256, device="cuda"):
    masks = []
    for lmk in batch_lmk:
        kernel = np.ones((3, 3), np.uint8)
        canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        points = np.array(lmk[1:15], np.int32)
        skin_mask = cv2.fillConvexPoly(canvas, points=points, color=(1, 1, 1))
        dilation_skin_mask = cv2.dilate(skin_mask, kernel, iterations=dilate_iter)
        masks.append(np.expand_dims(dilation_skin_mask.transpose(2, 0, 1), axis=0))
    masks = np.concatenate(masks, axis=0)
    return torch.from_numpy(masks).to(device)


def get_lmk_imgs(
    batch_lmk, color=(1, 1, 1), size=2, image_size=256, types="full", device="cuda"
):
    lmk_imgs = []
    for lmk in batch_lmk:
        canvas = np.zeros((image_size, image_size, 3)).astype(np.uint8)
        if types == "full":
            for lmk_ in lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        else:
            for lmk_ in lmk[:17]:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

            indexes = [
                27,
                28,
                29,
                30,
                31,
                33,
                35,
                49,
                51,
                53,
                61,
                63,
                67,
                65,
                55,
                57,
                59,
            ]
            _lmk = lmk[indexes, :]
            for lmk_ in _lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)

        lmk_imgs.append(np.expand_dims(canvas.transpose(2, 0, 1), axis=0))
    lmk_imgs = np.concatenate(lmk_imgs, axis=0)
    return torch.from_numpy(lmk_imgs).to(device)
