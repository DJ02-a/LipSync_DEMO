
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from nets import MyGenerator


class LipSyncer(nn.Module):
    def __init__(self, ckpt_path = 'ckpt/best.pt', skip=False, ref_input=False):
        super(LipSyncer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.G = MyGenerator(skip=skip, ref_input=ref_input).to(self.device)
        ckpt_dict_G = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.G.load_state_dict(ckpt_dict_G['model'], strict=False)
        self.G.eval()
        del ckpt_dict_G
        
        self.img_size = 256
        self.tf_color = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def pp_image(self, img_path, grayscale=False, flip=False):

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.tf_color(img)

        return img

    def get_convexhull_mask(self, faces, lmks, iter_num=15):
        faces = faces.clone().detach().cpu().numpy()
        lmks = lmks.clone().detach().cpu().numpy()
        
        masks = []
        for i in range(faces.shape[0]):
            face = faces[i].transpose(1,2,0)
            lmk = lmks[i]
            kernel = np.ones((3, 3), np.uint8)
            canvas = np.zeros_like(face.copy()).astype(np.uint8)
            points = np.array([lmk[2], lmk[3], lmk[4], lmk[5], lmk[6], lmk[7], lmk[8], lmk[9], lmk[10], lmk[11], lmk[12], lmk[13], lmk[14]], np.int32)
            skin_mask = cv2.fillConvexPoly(canvas, points=points, color=(1,1,1))
            dilation_skin_mask = cv2.dilate(skin_mask, kernel, iterations=iter_num)
            masks.append(np.expand_dims(dilation_skin_mask.transpose(2,0,1), axis=0))
        masks = np.concatenate(masks, axis=0)
        return torch.from_numpy(masks).to(self.device)
    
    def get_guide_image(self, faces, lmks, color=(1,1,1), size=2):
        faces = faces.clone().detach().cpu().numpy()
        lmks = lmks.clone().detach().cpu().numpy()
        
        lmk_imgs = []
        for i in range(faces.shape[0]):
            face = faces[i].transpose(1,2,0)
            lmk = lmks[i]
            canvas = np.zeros_like(face.copy()).astype(np.uint8)
            for lmk_ in lmk:
                cv2.circle(canvas, (int(lmk_[0]), int(lmk_[1])), 1, color, size)
            lmk_imgs.append(np.expand_dims(canvas.transpose(2,0,1), axis=0))
        lmk_imgs = np.concatenate(lmk_imgs, axis=0)
        return torch.from_numpy(lmk_imgs).to(self.device)
    
    def forward(self, images, ref_imgs, audio_feature, lmks, dil_iter_num=15, jaw_lmks=None):
        input_images = []
        masks = []
        for i in range(5):
            image = images[i].unsqueeze(0)
            lmk = lmks[i].unsqueeze(0)
            # for self-reconstruction code
            if jaw_lmks is None:
                mask = self.get_convexhull_mask(image, lmk, dil_iter_num)
                guide_image = self.get_guide_image(image, lmk)
            # for cross-id code
            else:
                jaw_lmk = jaw_lmks[i].unsqueeze(0)
                t_mask = self.get_convexhull_mask(image, lmk, dil_iter_num)
                jaw_mask = self.get_convexhull_mask(image, jaw_lmk, dil_iter_num)
                mask = t_mask | jaw_mask # target mask와 jaw mask의 모든부분 포함
                guide_image = self.get_guide_image(image, jaw_lmk)
            
            masked_face = image*(1-mask)
            masks.append(mask)
            input_images.append(guide_image*mask + masked_face)
        input_images = torch.cat(input_images, dim=0)

        output = self.G(input_images, ref_imgs, audio_feature)
        
        results = []
        for i in range(5):
            results.append(output[i]*masks[i] + images[i].unsqueeze(0)*(1-masks[i]))
        results = torch.cat(results, dim=0)
        
        return input_images, results