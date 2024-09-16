import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import cv2 as cv
import numpy as np
import torch.utils.data
from tqdm import tqdm
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights

def pre_transform(im, shape=[640, 640], stride=32):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(shape, auto=same_shapes, stride=stride)
    return [letterbox(image=x) for x in im]

def preprocess(im):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = pre_transform(im)
        im = np.stack(im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    # im = im.to(device)
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

def unpad_neck(neck, ori_shape):
    neck_features_unpad = []
    for i in range(len(neck)):
        padh, padw = neck[i].shape[2], neck[i].shape[3]
    
        #compute aspect ratio
        ori_aspect_ratio = ori_shape[0] / ori_shape[1]  # h / w
        pad_aspect_ratio = padh / padw

        if ori_aspect_ratio > pad_aspect_ratio: # w was pad
            scale =  padh / ori_shape[0]
            unpad_w = round(ori_shape[1] * scale + 0.001)
            dw = (padw - unpad_w) / 2

            left, right = int(round(dw - 0.001)), int(round(dw + 0.001))
            assert left + right == padw - unpad_w, f'{left} + {right} != {padw} - {unpad_w}'
            unpad = neck[i][:, :, :, left : padw - right]
            neck_features_unpad.append(unpad.cpu().detach())

        else: # h was pad
            scale = padw / ori_shape[1]
            unpad_h = round(ori_shape[0] * scale + 0.001)
            dh = (padh - unpad_h) / 2
            top, bottom = int(round(dh - 0.001)), int(round(dh + 0.001))
            assert top + bottom == padh - unpad_h, f'{top} + {bottom} != {padh} - {unpad_h}'
            unpad = neck[i][:, :, top : padh - bottom, :]
            neck_features_unpad.append(unpad.cpu().detach())

    return neck_features_unpad

def init_face_model(weight_path="yolov8n-face.pt", device="cpu"):
    device = select_device(device, batch=1)
    model = attempt_load_weights(weight_path, device=device, fuse=True)
    model.fuse()
    model.to(device)
    return model


def get_feat_from_img_path(img_path, model):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image Not Found {img_path}")
    ori_shape = img.shape[:2]  # (h, w, c)
    img = preprocess([img])
    _, neck = model(img) # cpu default
    neck_unpad_features = unpad_neck(neck, ori_shape)
    return neck_unpad_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='/mnt/data/lanxing/UTK-face-wild/images/', type=str, help='source')
    parser.add_argument('--weights', type=str, default='yolov8n-face.pt', help='trained weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    model = init_face_model(opt.weights, opt.device)

    img_names = sorted(os.listdir(opt.source))
    img_paths = [os.path.join(opt.source, img_name) for img_name in img_names]
    neck_unpad_features_list = []
    for img_path in tqdm(img_paths[:100]):
        neck_unpad_features = get_feat_from_img_path(img_path, model)
        
