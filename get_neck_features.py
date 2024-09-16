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

max_error = 0
def pre_transform(im, opt):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(opt.imgs, auto=same_shapes, stride=opt.stride)
    return [letterbox(image=x) for x in im]

def preprocess(im, opt):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = pre_transform(im, opt)
        im = np.stack(im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(opt.device)
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

        ## check quantization error
        if i ==0:
            global max_error
            max_error = max(max_error, abs(min(ori_shape[0],ori_shape[1])  / max(ori_shape[0],ori_shape[1]) - min(unpad.shape[2], unpad.shape[3]) / max(unpad.shape[2], unpad.shape[3])))
    return neck_features_unpad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='/mnt/data/lanxing/UTK-face-wild/images/', type=str, help='source')
    parser.add_argument('--weights', type=str, default='yolov8n-face.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--stride', type=int, default=32, help='stride')
    opt = parser.parse_args()
    
    device = select_device(opt.device, batch=opt.batch)
    
    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = attempt_load_weights(weights, device=device, fuse=True)
        model.fuse()
        print(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."

    model = model.to(device)

    img_names = sorted(os.listdir(opt.source))
    img_paths = [os.path.join(opt.source, img_name) for img_name in img_names]
    neck_unpad_features_list = []
    for img_path in tqdm(img_paths[:100]):
        img = cv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image Not Found {img_path}")
        ori_shape = img.shape[:2]  # (h, w, c)
        img = preprocess([img], opt)
        y, neck = model(img) #neck : [[1,64,80,64], [1,128,40,32], [1,256,20,16]]
        neck_unpad_features = unpad_neck(neck, ori_shape)  # [[1,64,80,60], [1,128,40,30], [1,256,20,16]]
        neck_unpad_features_list.append(neck_unpad_features)

    print(f"Max error: {max_error}")