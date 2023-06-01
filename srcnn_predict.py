import os

import matplotlib.pyplot as plt

from matlab import imresize, convertDouble2Byte as d2int
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb
import numpy as np
import torch
from utils import PSNR, rgb2y_uint8, SSIM
from torchvision.utils import save_image
from torchsummary import summary as summary

from PIL import Image

# import the network we want to predict for
from model_4 import Net

testWord = 'bird'   # line22, 134

def val_psnr(model, th, dilker, dilation, val_path, scale, boundary, psnrs):
    images = sorted(os.listdir(val_path))
    # images = [s for s in images if testWord in s] #############
    avg_psnr = 0
    avg_ssim = 0
    model.eval()
    len_image = 0
    with torch.no_grad():
        for image in images:
            if image.endswith('txt'): # exists 2 type of file (.jgp, .txt)
                continue
            img = imread(val_path + image)
            # img = np.arange(24).reshape(4, 2, 3) # for debugging
            cbcr = None
            len_image += 1 # len(images) = jpg + txt, but need only jpg's count
            try:
                cbcr = rgb2ycbcr(img)[:, :, 1:3]
                y = rgb2ycbcr(img)[:, :, 0:1]
            except: # if image channel is 1,
                cbcr = np.zeros((img.shape[0], img.shape[1], 2))
                y = np.expand_dims(img, axis=-1)

            y = np.float64(y) / 255.
            height, width, channel = y.shape

            hr_y = y[0:height - (height % scale), 0: width - (width % scale), :]
            hr_cbcr = cbcr[0:height - (height % scale), 0: width - (width % scale), :]
            lr = imresize(hr_y, scalar_scale=1 / scale, method='bicubic')
            
            lr = np.moveaxis(lr, -1, 0)  # 텐서 계산을 위해 차원축 이동
            lr = np.expand_dims(lr, axis=0)  # 텐서 계산을 위해 차원 확장
            lr = torch.from_numpy(lr).float().to(device)
            hr_cbcr = np.moveaxis(hr_cbcr, -1, 0)
            
            out = model(lr, th=th)
            output = out.cuda().data.cpu()
            
            ################################# main changes #################################
            result = np.concatenate((output[0] * 255, hr_cbcr), axis=0)
            result = np.moveaxis(result, 0, -1)
            
            sr_image = ycbcr2rgb(result) # ISSUE: negative value (range is -0.5xx ~ 1.xx)
            sr_image = d2int(sr_image)
            sr_image = Image.fromarray(sr_image) # <class 'numpy.ndarray'>
            sr_image.save(f"/home/user/Desktop/yolov5-face/data/widerface/srcnn/train/{image}")
            ################################################################################

            output = output.numpy()

            hr_y = d2int(hr_y)
            output = d2int(output)

            output = output[0]
            output = np.moveaxis(output, 0, -1)

            avg_ssim +=SSIM(output, hr_y, boundary)
            avg_psnr += PSNR(hr_y, output, boundary=boundary)
            print(SSIM(output, hr_y, scale), "/", PSNR(hr_y, output, boundary=boundary))
            psnrs.append(PSNR(hr_y, output, boundary=boundary))

    print(round(avg_ssim/len_image, 4), end=' ')
    return avg_psnr/len_image, avg_ssim/len_image


def predict(datasets, model_paths, r=[4], th=[0.04]):


    sets = datasets #['set5/'] #, 'Set14/', 'BSDS100/', 'Urban100/', 'DIV2K_val/'''
    dir_n = 'outputs/ASCNN/'

    paths = model_paths # ['baseline']

    th = th
    dilation= True
    dilker = 3

    scale = 2
    r = r
    boundary = scale
    model = Net(scale, r=r).float().to(device)
    print('Number of Parameters:', sum(p.numel() for p in model.parameters()))

    # for name, param in model.named_parameters():
    #     if "low_par" in name and "weight" in name:
    #         print(name)
    #         print(param.size())


    '''
    the code below (from line 102) is to test for different models quickly at the same time
    try to run without this code
    just use: 
    ckpnt = torch.load(path) # load weight path
    model.load_state_dict(ckpnt) # load the weights to model
    avg_psnr, avg_ssim = val_psnr(model, th, dilker, dilation, val_path, scale) # evaluate model with validation path
    '''
    psnrs = []
    for set in sets:
        for idx, path in enumerate(paths):
            path = dir_n + path + '.pth'
            ckpnt = torch.load(path)
            model.load_state_dict(ckpnt)
            val_path = '/home/user/Desktop/yolov5-face/data/widerface/train/'
            # val_path = './TestSet/' + set

            result1, result2 = val_psnr(model, th, dilker, dilation, val_path, scale, boundary, psnrs)
            print('avg PSNR:', round(result1, 5), end='/')
            print('avg SSIM:', round(result2, 4))

    return psnrs


if __name__=="__main__":
    device = torch.device('cuda')

    set = 'Set14/' # .pth file exists in set
    psnr_4path = predict([set], ['4path2'], r=[2, 4, 8], th=[0.075, 0.035, 0.013])

    images = sorted(os.listdir('./TestSet/'+set))
    # images = [s for s in images if testWord in s]

