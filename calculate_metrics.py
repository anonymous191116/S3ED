import pytorch_ssim
import torch
from torch.autograd import Variable
import os
from PIL import Image
import torchvision.transforms as transforms
import os
from util.metrics import PSNR
# from ssim import SSIM
import pytorch_ssim
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ori_path='A_B/test/'
res_path='test_latest/images/'

def getimgcoco(str):
    rel=str[:25]+'_Real.png'
    fake=str[:25]+'_DeBlurred.png'
    return  rel,fake
def getimggopro(str):
    rel=str[:-4]+'_Real.png'
    fake=str[:-4]+'_DeBlurred.png'
    return  rel,fake

img_list=os.listdir(ori_path)
trans=transforms.Compose([transforms.ToTensor()])
SSIM_SUM = 0.
PSNR_SUM = 0.

process_num = 0
for img in img_list:
    relimg,fakeimg=getimgcoco(img)
    #relimg, fakeimg = getimggopro(img)
    img1 = Image.open(res_path + relimg).convert('RGB')
    img2 = Image.open(res_path+fakeimg).convert('RGB')
    PSNR_S = PSNR(np.array(img1).astype(float), np.array(img2).astype(float))
    PSNR_SUM += PSNR_S
    img1 = trans(img1).unsqueeze(0)
    img2 = trans(img2).unsqueeze(0)
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    ssim_loss=pytorch_ssim.ssim(img1, img2,window_size=16)
    SSIM_SUM+=ssim_loss.data
    process_num+=1
    if process_num%100 == 0:
        print("processing %5d/%5d"%(process_num,len(img_list)))

print('---average SSIM is %.5f:'%(SSIM_SUM/len(img_list)))
print('---average PSNR is %.5f:'%(PSNR_SUM/len(img_list)))