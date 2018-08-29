import os
from PIL import Image

import torch

from predict import predict_img
from utils import rle_encode
from unet import UNet
import ipdb

def submit(net, gpu=False):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = '/home/pengbo/project/datasets/TGS_Salt/images/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSION.csv', 'a') as f:
        f.write('id,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, gpu)
            #ipdb.set_trace()
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i.split('.')[0], ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('./checkpoints/CP500.pth'))
    submit(net, True)
