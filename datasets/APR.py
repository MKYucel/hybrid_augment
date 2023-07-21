import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
import torchvision.transforms as T


class HybridAugmentPlusSingle(object):
    def __init__(self, img_size=32, aug=None, ks = 3 , sigma = 0.5, prob=0.5, blur_strength = 1):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

        self.blur_strength = blur_strength
        self.sigma = sigma
        self.ks = ks
        self.prob = prob
        self.APR = APRecombination(img_size = img_size)
        print("HybridAugment++ Single, ks:", ks, " sigma:", sigma, "prob:", self.prob, "blur_str:", self.blur_strength)

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > self.prob:
            return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        blurrer = T.GaussianBlur(kernel_size=self.ks * self.blur_strength, sigma=self.sigma * self.blur_strength)
        reconvert = T.ToPILImage()

        trans1 = T.ToTensor()

        x = trans1(x)
        x_aug = trans1(x_aug)

        lfc_f = blurrer(x)
        hfc_f = x - lfc_f
        lfc_f = trans1(self.APR(reconvert(lfc_f)))

        lfc_s = blurrer(x_aug)
        hfc_s = x_aug - lfc_s
        lfc_s = trans1(self.APR(reconvert(lfc_s)))

        p = random.uniform(0, 1)

        if p > self.prob:
            hybrid_im =  lfc_f + hfc_s
        else:
            hybrid_im =  lfc_s + hfc_f


        hybrid_im = reconvert(hybrid_im)

        return hybrid_im

class HybridAugmentSingle(object):
    def __init__(self, img_size=32, aug=None, ks = 3 , sigma = 0.5,  prob=0.5):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

        self.sigma = sigma
        self.ks = ks
        self.prob = prob
        print("HybridAugment Single, ks:", ks, " sigma:", sigma, "prob:", self.prob)

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > self.prob:
            return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        blurrer = T.GaussianBlur(kernel_size=self.ks, sigma=self.sigma)
        reconvert = T.ToPILImage()

        trans1 = T.ToTensor()

        x = trans1(x)
        x_aug = trans1(x_aug)
        lfc_f = blurrer(x)
        hfc_f = x - lfc_f

        lfc_s = blurrer(x_aug)
        hfc_s = x_aug - lfc_s

        p = random.uniform(0, 1)

        if p > self.prob:
            hybrid_im =  lfc_f + hfc_s
        else:
            hybrid_im =  lfc_s + hfc_f

        hybrid_im = reconvert(hybrid_im)

        return hybrid_im

class APRecombination(object):
    def __init__(self, img_size=32, aug=None):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''

        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > 0.5:
            return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        x = np.array(x).astype(np.uint8) 
        x_aug = np.array(x_aug).astype(np.uint8)
        
        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        fft_2 = np.fft.fftshift(np.fft.fftn(x_aug))
        
        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
        abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

        fft_1 = abs_1*np.exp((1j) * angle_2)
        fft_2 = abs_2*np.exp((1j) * angle_1)

        p = random.uniform(0, 1)

        if p > 0.5:
            x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        else:
            x = np.fft.ifftn(np.fft.ifftshift(fft_2))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        
        return x
