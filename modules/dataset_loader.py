from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
import cv2
import pdb


def ResizeCrop(image, sz, div_factor):
    
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                                   image_size[0] // div_factor])(image)
    
    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)
    
    return image

def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image

def MS_transform(image):
#   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7/6)
    image[:,:,0] = compute_MS_transform(image[:,:,0], window)
    image[:,:,0] = (image[:,:,0] - np.min(image[:,:,0]))/(np.ptp(image[:,:,0])+1e-3)
    image[:,:,1] = compute_MS_transform(image[:,:,1], window)
    image[:,:,1] = (image[:,:,1] - np.min(image[:,:,1]))/(np.ptp(image[:,:,1])+1e-3)
    image[:,:,2] = compute_MS_transform(image[:,:,2], window)
    image[:,:,2] = (image[:,:,2] - np.min(image[:,:,2]))/(np.ptp(image[:,:,2])+1e-3)
    
    image = Image.fromarray((image*255).astype(np.uint8))
    return image

def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
         im = im.convert('HSV')
    elif val == 3:
         im = MS_transform(im)
    return im

class image_data(Dataset):
    def __init__(self, file_path, image_size = (256,256), transform = True):
        self.fls = pd.read_csv(file_path)
        self.image_size = image_size
        
        self.tranform_toT = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                ])
    
    def __len__(self):
        return len(self.fls)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.fls.iloc[idx]['File_names'].rstrip()
        image_orig = Image.open(img_name)
        
        if image_orig.mode == 'L':
            image_orig = np.array(image_orig)
            image_orig = np.repeat(image_orig[:,:,None],3,axis=2)
            image_orig = Image.fromarray(image_orig)
        elif image_orig.mode != 'RGB':
            image_orig = image_orig.convert('RGB')
            
        
        # Data augmentations
        
        # scaling transform and random crop
        div_factor = np.random.choice([1,2],1)[0]
        image_2 = ResizeCrop(image_orig, self.image_size, div_factor)
        
        # change colorspace
        colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
        image_2 = colorspaces(image_2, colorspace_choice)
        image_2 = self.tranform_toT(image_2)
        
        # scaling transform and random crop
        image = ResizeCrop(image_orig, self.image_size, 3 - div_factor)
        
        # change colorspace
        colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
        image = colorspaces(image, colorspace_choice)
        image = self.tranform_toT(image)
        
        # read distortion class, for authentically distorted images it will be 0
        label = self.fls.iloc[idx]['labels']
        label = label[1:-1].split(' ')
        label = np.array([t.replace(',','') for t in label]).astype(np.float32)

        depth_map_path = os.path.join('/scratch/09519/vedasam_ch/depth_data', img_name)
        depth_map_path = os.path.splitext(depth_map_path)[0] + '.npy'
        depth_map = np.load(depth_map_path)
        depth_map = np.clip(depth_map, 0, None)
        # pdb.set_trace()
        depth_map = cv2.resize(depth_map, (image.size()[1], image.size()[2]), interpolation=cv2.INTER_CUBIC)

        depth_map = torch.from_numpy(depth_map).float() / 14000.0
        image = torch.cat([image, depth_map.unsqueeze(0)], dim=0)

        depth_map_np = depth_map.cpu().numpy()
        depth_map_2 = cv2.resize(depth_map_np, (image_2.size()[1], image_2.size()[2]), interpolation=cv2.INTER_CUBIC)
        depth_map_2 = torch.from_numpy(depth_map_2).float() / 14000.0
        image_2 = torch.cat([image_2, depth_map_2.unsqueeze(0)], dim=0)


        return image, image_2, label
