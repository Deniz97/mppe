import numpy as np
import torch
from torch.utils.data import Dataset

from server.py_rmpe_data_generator import RawDataGenerator


class CocoPoseDataset(Dataset):
    """MsCOCO pose dataset"""
    
    def __init__(self,h5_file, config_object, shuffle=True, augment=True, num_out=4, limit=None, segmentation=True, transform=None, visualize=False):
        self.raw_data_generator = RawDataGenerator(h5_file,config_object,augment=augment)
        self.raw_generator = self.raw_data_generator.gen
        self.segmentation = segmentation
        self.keypoints = None #this probably will not work as expected as it is
        self.num_out = num_out
        self.heat_num = 18
        self.transform = transform
        self.visualize = visualize
    def __len__(self):
        return self.raw_data_generator.total_data_count
    def preprocess_input(self, image):
        '''
        :param image: image in RGB format
        :return: image normalized by imagenet means
        '''
        image = image.astype(np.float32)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        return image
    def __getitem__(self,idx):
        
        foo = self.raw_generator(idx)
        
        if len(foo)==5:
            data_img, mask_img, mask_all, label, kpts = foo
        else:
            data_img, mask_img, mask_all, label = foo
            kpts = None
                
        
        #data_img = np.transpose(data_img, (1, 2, 0))[:,:,::-1]
        if not self.visualize:
            data_img = self.preprocess_input(data_img)
        #label = np.transpose(label, (1, 2, 0))
        mask_miss = np.repeat(mask_img[np.newaxis,:, :], self.heat_num, axis=0)
        print(mask_miss.shape)
        print(label.shape)
        print(mask_all.shape)
        if self.segmentation:
            label = np.vstack((label, np.expand_dims(mask_all, axis=0)))
            mask_miss = np.vstack((mask_miss, np.ones((1,mask_miss.shape[1], mask_miss.shape[2]))))
        
        self.keypoints = kpts #TODO
        
        data_img = torch.from_numpy(data_img)
        label = torch.from_numpy(label)
        mask_miss = torch.from_numpy(mask_miss)
        print(label.shape)
        if self.transform:
            pass
            #sample = self.transform(sample)
        
        return [data_img]+[mask_miss],[label]*self.num_out