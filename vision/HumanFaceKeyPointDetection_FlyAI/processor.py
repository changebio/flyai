# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download

from path import DATA_PATH



class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        pred_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        image = pred_transforms(image)
        #print('image process')
        return image

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        input_y = numpy.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
        return input_y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.around(data,decimals=1)



class DatasetFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        one = self.df.iloc[index]
        image_path = one['image_path']
        label = numpy.array(list(one[1:]))
        #image_path,label,_ = self.df.iloc[index]
        path = check_download(image_path, self.root)
        image = Image.open(path)
        image = image.convert('RGB')

        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

