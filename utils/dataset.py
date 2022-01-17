'''
@author: Tomás de la Sotta (github @tomn681)
'''

import os
import numpy as np
import pandas as pd

import torch
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms

'''
Class Clothing Dataset:

Constructs a Deep Fashion Object-Detection dataset.
'''
class ClothingDataset(torch.utils.data.Dataset):

	'''
	Inputs:
		- basepath: (String) Annotations file directory path. 
	    
		        Directory must contain files:
		        
		          File Name:                File Column Data:
		            - 'test.txt':               (String) Path
		            - 'test_bbox.txt':          (Int, Int, Int, Int) X1, Y1, X2, Y2
		            - 'test_cate.txt':          (Int) Class
		            - 'train.txt':              (String) Path
		            - 'train_bbox.txt':         (Int, Int, Int, Int) X1, Y1, X2, Y2
		            - 'train_cate.txt':         (Int) Class
		            - 'val.txt':                (String) Path
		            - 'val_bbox.txt':           (Int, Int, Int, Int) X1, Y1, X2, Y2 
		            - 'val_cate.txt':           (Int) Class
		            
		- transforms: (object containing torchvision.tranforms) Data Augmentation Transforms.
		
		- train: (String) If 'train', opens train and validation files.
			If 'test' opens test files and transforms = None.
        '''
        
	def __init__(self, basepath, transforms=None, train='train'):
		self.basepath = basepath
		self.transforms = transforms if train != 'test' else None
		
		imgs = pd.read_csv(os.path.join(basepath, 'test.txt'), sep=' ', names=('path',), index_col=False)
		imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'test_bbox.txt'), sep=' ', names=('x1', 'x2', 'y1', 'y2'), index_col=False), lsuffix='_left', rsuffix='_right')
		imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'test_cate.txt'), sep=' ', names=('class',), index_col=False), lsuffix='_left', rsuffix='_right')
		
		#Defaults to train
		if train != 'test':
			imgs = pd.read_csv(os.path.join(basepath, 'train.txt'), sep=' ', names=('path',), index_col=False)
			imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'train_bbox.txt'), sep=' ', names=('x1', 'x2', 'y1', 'y2'), index_col=False), lsuffix='_left', rsuffix='_right')
			imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'train_cate.txt'), sep=' ', names=('class',), index_col=False), lsuffix='_left', rsuffix='_right')
			
			val_imgs = pd.read_csv(os.path.join(basepath, 'val.txt'), sep=' ', names=('path',), index_col=False)
			val_imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'val_bbox.txt'), sep=' ', names=('x1', 'x2', 'y1', 'y2'), index_col=False), lsuffix='_left', rsuffix='_right')
			val_imgs = imgs.join(pd.read_csv(os.path.join(basepath, 'val_cate.txt'), sep=' ', names=('class',), index_col=False), lsuffix='_left', rsuffix='_right')
			
			#imgs = pd.concat([imgs, val_imgs])
			#imgs.reset_index(drop=True, inplace=True)
		
		total_images = len(imgs)
		
		imgs.drop(imgs[(imgs['x1']-imgs['x2'])==0].index, inplace=True)
		imgs.drop(imgs[(imgs['y1']-imgs['y2'])==0].index, inplace=True)
		imgs.reset_index(drop=True, inplace=True)
		print(f'Dropped {total_images-len(imgs)} images with box area 0')
		
		self.imgs = imgs
		
		self.to_tensor = torchvision.transforms.ToTensor()
		self.resize = torchvision.transforms.Resize((224, 224))
		self.to_image = torchvision.transforms.ToPILImage()
		
	'''
	getitem:
	
	Default __getitem__ method. Allows to iterate on the dataset.
	
	Inputs: 
		- idx: (Int) Item number to retrieve
		
	Outputs:
		- img: (torch.Tensor) Standard ResNet Resized Image (Size 224x224)
		
		- target: (dict(list))
			Keys:
				- boxes: (torch.Tensor) images box corners (x1, y1, x2, y2)
				- labels: (torch.Tensor) images labels
				- image_id: (torch.Tensor) images ids
				- area: (torch.Tensor) images box areas
				- img_path: (String) images relative paths
	'''
	def __getitem__(self, idx):
		img_path = os.path.join('../data', self.imgs['path'][idx])
		
		img = Image.open(img_path).convert('RGB')

		xmin = min(self.imgs['x1'][idx], self.imgs['x2'][idx])
		xmax = max(self.imgs['x1'][idx], self.imgs['x2'][idx])
		ymin = min(self.imgs['y1'][idx], self.imgs['y2'][idx])
		ymax = max(self.imgs['y1'][idx], self.imgs['y2'][idx])
		box = [xmin, ymin, xmax, ymax]

		boxes = torch.as_tensor([box,], dtype=torch.float32)
		
		labels = torch.as_tensor([self.imgs['class'][idx],])

		image_id = torch.tensor([idx,])
		area = torch.tensor([(box[3] - box[1]) * (box[2] - box[0]),])

		target = {}
		target['boxes'] = boxes
		target['labels'] = labels
		target['image_id'] = image_id
		target['area'] = area
		target['img_path'] = img_path
		
		if self.transforms is not None:
		    img, target = self.transforms(img, target)
		    
		img = self.to_tensor(img)
		
		img = self.resize(img)

		return img, target
		
	'''
	len:
	
	Default len method. Allos to get the amount of images in dataset.
	
	Inputs: 
		- None
		
	Outputs:
		- len: (Int) Number of images in dataset
	'''
	def __len__(self):
		return len(self.imgs)
		
	'''
	plot:
	
	Image showing application. Shows images with its corresponding
	bounding boxes and classes.
	
	Inputs:
		- None
		
	Outputs: 
		- Matplotlib image (Standard Output, no return type)
	'''
	def plot(self):
		cnt = 0
		n_imgs = 10

		imgs = []

		for image, target in self:
			label = target['labels']
			box = target['boxes'][0]
			image = np.array(Image.open(target['img_path']).convert('RGB'))
			startpoint, endpoint = (int(box[0]), int(box[2])), (int(box[1]), int(box[3]))
			color = (0, 255, 0)
			thickness = 2
			image = cv2.rectangle(image, startpoint, endpoint, color, thickness)
			imgs.append((image, label))
			cnt += 1
			if cnt > n_imgs:
				break
		fig = plt.figure(figsize=(16, 8))
		columns = 5
		rows = 2
		for i in range(1, columns*rows +1):
			img = imgs[i][0]
			fig.add_subplot(rows, columns, i)
			plt.imshow(img)
			plt.title(f'Clase: {imgs[i][1]}')
		plt.show()
