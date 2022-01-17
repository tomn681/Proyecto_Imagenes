'''
@author: Tomás de la Sotta (github @tomn681)
'''

import os
import time
import torch
import utils.utils as utils

from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms

'''
Class Evaluator:

Simple NN evaluator for object detection. It implements the ability 
plot real and predicted class and box values. It also allows to 
calculate the classification accuracy and the mean box IoU. 			
'''
class Evaluator():

	'''
	Evaluator Constructor 

	Inputs:

	    - model -> (torchvision.models.detection) Model to Evaluate
	    
	    - dataset -> (utils.dataset.ClothingDataset) Dataset used to test
	     
	    - n_classes -> (Int) Number of classes on 
		        dataset + 1 (Class 0 represents 
		        backround)
		        
	    - batch_size -> (Int) Batch size used
		        while testing. Default: 32
		        
	    - use_gpu -> (Bool) If False, training on CPU. Default: True
	    
	    '''
	def __init__(self, model, dataset, n_classes, batch_size=32, use_gpu=True):
		self.model = model
		
		self.batch_size = batch_size

		self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
		self.length = len(dataset)

		self.n_classes = n_classes
		self.device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
		
		self.model.to(self.device)
		
		self.to_image = torchvision.transforms.ToPILImage()
		
		
	'''
	get_iou:
	
	Static Method. Calculates the mean IoU for a given batch.
	
	Inputs:
		- real: (list) Target boxes.
		- predicted: (List) Predicted boxes.
		
	Outputs: 
		- Batch mean IoU value.
	'''
	@staticmethod
	def get_iou(real, predicted):
	
		output = []
	
		for real_box, predicted_box in zip(real, predicted):
			xmin, ymin, xmax, ymax = real_box
			xmin_pred, ymin_pred, xmax_pred, ymax_pred = predicted_box
			
			#Intersection
			delta_x = max(min(xmax, xmax_pred) - max(xmin, xmin_pred), 0)
			delta_y = max(min(ymax, ymax_pred) - max(ymin, ymin_pred), 0)
			
			intersection = delta_x*delta_y
			
			#Union
			union = (xmax-xmin)*(ymax-ymin) + (xmax_pred-xmin_pred)*(ymax_pred-ymin_pred) - intersection + 1e-8
			
			iou = abs(intersection/union)
			
			output.append(iou)
		
		return torch.mean(torch.stack(output))
		
	'''
	plot:
	
	Image showing application. Shows images with its corresponding
	real and predicted bounding boxes and classes.
	
	Inputs:
		- None
		
	Outputs: 
		- Matplotlib image (Standard Output, no return type)
	'''
	def plot(self):
		self.model.eval()
		self.model = self.model.to(self.device)
		
		cnt = 0
		n_imgs = 10

		imgs = []

		with torch.no_grad():
			for images, targets in self.dataloader:
			
				paths = [t['img_path'] for t in targets]
				targets = [{k: v[0] for k, v in t.items() if k != 'img_path'} for t in targets]
				
				boxes = torch.stack([t['boxes'] for t in targets])
				labels = torch.stack([t['labels'] for t in targets])
				
				images = list(image.to(self.device) for image in images)
				
				predictions = self.model(images)
				
				predictions = [{k: v for k, v in t.items()} for t in predictions]
				
				pred_boxes = torch.stack([t['boxes'] for t in predictions])
				pred_labels = torch.stack([t['labels'] for t in predictions])
				
				for idx, image in enumerate(images):
				
					image = np.array(Image.open(paths[idx]).convert('RGB'))
					
					box = boxes[idx]
					label = labels[idx]
				
					startpoint, endpoint = (int(box[0]), int(box[2])), (int(box[1]), int(box[3]))
					color = (0, 255, 0)
					thickness = 2
					image = cv2.rectangle(image, startpoint, endpoint, color, thickness)
					
					pred_box = pred_boxes[idx][0]
					pred_label = pred_labels[idx][0]

					startpoint, endpoint = (int(pred_box[0]), int(pred_box[2])), (int(pred_box[1]), int(pred_box[3]))
					color = (255, 0, 0)
					thickness = 2
					image = cv2.rectangle(image, startpoint, endpoint, color, thickness)
				
					imgs.append((image, label, pred_label, box, pred_box))
					
				cnt += self.batch_size
				if cnt > n_imgs:
					break
			
		fig = plt.figure(figsize=(16, 10))
		columns = 5
		rows = 2
		for i in range(1, columns*rows +1):
			img = imgs[i][0]
			fig.add_subplot(rows, columns, i)
			plt.imshow(img)
			plt.title(f'Clase Real: {imgs[i][1]}\nClase Predicha: {imgs[i][2]}\nIoU: {round(self.get_iou([imgs[i][3]], [imgs[i][4]]).item(), 4)}')
		plt.show()
		
		
	'''
	evaluate:
	
	Calculates the accuracy and mean box IoU of the given dataset.
	
	Inputs:
		- None
		
	Outputs: 
		- metric_logger (utils.utils.MetricLogger)
    			
		- Classification accuracy (float)
		
		- Box mean IoU (float)
	'''
	def evaluate(self):
		self.model.eval()
		self.model = self.model.to(self.device)
		
		print('Evaluating model...')
		
		iou, label = [], torch.zeros(0, device=self.device)
		
		batch_cnt, eta, tt = 0, 0, 0
		
		with torch.no_grad():
		
			for images, targets in self.dataloader:
			
				start_time = time.time()
			
				#Expected values
				images = list(image.to(self.device) for image in images)
				targets = [{k: v.to(self.device) for k, v in t.items() if k != 'img_path'} for t in targets]
				
				boxes = torch.stack([t['boxes'][0] for t in targets])
				labels = torch.cat([t['labels'] for t in targets])
				
				#Predictions
				predictions = self.model(images, targets)
				
				predictions = [{k: v[0].to(self.device) for k, v in t.items()} for t in predictions]
				
				pred_boxes = torch.stack([t['boxes'] for t in predictions])
				pred_labels = torch.stack([t['labels'] for t in predictions])
				
				#Evaluation
				label = torch.cat([label, labels == pred_labels])
				iou.append(self.get_iou(boxes, pred_boxes))
				
				batch_cnt += 1
				
				tt = (time.time()-start_time)*(1//batch_cnt) + tt*(1-1//batch_cnt) #Mean total time
				eta = max(tt*(self.length//self.batch_size-batch_cnt), 0) #Estimated time for remaining batches
				
				print(f'BATCH {batch_cnt} FINISHED IN\t' + f'{int(time.time()-start_time)//60}'.zfill(2) 
					+ ':' +f'{int(time.time()-start_time)%60}'.zfill(2) + f's ({int((time.time()-start_time)*1000)}ms)\t ETA: ' + f'{int(eta//60)}'.zfill(2) 
					+ ':' + f'{int(eta%60)}'.zfill(2) + 's')
				
				
				
					
		accuracy = sum(label)/len(label)
		iou = sum(iou)/len(iou)
		
		print('\nFinished Evaluating!')
		print(f'\nClassification Accuracy: {accuracy}, Box Mean IoU: {iou}')

		return accuracy, iou

