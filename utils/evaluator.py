import os
import time
import torch
import utils.utils as utils

from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms

class Evaluator():

	def __init__(self, model, dataset, n_classes, batch_size=32, use_gpu=True):
		self.model = model
		
		self.batch_size = batch_size

		self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
		self.length = len(dataset)

		self.n_classes = n_classes
		self.device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
		
		self.model.to(self.device)
		
		self.to_image = torchvision.transforms.ToPILImage()
		
	def load_from_disk(self, path):
		self.model.to('cpu')
		self.model.load_state_dict(torch.load(path))
		self.model.to(self.device)
		
	@staticmethod
	def get_iou(real_box, predicted_box):
		xmin, ymin, xmax, ymax = real_box
		xmin_pred, ymin_pred, xmax_pred, ymax_pred = predicted_box
		
		#Intersection
		delta_x = max(min(xmax, xmax_pred) - max(xmin, xmin_pred), 0)
		delta_y = max(min(ymax, ymax_pred) - max(ymin, ymin_pred), 0)
		
		intersection = delta_x*delta_y
		
		#Union
		union = (xmax-xmin)*(ymax-ymin) + (xmax_pred-xmin_pred)*(ymax_pred-ymin_pred) - intersection + 1e-8
		
		return intersection/union
		
	def plot(self):
		self.model.eval()
		
		cnt = 0
		n_imgs = 10

		imgs = []

		with torch.no_grad():
			for images, targets in self.dataloader:
			
				targets = [{k: v[0] for k, v in t.items() if k != 'img_path'} for t in targets]
				
				boxes = torch.stack([t['boxes'] for t in targets])
				labels = torch.stack([t['labels'] for t in targets])
				
				images = list(image.to(self.device) for image in images)
				
				predictions = self.model(images)
				print(predictions)
				
				predictions = [{k: v for k, v in t.items()} for t in predictions]
				
				pred_boxes = torch.stack([t['boxes'] for t in predictions])
				pred_labels = torch.stack([t['labels'] for t in predictions])
				
				for idx, image in enumerate(images):
				
					image = np.array(self.to_image(image))
					
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
				
					imgs.append((image, label, pred_label))
					
				cnt += self.batch_size
				if cnt > n_imgs:
					break
			
		fig = plt.figure(figsize=(16, 8))
		columns = 5
		rows = 2
		for i in range(1, columns*rows +1):
			img = imgs[i][0]
			fig.add_subplot(rows, columns, i)
			plt.imshow(img)
			plt.title(f'Clase Real: {imgs[i][1]}\nClase Predicha: {imgs[i][2]}')
		plt.show()
		
		self.model.to(self.device)
		
	def evaluate(self):
		self.model.eval()
		
		print('Evaluating model...')
		
		iou, label = torch.zeros(0), torch.zeros(0)
		
		batch_cnt = 0
		
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
				
				pred_boxes = torch.stack([t['boxes'][0] for t in predictions])
				pred_labels = torch.stack([t['labels'] for t in predictions])
				
				#Evaluation
				label = torch.cat([label, labels == pred_labels])
				#iou = torch.cat(get_iou(boxes, pred_boxes))
				
				tt = time.time()-start_time
				eta = eta*(1-1//(batch_cnt+1)) + tt*(self.length//self.batch_size-batch_cnt)*(1//(batch_cnt+1))
				
				print(f'BATCH {batch_cnt} FINISHED IN {int(tt//60)}:{int(tt%60)}s\t ETA: {int(eta//60)}:{int(eta%60)}s')
				
				batch_cnt += 1
				
				
					
		accuracy = sum(label)/len(label)
		iou = sum(iou)/len(iou)
		
		print('Finished Evaluating!')
		print(f'Classification Accuracy: {accuracy}, Box Mean IoU: {iou}')

		return metric_logger, accuracy, iou

