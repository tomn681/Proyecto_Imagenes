import os
import time
import torch
import numpy as np
import utils.utils as utils

class Trainer():

	def __init__(self, model, dataset, n_classes, train_size=0.7, train_batch_size=32, optimizer=None, lr_scheduler=None, use_gpu=True):
		self.model = model
		
		self.train_batch_size = train_batch_size
		
		params = [p for p in self.model.parameters() if p.requires_grad]
		
		self.optimizer = torch.optim.Adam(params, lr=0.001)
		if optimizer is not None:
			self.optimizer = optimizer
			
		self.lr_scheduler = None
		if lr_scheduler is not None:
			self.lr_scheduler = lr_scheduler

		train_size = int(train_size*len(dataset))
		dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

		#Dataloaders
		self.train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, collate_fn=utils.collate_fn)
		self.test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

		self.n_classes = n_classes
		self.device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
		
		self.model.to(self.device)
		
	def load_from_disk(self, path):
		self.model.to('cpu')
		self.model = torch.load(path)
		self.model.to(self.device)
		return self.model
		
	def train_one_epoch(self, epoch, print_freq):
		self.model.train()
		
		loss_classifier, loss_box = [], []

		metric_logger = utils.MetricLogger(delimiter="  ")
		metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
		header = f"Epoch: [{epoch}]"

		for images, targets in metric_logger.log_every(self.train_dataloader, print_freq, header):

			# Forward
			images = list(image.to(self.device) for image in images)

			targets = [{k: v.to(self.device) for k, v in t.items() if k != 'img_path'} for t in targets]

			loss_dict = self.model(images, targets)
			losses = sum(loss for loss in loss_dict.values())

			loss_classifier.append(loss_dict['classification'].item())
			loss_box.append(loss_dict['bbox_regression'].item())

			#Backward
			self.optimizer.zero_grad()

			losses.backward()
			self.optimizer.step()

			metric_logger.update(loss=losses, **loss_dict)
			metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

		return metric_logger, np.mean(loss_classifier), np.mean(loss_box)
		
	def validate(self, print_freq=100):
		self.model.train()
		
		loss_classifier, loss_box = [], []
		
		metric_logger = utils.MetricLogger(delimiter="  ")
		header = 'Test:'

		with torch.no_grad():
			for images, targets in metric_logger.log_every(self.test_dataloader, print_freq, header):

				images = list(image.to(self.device) for image in images)
				targets = [{k: v.to(self.device) for k, v in t.items() if k != 'img_path'} for t in targets]

				loss_dict = self.model(images, targets)
				losses = sum(loss for loss in loss_dict.values())

				loss_classifier.append(loss_dict['classification'].item())
				loss_box.append(loss_dict['bbox_regression'].item())

				metric_logger.update(loss=losses, **loss_dict)

		return metric_logger, np.mean(loss_classifier), np.mean(loss_box)
		
	def train(self, epochs=100, print_every=100, checkpoints_path='./checkpoints', checkpoint_prefix='checkpoint_'):
		start_time = time.time()
		
		#Prepare Checkpoint directory
		if not os.path.exists(checkpoints_path):
			os.makedirs(checkpoints_path)
			
		#Save Base State
		path = os.path.join(checkpoints_path, checkpoint_prefix + 'init.pt')
		torch.save(self.model, path)
			
		train_loss_classifier, train_loss_box, test_loss_classifier, test_loss_box = [], [], [], []
			
		#Training
		for epoch in range(epochs):
		
			# train for one epoch
			_, loss_classifier, loss_box = self.train_one_epoch(epoch, print_freq=print_every)
			train_loss_classifier.append(loss_classifier)
			train_loss_box.append(loss_box)
			
			#Save state dictionary
			path = os.path.join(checkpoints_path, checkpoint_prefix + str(epoch) + '.pt')
			torch.save(self.model, path)
			
			#Update the learning rate
			if self.lr_scheduler is not None:
				self.lr_scheduler.step()
				
			#Validate
			_, loss_classifier, loss_box = self.validate()
			test_loss_classifier.append(loss_classifier)
			test_loss_box.append(loss_box)
			
		#Compress Data
		train_loss = {'loss_classifier': train_loss_classifier, 'loss_box': train_loss_box}
		test_loss = {'loss_classifier': test_loss_classifier, 'loss_box': test_loss_box}
			
		tt = time.time() - start_time
		print(f'\nFinished Training in {int(tt//3600)}:{int((tt//60))%60}:{int(tt%60%60)}')
		
		return train_loss, test_loss
