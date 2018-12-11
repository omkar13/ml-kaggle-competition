"""
Predicts the top 5 classes on test data.
"""
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
import os
import csv
from torch.autograd import Variable

import numpy as np

from function import load_model

from dataset import testData
num_classes = 18
use_gpu = True
gpu_id = 1

#config of model to load
lr=0.005
was_parallel = True

checkpoint_dir='kaggle/resnet101_entiredata2augs/lr=%.4f'%lr
data_dir='/home/omkar/Documents/Omkar/kaggle'


"""
#data augmentation and normalization for training
#normalization for validation
data_transforms = transforms.Compose(
		[transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
		])
"""

#if using zhang's code
data_transforms = transforms.Compose(
		[transforms.Resize([224, 224]),
	    transforms.ToTensor(),
    	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		])

dataloader = torch.utils.data.DataLoader(testData(transforms = data_transforms), batch_size = 10, shuffle= False, num_workers = 4)
net = torchvision.models.resnet101(pretrained=True)
net.fc=nn.Linear(2048,18)
load_model(net,checkpoint_dir,'latest',was_parallel=was_parallel)

if use_gpu:
	net = net.cuda(gpu_id)

net.eval()
id_list = []
id_no = 1
#pred_list = []

test_image_id = 1

top5_dir = os.path.join(os.getcwd(), 'resnet101_top5')
if os.path.exists(top5_dir) == False:
	os.mkdir(top5_dir)

for inputs in dataloader:
	if use_gpu:
		inputs = Variable(inputs.cuda(gpu_id))
	else:
		inputs = Variable(inputs)

	outputs = net(inputs)
	_,preds = torch.max(outputs.data,1)
	batch_size = len(outputs)

	for i in range(batch_size):
		if test_image_id%100==0:
			print('Working for image id: ' + str(test_image_id))

		confidence_list = outputs[i]
		softmax_func = nn.Softmax()
		confidence_list = softmax_func(confidence_list)
		sort_indices = confidence_list.data.cpu().numpy().argsort()
		top_5_indices = sort_indices[-5:][::-1]
		top_5_confidences = [confidence_list[x] for x in top_5_indices]


		file_name = str(test_image_id) + '.csv'

		with open(os.path.join(top5_dir,file_name),'w+') as myFile:
			wr = csv.writer(myFile, delimiter=',')
			wr.writerow(['label', 'confidence'])
			for j in range(len(top_5_indices)):
				wr.writerow([top_5_indices[j] , top_5_confidences[j].data.cpu().numpy()[0]])

		test_image_id+=1
