"""
Predicts labels for test data using trained model
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

from function import load_model

from dataset import testData
num_classes = 18
use_gpu = True
gpu_id = 1

#config of model to load
lr=0.005
was_parallel = True

checkpoint_dir='kaggle/checkpoint/lr=%.4f'%lr
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

dataloader = torch.utils.data.DataLoader(testData(transforms = data_transforms), batch_size = 4, shuffle= False, num_workers = 4)
net = torchvision.models.resnet152(pretrained=True)
net.fc=nn.Linear(2048,18)
load_model(net,checkpoint_dir,'latest',was_parallel=was_parallel)

if use_gpu:
	net = net.cuda(gpu_id)

net.eval()
id_list = []
id_no = 1
pred_list = []

for inputs in dataloader:
	if use_gpu:
		inputs = Variable(inputs.cuda(gpu_id))
	else:
		inputs = Variable(inputs)

	outputs = net(inputs)
	_,preds = torch.max(outputs.data,1)
	pred_list.extend(preds.cpu().numpy())

print(len(pred_list))

file_name = 'results1.csv'
with open(file_name,'w+') as myFile:
	wr = csv.writer(myFile, delimiter=' ')
	for i in range(len(pred_list)):
		wr.writerow([str(i+1) , str(pred_list[i])])
