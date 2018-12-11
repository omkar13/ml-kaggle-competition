import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from function import load_model,save_model,plot_loss,update_learning_rate,plot_acc_both
import numpy as np
import os
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
#-------------------------------------------------------
#lr = 0.0030843
lr = 0.005
wt = 0.0005
batch_size = 100
cuda_id = 0
cuda_id_list=[0,1]
continue_train=True
checkpoint_dir='kaggle/resnet101_with2augmentations/lr=%.4f'%lr
#'kaggle/checkpoint/lr=%.4f'%lr
decay=False
parallel=True
was_parallel=True
num_class=18
#-----------------------------------------------
if not os.path.exists(os.path.join(checkpoint_dir,'model')):
    os.makedirs(os.path.join(checkpoint_dir,'model'))

train_set=ImageFolder('kaggle/train',transform=transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomCrop([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]))
val_set=ImageFolder('kaggle/val',transform=transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]))


#make weight for each class
num_each_class=[0]*20
for i in range (len(train_set)):
    num_each_class[train_set.imgs[i][1]]+=1
weight=[0.]*len(train_set)
for i in range(len(train_set)):
    weight[i]=len(train_set)/num_each_class[train_set.imgs[i][1]]
weight=torch.DoubleTensor(weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,sampler=sampler,shuffle=False, num_workers=4, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)


resnet101 = torchvision.models.resnet101(pretrained=True)
resnet101.fc=nn.Linear(2048,18)



if continue_train==True:
    print('Loading network')
    load_model(resnet101,checkpoint_dir,5,was_parallel=was_parallel)

resnet101.cuda(cuda_id)



Loss_function = nn.CrossEntropyLoss()
if parallel:
    resnet101=torch.nn.DataParallel(resnet101, device_ids=cuda_id_list)


total_val_img = np.zeros(18)
total_correct_val_img = np.zeros(18)

#--------------------------training-----------------------

print ('evaluating now')
resnet101.eval()

#val
for data in val_loader:
    batch_data, batch_label = data
    batch_data = Variable(batch_data,volatile = True).cuda(cuda_id)
    batch_label = Variable(batch_label,volatile = True).cuda(cuda_id)
    pred = resnet101(batch_data)
    loss = Loss_function(pred, batch_label)

    _, predict_index = torch.max(pred, 1)
    
    correct_number = torch.sum(predict_index == batch_label).float()

    for class_id in range(len(total_val_img)):
        class_id_array = [class_id]*len(batch_label)
        
        for ii in range(len(class_id_array)):
            if class_id_array[ii] == batch_label.data.cpu().numpy()[ii] and predict_index.data.cpu().numpy()[ii] == batch_label.data.cpu().numpy()[ii]:
                total_correct_val_img[class_id] += 1
            if class_id_array[ii] == batch_label.data.cpu().numpy()[ii]:
                total_val_img[class_id] += 1

for class_id in range(len(total_val_img)):
    print('For class: ' + str(class_id) + ', accuracy = ' + str(float(total_correct_val_img[class_id])/total_val_img[class_id]))

accuracy_array = [float(total_correct_val_img[class_id])/total_val_img[class_id] for class_id in range(len(total_val_img))]

accuracy_file = 'accuracy_res101'
np.save(accuracy_file,accuracy_array)
