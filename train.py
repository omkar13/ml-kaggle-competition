"""
Training script for a resnet model
"""

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
continue_train=False
checkpoint_dir='kaggle/checkpoint/lr=%.4f'%lr
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


resnet101 = torchvision.models.resnet152(pretrained=True)
resnet101.fc=nn.Linear(2048,18)



if continue_train==True:
    print('Loading network')
    load_model(resnet101,checkpoint_dir,'latest',was_parallel=was_parallel)
    loss_list=list(np.loadtxt(os.path.join(checkpoint_dir,'loss_history_train.txt')))
    acc_list = list(np.loadtxt(os.path.join(checkpoint_dir, 'acc_history_train.txt')))
    loss_list_val = list(np.loadtxt(os.path.join(checkpoint_dir,'loss_history_val.txt')))
    acc_list_val = list(np.loadtxt(os.path.join(checkpoint_dir, 'acc_history_val.txt')))
    lowest_loss=min(loss_list_val)
    start_epoch=len(loss_list)
else:
    start_epoch=0
    loss_list = []
    loss_list_val = []
    acc_list=[]
    acc_list_val=[]
    lowest_loss=9999

resnet101.cuda(cuda_id)

optimizer = optim.SGD(resnet101.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
Loss_function = nn.CrossEntropyLoss()
if parallel:
    resnet101=torch.nn.DataParallel(resnet101, device_ids=cuda_id_list)
#--------------------------training-----------------------

print ('Let us go training')

for epoch in range(start_epoch,50):

    print('Now epoch %03d' % epoch)
    running_loss=0 #average loss in an epoch
    running_acc=0
    running_loss_val = 0
    iter_num = 0
    iter_num_val=0
    begin_time=time.time()

    for data in train_loader:
        batch_data,batch_label=data
        batch_data=Variable(batch_data).cuda(cuda_id)
        batch_label=Variable(batch_label).cuda(cuda_id)

        pred = resnet101(batch_data)
        loss = Loss_function(pred, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,predict_index=torch.max(pred,1)
        correct_number=torch.sum(predict_index==batch_label).float()
        #print (correct_number)
        acc=correct_number/batch_size
        print('epoch:%03d iter:%04d loss:%.3f  acc:%.2f left:%04d' % (epoch, iter_num, loss, acc.data[0], len(train_set)//batch_size-iter_num))
        iter_num = iter_num + 1
        running_acc=running_acc+acc
        running_loss = running_loss + loss.data[0]

    loss_list.append(running_loss/iter_num)
    acc_list.append(running_acc.data[0]/iter_num)
    print ('evaluating now')
    resnet101.eval()

#val
    running_acc_val=0
    for data in val_loader:
        iter_num_val=iter_num_val+1
        batch_data, batch_label = data
        batch_data = Variable(batch_data,volatile = True).cuda(cuda_id)
        batch_label = Variable(batch_label,volatile = True).cuda(cuda_id)
        pred = resnet101(batch_data)
        loss = Loss_function(pred, batch_label)

        _, predict_index = torch.max(pred, 1)
        correct_number = torch.sum(predict_index == batch_label).float()
        acc = correct_number / batch_size
        running_acc_val+=acc

        running_loss_val = running_loss_val + loss.data[0]

    acc_list_val.append(running_acc_val.data[0]/iter_num_val)
    loss_list_val.append(running_loss_val/iter_num_val)
    #save model
    if running_loss_val / iter_num_val < lowest_loss:
        lowest_loss = running_loss_val / iter_num_val
        save_model(resnet101, checkpoint_dir, epoch, cuda_id, True)
    else:
        save_model(resnet101, checkpoint_dir, epoch, cuda_id, False)


    resnet101.train()

    np.savetxt(os.path.join(checkpoint_dir, 'loss_history_train.txt'), np.array(loss_list))
    np.savetxt(os.path.join(checkpoint_dir, 'loss_history_val.txt'), np.array(loss_list_val))
    np.savetxt(os.path.join(checkpoint_dir, 'acc_history_val.txt'), np.array(acc_list_val))
    np.savetxt(os.path.join(checkpoint_dir, 'acc_history_train.txt'), np.array(acc_list))
    plot_loss(loss_list, loss_list_val, checkpoint_dir)

    plot_acc_both(acc_list,acc_list_val, checkpoint_dir)
    if decay:
        update_learning_rate(optimizer,0.05)
    print('This epoch costs time of :%02d s' % (time.time() - begin_time))

