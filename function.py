"""
Contains Utility functions
"""


import os
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
def save_model(net, checkpoint_dir,epoch,cuda_id,best=False):
        if best:
            save_filename = '%03d_net_best.pth' % (epoch)
        else:
            save_filename = '%03d_net.pth' % (epoch)
        save_filename_latest='latest.pth'
        save_path=os.path.join(checkpoint_dir,'model',save_filename)
        save_path_latest = os.path.join(checkpoint_dir, 'model', save_filename_latest)
        torch.save(net.cpu().state_dict(), save_path)
        torch.save(net.cpu().state_dict(), save_path_latest)
        net.cuda(cuda_id)

def load_model(net,checkpoint_dir, epoch,was_parallel=False):
    #load nework from  epoch or 'latest'
    if epoch=='latest':
        save_filename='latest.pth'
    else:
        save_filename = '%03d_net.pth' % (epoch)
    save_path = os.path.join(checkpoint_dir, 'model', save_filename)
    load_param(net, save_path, was_parallel)

def load_param(net,save_path,was_parallel):
    if not was_parallel:
        net.load_state_dict(torch.load(save_path))
    else:
        new_state_dict = OrderedDict()
        state_dict=torch.load(save_path)
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

def plot_loss(loss_list,loss_list_val,checkpoint_dir):
    x=range(len(loss_list))
    y=loss_list
    y2=loss_list_val
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Train loss')
    plt.plot(x, y2, color='red', marker='o',label='Val loss')
    plt.xticks(range(0,len(loss_list)+3,(len(loss_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'loss_fig.pdf'))

def plot_loss_phase_4(loss_list_1,loss_list_2,loss_list_val_1,loss_list_val_2,checkpoint_dir):
    x=range(len(loss_list_1))
    y1=loss_list_1
    y2=loss_list_val_1
    y3 = loss_list_2
    y4 = loss_list_val_2

    plt.switch_backend('agg')
    plt.plot(x,y1,color='blue',marker='o',label='Train loss1')
    plt.plot(x, y2, color='red', marker='o',label='Val loss1')
    plt.plot(x, y3, color='indigo', marker='o', label='Train loss2')
    plt.plot(x, y4, color='orangered', marker='o', label='Val loss2')
    plt.xticks(range(0,len(loss_list_1)+3,(len(loss_list_1)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'loss_fig.pdf'))

def plot_acc(acc_list,checkpoint_dir):#only plt val
    x=range(len(acc_list))
    y=acc_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Val Acc')
    plt.xticks(range(0,len(acc_list)+3,(len(acc_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'acc_fig.pdf'))

def plot_acc_both(acc_list,acc_list_val,checkpoint_dir):#plot acc of train and val
    x=range(len(acc_list))
    y=acc_list
    y2=acc_list_val
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Train acc')
    plt.plot(x, y2, color='red', marker='o',label='Val acc')
    plt.xticks(range(0,len(acc_list)+3,(len(acc_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'acc_fig.pdf'))

def update_learning_rate(optimizer,decay=0.05):
    for param in optimizer.param_groups:
        lr=param['lr']
        lr=lr*(1-decay)
        param['lr']=lr

def optim_or_not(model,yes):
    for param in model.parameters():
        if yes:
            param.requires_grad=True
        else:
            param.requires_grad = False

def evaluate(pred,label,method='macro'):#F1
    data_type=label.data.type()
    if method=='macro':
        pred=(pred>0.5)
        pred.data=pred.data.type(data_type)
        label = (label > 0.5)
        label.data = label.data.type(data_type)
        a=2*torch.sum(pred*label,0)
        b=torch.sum(pred,0)+torch.sum(label,0)
        return torch.mean(a/b).data[0]
    elif method=='micro':
        pred = (pred > 0.5)
        pred.data = pred.data.type(data_type)
        label = (label > 0.5)
        label.data = label.data.type(data_type)
        a=torch.sum(pred*label)*2
        b=torch.sum(pred)+torch.sum(label)
        return (a/b).data[0]
    else:
        return None


def make_weights_for_balanced_classes(label_list, nclasses=20):
    count = [0] * nclasses #count num of images for each class
    for label in label_list:
        for i in range(nclasses):
            count[i] = count[i] + label[i]
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i]) #weight of the class is the inverse of the number and multiply by all num of imgs
    weight = [0] * len(label_list)
    for idx, val in enumerate(label_list): #assign weight to each image by weight
        present_class = val.index(1)
        weight[idx] = weight_per_class[present_class]


    return weight
