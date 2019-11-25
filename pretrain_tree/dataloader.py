import torch
import torch.tensor
import torch.nn as nn
import json
import numpy as np
import torch.utils.data as Data
import os

def name_img2id(img_id):
    if len(img_id)==31:
        return (str(int(img_id[15:27])))
    if len(img_id) == 29:
        return(str(int(img_id[13:25])))
    if len(img_id)==30:
        return (str(int(img_id[14:26])))

def make_train_dataset(jsonfile):
    dataset=[]
    jsondata=json.load(open(jsonfile))
    keylist1=jsondata['train'].keys()
    keylist2=jsondata['val4w'].keys()
    for key in keylist1:
        fmname=name_img2id(key)
        for x in range(5):
            treelabel=jsondata['train'][key][x]
            dataset.append([fmname, treelabel])
    for key in keylist2:
        fmname=name_img2id(key)
        for x in range(5):
            treelabel=jsondata['val4w'][key][x]
            dataset.append([fmname, treelabel])
    return dataset

def make_val_dataset(jsonfile):
    dataset = []
    jsondata = json.load(open(jsonfile))
    keylist = jsondata['val'].keys()
    for key in keylist:
        fmname = name_img2id(key)
        for x in range(5):
            treelabel = jsondata['val'][key][x]
            dataset.append([fmname, treelabel])
    return dataset

def make_test_dataset(jsonfile):
    dataset=[]
    jsondata=json.load(open(jsonfile))
    keylist=jsondata['test'].keys()
    for key in keylist:
        fmname=name_img2id(key)
        for x in range(1):
            treelabel=jsondata['test'][key][0]
            dataset.append([fmname, treelabel])
    return dataset

class H5Dataset(Data.Dataset):

    def __init__(self, jsondata,att_pth,entitynum,relationnum,opt):
        if opt==0:###0===train
            self.dataset = make_train_dataset(jsondata)
        if opt==1:###1==val
            self.dataset=make_val_dataset(jsondata)
        if opt==2:###2==test
            self.dataset=make_test_dataset(jsondata)
        self.att_pth = att_pth
        self.entity=entitynum
        self.relation=relationnum


    def __getitem__(self, index):
        fmname,target=self.dataset[index]
        fm=np.load(os.path.join(self.att_pth,fmname+'.npz'))['feat']
        target0 =   torch.tensor(target[0]).long()
        target1 =   torch.tensor(target[1]).long()
        target2 =   torch.tensor(target[2]).long()
        target3 =   torch.tensor(target[3]).long()
        target4 =   torch.tensor(target[4]).long()
        target5 =   torch.tensor(target[5]).long()
        target6 =   torch.tensor(target[6]).long()

        fm=torch.from_numpy(fm).float()#2048

        return fm,target0,target1,target2,target3,target4,target5,target6,fmname

    def __len__(self):
        return len(self.dataset)







