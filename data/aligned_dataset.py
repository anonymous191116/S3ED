import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import  json
def remove_path(str,path):
    path=path+'/train/'
    return str[len(path):]
def remove_test_path(str,path):
    path=path+'/test/'
    return str[len(path):]
def name_img2id(img_id):
    if len(img_id)==31:
        return (str(int(img_id[15:27])))
    if len(img_id) == 29:
        return(str(int(img_id[13:25])))
    if len(img_id)==30:
        return (str(int(img_id[14:26])))
    if len(img_id) == 28:
        return (str(int(img_id[12:24])))
    else:
        return img_id


def make_tree_label(datas):
    label = {}
    for sp in ['train','test','val','val4w']:
        for name in datas[sp].keys():
            label[name_img2id(name)] = datas[sp][name]
    return label

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.att_path = opt.att_path
        self.tree_labels = make_tree_label(json.load(open(opt.tree_label_path)))

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        #print(AB_path)
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
        # print(AB_path)
        if self.opt.model != 'test':
            attname=remove_path(AB_path,self.root)
        else:
            attname = remove_test_path(AB_path, self.root)
        # print(attname)
        attname=name_img2id(attname)#('C'+attname)
        # print(os.path.join(self.att_path,attname+'.npz'))
        att = np.load(os.path.join(self.att_path,attname+'.npz'))['feat']
        att = torch.from_numpy(att).float()

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
        if self.opt.model != 'test':
            tree_label = torch.from_numpy(np.array(self.tree_labels[attname])).permute(1,0).long() #7 5
        else:
            tree_label = -1


        return {'A': A,
                'B': B,
                'A_paths': AB_path,
                'B_paths': AB_path,
                'att':att,
                'tree_label':tree_label}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
