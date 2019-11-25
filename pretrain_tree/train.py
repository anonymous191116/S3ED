import torch
import torch.optim as optim
import json
import torch.nn.init as init
import tensorboardX as tb
import time
from dataloader import *
from tree import VPtree
import os
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    # if classname.find('Conv') != -1:
    #     init.xavier_normal_(m.weight.data)
    #     init.constant_(m.bias.data, 0.0)
    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


opt={
    'batch_size':   256,
    'max_epoch':    5,
    'val_iter':     1,
    'log_iter':     10,
    'lr':           6e-4,
    'lr_drop_per':  1,
    'lr_drop':      0.6,
    'jsondata_pth': "data/TreeNodeIndex.json",
    'attdata_pth':   "data/low_blurcoco_resnet_att/",
    'save_pth':     "data/save_tree_low/",
}


RELATION_NUM=247
ENTITY_NUM=749



def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

tb_summary_writer = tb and tb.SummaryWriter(opt['save_pth'])

jsondata=opt['jsondata_pth']
attdata=opt['attdata_pth']

train_dataset= H5Dataset(jsondata=jsondata,att_pth=attdata,entitynum=ENTITY_NUM,relationnum=RELATION_NUM,opt=0)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt['batch_size'],num_workers=8)

val_dataset=H5Dataset(jsondata=jsondata,att_pth=attdata,entitynum=ENTITY_NUM,relationnum=RELATION_NUM,opt=1)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=opt['batch_size'],num_workers=8)
print('init dataset success! Length of train: %d  Lenth of val:  %d'%(len(train_loader),len(val_loader)))
net=VPtree(ENTITY_NUM,RELATION_NUM)
print(net)
net.apply(weights_init)
print("init para success!")

net.cuda()
criterion_E = nn.CrossEntropyLoss()
criterion_R = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt['lr_drop_per'], gamma=opt['lr_drop'])
iter=0
running_loss = 0.0
entity_loss = 0.0
relation_loss = 0.0

for epoch in range(opt['max_epoch']):
    for i, data in enumerate(train_loader):
        start = time.time()
        inputs,target0,target1,target2,target3,target4,target5,target6,fname = data
        inputs=inputs.cuda()
        target0=target0.cuda()
        target1=target1.cuda()
        target2=target2.cuda()
        target3=target3.cuda()
        target4 = target4.cuda()
        target5 = target5.cuda()
        target6=target6.cuda()
        outputs = net(inputs)
        optimizer.zero_grad()

        loss_e = (criterion_E(outputs[0], target0)+criterion_E(outputs[2], target2)+criterion_E(outputs[4], target4)+criterion_E(outputs[6], target6))/4
        loss_r = (criterion_R(outputs[1], target1)+criterion_R(outputs[3], target3)+criterion_R(outputs[5], target5))/3
        loss=loss_e+loss_r

        #optimizer.zero_grad()
        loss.backward()
        entity_loss+=loss_e.item()
        relation_loss += loss_r.item()
        running_loss += loss.item()
        optimizer.step()
        iter+=1
        if iter % opt['log_iter'] == 0:
            print('[%d, %5d] loss: %.5f     entity_loss:%.5f     relation_loss:%.5f     time_iter:%.5f' %
                  (epoch + 1, iter + 1, running_loss,entity_loss,relation_loss,time.time() - start))

            add_summary_value(tb_summary_writer, 'loss2/train_loss', running_loss, iter)
            add_summary_value(tb_summary_writer, 'loss2/entity_loss', entity_loss, iter)
            add_summary_value(tb_summary_writer, 'loss2/relation_loss', relation_loss, iter)

            running_loss = 0.0
            relation_loss =0.0
            entity_loss =0.0
    if epoch % opt['val_iter']==0:
        score=0.0
        entity_score=0.0
        relation_score=0.0
        sumsc=0.0
        with torch.no_grad():
            for ii, data in enumerate(val_loader):
                inputs, target0, target1, target2, target3, target4, target5, target6, fname = data
                inputs = inputs.cuda()
                target0 = target0.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
                target3 = target3.cuda()
                target4 = target4.cuda()
                target5 = target5.cuda()
                target6 = target6.cuda()

                outputs = net(inputs)
                loss_e = (criterion_E(outputs[0], target0) + criterion_E(outputs[2], target2) + criterion_E(
                    outputs[4], target4) + criterion_E(outputs[6], target6)) / 4
                loss_r = (criterion_R(outputs[1], target1) + criterion_R(outputs[3], target3) + criterion_R(
                    outputs[5], target5)) / 3
                loss = loss_e + loss_r

                entity_score += loss_e.item()
                relation_score += loss_r.item()
                score += loss.item()

                if ii % opt['log_iter'] == 0:
                    print('val the model: [%d, %5d] loss: %.5f     entity_loss:%.5f     relation_loss:%.5f' %
                          (epoch + 1, ii + 1, score, entity_score, relation_score))
                    sumsc+=score
                    score = 0.0
                    relation_score = 0.0
                    entity_score = 0.0
        sumsc/=500
        add_summary_value(tb_summary_writer, 'val_loss', sumsc, iter)
        scheduler.step()
        add_summary_value(tb_summary_writer, 'learning rate', optimizer.param_groups[0]['lr'], iter)
        torch.save(net.state_dict(), opt['save_pth'] + 'model' + str(epoch) + '.pth')
