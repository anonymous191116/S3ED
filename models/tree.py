import torch
import torch.tensor
import torch.nn as nn
import torch.nn.functional as F
class VPtree(nn.Module):
    def __init__(self,ENTITY_NUM,RELATION_NUM):
        super(VPtree, self).__init__()
        self.entity_num=ENTITY_NUM
        self.relation_num = RELATION_NUM
        self.entityconv1 = nn.Conv2d(2048,64,3)
        self.entityconv2 = nn.Conv2d(2048,64,3)
        self.entityconv3 = nn.Conv2d(2048,64,3)
        self.entityconv4 = nn.Conv2d(2048,64, 3)
        self.relationconv12 = nn.Conv2d(64+64,64, 3, padding=1)
        self.relationconv34 = nn.Conv2d(64+64,64, 3, padding=1)
        self.relationconvroot = nn.Conv2d(64+64,64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.relationconvroot = nn.Conv2d(2048, 2048, 3)
        self.fc1_entity = nn.Linear(64, self.entity_num)
        self.fc1_relation = nn.Linear(64, self.relation_num)


    def forward(self, att_feats):
        x = att_feats.permute(0, 3, 1, 2)  # [batch,2048,14,14]

        sub1 = self.relu(self.bn(self.entityconv1(x)))  # [batch,64,12,12]
        obj1 = self.relu(self.bn(self.entityconv2(x)))
        sub2 = self.relu(self.bn(self.entityconv3(x)))
        obj2 = self.relu(self.bn(self.entityconv4(x)))
        re12 = self.relu(self.bn(self.relationconv12(torch.cat((sub1,obj1),1)))) # [batch,64,12,12]
        re34 = self.relu(self.bn(self.relationconv34(torch.cat((sub2,obj2),1))))
        reroot = self.relu(self.bn(self.relationconvroot(torch.cat((re12,re34),1))))
       # print(sub1.size())
        logit_sub1 = self.fc1_entity(self.avgpool(sub1).squeeze(2).squeeze(2))  # [batch,64]
        logit_obj1 = self.fc1_entity(self.avgpool(obj1).squeeze(2).squeeze(2))
        logit_sub2 = self.fc1_entity(self.avgpool(sub2).squeeze(2).squeeze(2))
        logit_obj2 = self.fc1_entity(self.avgpool(obj2).squeeze(2).squeeze(2))
        logit_re12 = self.fc1_relation(self.avgpool(re12).squeeze(2).squeeze(2))
        logit_re34 = self.fc1_relation(self.avgpool(re34).squeeze(2).squeeze(2))
        logit_reroot = self.fc1_relation(self.avgpool(reroot).squeeze(2).squeeze(2))

        return  sub1,obj1,sub2,obj2,re12,re34,reroot,\
                logit_sub1, logit_obj1, logit_sub2, logit_obj2, \
                logit_re12, logit_re34, logit_reroot