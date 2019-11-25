from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
# import networks_addSemFea as networks
import torch
from tree import VPtree

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def init_tree(self, en=749, re=247):
        self.netTree = VPtree(en, re).cuda()
        self.tree_loss_func = torch.nn.CrossEntropyLoss()

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)

        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        self.use_tree = True
        if self.use_tree:
            self.init_tree()
            print('using vptree...')
            self.load_network(self.netTree, 'T', opt.which_epoch)
            print('-----------------------------------------------')
        else:
            print('tree not use...')
            print('-----------------------------------------------')

        self.netTree.eval()
        self.netG.eval()

    def set_input(self, input,att):
        # we need to use single_dataset mode
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.att = att

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            self.fake_B = self.netG.forward(self.real_A,self.att)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('Blurred', real_A), ('DeBlurred', fake_B),('Real',real_B)])
