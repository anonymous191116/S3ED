import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
#from tree import VPtree
import torch
import pdb
import os
from collections import OrderedDict

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train(opt, data_loader, model, visualizer, tree_param):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    
    if len(tree_param['pretrain_path'])!=0:
        model.netTree.load_state_dict(torch.load(tree_param['pretrain_path']))
        print("loading pretrained tree model from %s"%tree_param['pretrain_path'])

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            #tree step
            sub1, obj1, sub2, obj2, re12, re34, reroot, \
            logit_sub1, logit_obj1, logit_sub2, logit_obj2, \
            logit_re12, logit_re34, logit_reroot = model.netTree(data['att'].cuda())

            tree_pre = [logit_sub1, logit_re12, logit_obj1, logit_reroot, logit_sub2, logit_re34, logit_obj2]
            tree_gt  = data['tree_label'].cuda()# b 7 5

            tree_loss = torch.tensor(0.).cuda()
            for num_tree in range(5):
                for num_node in range(7):
                    tree_loss +=  model.tree_loss_func(tree_pre[num_node],tree_gt[::,num_node,num_tree])/5.

            tree_loss_log = tree_loss.item()

            att = torch.cat((sub1, re12, obj1,, reroot sub2,, re34 obj2),1)

            #gan step
            model.set_input(data,att)
            model.get_tree_loss(tree_loss)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                print('tree loss : %.4f' % tree_loss_log)
                results = model.get_current_visuals()
                psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
                print('PSNR on Train = %f' %
                      (psnrMetric))
                visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()





opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
model = create_model(opt)

tree_param = {
    'pretrain_path':'',
    'en_num':749,
    're_num':247,
    'lr':8e-4
}

visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer,tree_param)
