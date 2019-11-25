import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from PIL import Image
import torch
import json
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle1
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

counter = 0

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    counter = i
    sub1, obj1, sub2, obj2, re12, re34, reroot, \
    logit_sub1, logit_obj1, logit_sub2, logit_obj2, \
    logit_re12, logit_re34, logit_reroot = model.netTree(data['att'].cuda())

    att = torch.cat((sub1, obj1, sub2, obj2, re12, re34, reroot), 1)
    # att =  [sub1, obj1, sub2, obj2, re12, re34, reroot]
    model.set_input(data, att)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()

    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
