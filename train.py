import random
import argparse
import yaml
import torch
import os
import util
from util import build_model, train_one_epoch
from dataloader import generate_dataset_loader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR


def get_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', help='settings of detector in yaml format')
    argument = parse.parse_args()

    return argument

def calculate_average_brightness(frame):
    total_brightness = 0
    pixel_count = 0

    for row in frame:
        for pixel in row:
            total_brightness += pixel
            pixel_count += 1

    return total_brightness / pixel_count if pixel_count > 0 else 0


if __name__ == '__main__': 
    argument = get_arguments()
    assert (os.path.exists(argument.config))
    cnfig = yaml.load(open(argument.config, 'r'), Loader=yaml.Loader)
    print("******* Building models. *******")
    print(cnfig)
    model = util.build_model(cnfig['model'])
    model = model.cuda()

    if cnfig['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    model = torch.nn.DataParallel(model)

    optimus = torch.optim.AdamW(model.parameters(), lr=cnfig['lr'], weight_decay=1e-8)
    sched = MultiStepLR(optimus, milestones=[20, 25], gamma=0.1)
    loss = nn.BCEWithLogitsLoss()
    
    EMaxTree = cnfig['max_epoch']
    snapPath = cnfig['save_dir']
    if not os.path.exists(snapPath):
        os.makedirs(snapPath)

    max_epoch, max_acc = 0, 0

    for epochID in range(0, EMaxTree):
        print("******* Training epoch", str(epochID)," *******")
        print("******* Building datasets. *******")
        train_loader, val_loader = generate_dataset_loader(cnfig)
        max_epoch, max_acc, epoch_time = train_one_epoch(cnfig, model, loss, sched, optimus, epochID, max_epoch, max_acc, train_loader, val_loader, snapPath)
        print("******* Ending epoch", str(epochID)," Time ", str(epoch_time), "*******")

