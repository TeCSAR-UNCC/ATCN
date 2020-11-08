"""
Copyright (c) 2020, University of North Carolina at Charlotte All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Authors: Reza Baharani - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
         Steven Furgurson - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
"""

import os
import pandas as pd
import h5py
import matplotlib
import time
import shutil
import torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import numpy as np
from math import cos, pi
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model.net import ATCN
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, argparser
from data_processing.utils import data_generator

input_channels = 1
seq_length = int(784 / input_channels)


def validate(val_loader, val_loader_len, model, criterion, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    bar = Bar('Processing', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    ground_truth = np.empty((0, 0))
    predicted = np.empty((0, 0))
    for i, (inputs, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True).to(dtype=torch.long)
        inputs = inputs.cuda(non_blocking=True).to(dtype=torch.float)

        inputs = inputs.view(-1, input_channels, seq_length)
        if args.conv2d:
            inputs = torch.unsqueeze(inputs, 1)

        with torch.no_grad():
            # compute output
            output = model(inputs)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        y_pred = output.argmax(dim=1).cpu()
        predicted = np.append(predicted, y_pred)
        ground_truth = np.append(ground_truth, target.cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=i + 1,
            size=val_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg
        )
        bar.next()
    bar.finish()
    ground_truth = ground_truth.astype('int')
    predicted = predicted.astype('int')
    res = classification_report(ground_truth, predicted, target_names=labels)

    return (losses.avg, top1.avg, res)


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args):
    bar = Bar('Processing', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        c_lr = adjust_learning_rate(
            optimizer, epoch, i, train_loader_len, args)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True).to(dtype=torch.long)
        inputs = inputs.cuda(non_blocking=True).to(dtype=torch.float)

        inputs = inputs.view(-1, input_channels, seq_length)
        if args.conv2d:
            inputs = torch.unsqueeze(inputs, 1)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) | lr: {lr: .5f} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=i + 1,
            size=train_loader_len,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            lr=c_lr
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, iteration, num_iter, args):
    lr = optimizer.param_groups[0]['lr']
    return lr


def main(args):
    input_size = 784
    print('=> Making network ready for input size: {}'.format(input_size))
    train_loader, val_loader = data_generator(args.data, args.batch_size)

    # create model
    channel_sizes = args.nhid
    kernel_size = args.skrn
    dropout = args.dropout

    best_prec1 = 0

    print("=> creating model")
    # Load model for training and inference
    model = ATCN(input_size, args.predict_size, kernel_size,
                args.sdil, args.input_scaling, channel_sizes,
                inp_ch=1, dropout=dropout, conv2d=args.conv2d)

    model.cuda()

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, min_lr=1e-5, factor=args.gamma)

    # optionally resume from a checkpoint
    title = 'ECG_TCN_INPUT_SIZE={}'.format(input_size)
    if not os.path.isdir(args.checkpoint) and args.checkpoint is not '':
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint,
                                         'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss',
                          'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    if args.evaluate:
        from collections import OrderedDict
        best_model = os.path.join(args.checkpoint, "model_best.pth.tar")
        if os.path.isfile(best_model):
            print("=> loading pretrained weight '{}'".format(best_model))
            source_state = torch.load(best_model)
            print("=> Model loaded with {:.2f} Top1 accuracy.".format(
                source_state['best_prec1']))
            model.load_state_dict(source_state['state_dict'])
        else:
            print("=> no weight found at '{}'".format(best_model))

        _, _, res = validate(val_loader, len(val_loader), model, criterion)
        with open(os.path.join(args.checkpoint, 'scores.txt'), 'w') as file:
            file.write(res)
            file.close()
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))
    for epoch in range(args.start_epoch, args.epochs):

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_acc = train(train_loader, len(
            train_loader), model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, prec1, res = validate(
            val_loader, len(val_loader), model, criterion)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars(
            'loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {
                           'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        if is_best:
            with open(os.path.join(args.checkpoint, 'scores.txt'), 'w') as file:
                file.write(res)
                file.close()

        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        exp_lr_scheduler.step(val_loss)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:{:.2f} and the input size is {:d}'.format(
        best_prec1, input_size))


if __name__ == "__main__":
    args = argparser()
    main(args)
