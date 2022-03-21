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

from model.net import get_network
from model.config import config
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, argparser, binary_acc
from UCR_DataLoader.UCR_Archive_Dataset import UCR_DataSet
from UCR_DataLoader.UCR_Archive_Dataset import get_data


def validate(val_loader, val_loader_len, model, criterion, nb_classes, args, labels=None):
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

        if nb_classes > 2:
            target = target.cuda(non_blocking=True).to(dtype=torch.long)
        else:
            target = target.cuda(non_blocking=True).to(dtype=torch.float)
            target = torch.unsqueeze(target, 1)

        inputs = inputs.cuda(non_blocking=True).to(dtype=torch.float)

        with torch.no_grad():
            # compute output
            output = model(inputs)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if nb_classes > 2:
            prec1 = accuracy(output, target, topk=(1,))
        else:
            prec1 = binary_acc(output, target)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        y_pred = output.argmax(dim=1).cpu()
        predicted = np.append(predicted, y_pred)
        ground_truth = np.append(ground_truth, target.cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Dataset: {dt:} | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=i + 1,
            dt=args.dataset,
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

    res = None
    if labels is not None:
        ground_truth = ground_truth.astype('int')
        predicted = predicted.astype('int')
        res = classification_report(
            ground_truth, predicted, target_names=labels)

    return (losses.avg, top1.avg, res)


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, nb_classes):
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

        if nb_classes > 2:
            target = target.cuda(non_blocking=True).to(dtype=torch.long)
        else:
            target = target.cuda(non_blocking=True).to(dtype=torch.float)
            target = torch.unsqueeze(target, 1)

        inputs = inputs.cuda(non_blocking=True).to(dtype=torch.float)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if nb_classes > 2:
            prec1 = accuracy(output, target, topk=(1,))
        else:
            prec1 = binary_acc(output, target)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) | Config: {mc:} | Dataset: {dt:} | lr: {lr: .5f} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=i + 1,
            mc=args.model_configuration,
            dt=args.dataset,
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
    '''
    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * \
            (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * \
            (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / \
                        (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    '''
    return lr


def convert_atcn1d_to2d(model_1d, model_2d):
    atcn1d_state = model_1d.state_dict()
    atcn2d_state = model_2d.state_dict()
    for (ent_1d_key, ent_1d_val), (ent_2d_key, ent_2d_val) in zip(atcn1d_state.items(), atcn2d_state.items()):
        assert(ent_1d_key == ent_2d_key)
        if len(ent_2d_val.shape) == 4:
            new_val = ent_2d_val.squeeze(2)
        else:
            new_val = ent_2d_val
        assert(ent_1d_val.shape == new_val.shape)
        atcn1d_state[ent_1d_key] = new_val
        pass
    model_1d.load_state_dict(atcn1d_state)


def extend_4d(x, y):

    x = x.reshape(x.shape[1], x.shape[0])
    # This function is for making a 1D input time series to 4D.
    x = torch.unsqueeze(x, 0)
    return x, y


def main(args):
    transf = extend_4d if args.conv2d else None

    x_train, y_train, x_test, y_test, nb_class = get_data(args=args)

    train_dataset = UCR_DataSet(
        x=x_train, y=y_train, nb_class=nb_class, args=args, train=True, transform=transf)
    val_dataset = UCR_DataSet(
        x=x_test, y=y_test, nb_class=nb_class, args=None, train=False, transform=transf)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    input_size = train_dataset.length
    predict_size = train_dataset.num_class

    # create model

    best_prec1 = 0

    print("=> creating {} model.".format(args.model_configuration))
    # Load model for training and inference
    model = get_network(args.model_configuration, input_size,
                        prediction_size=predict_size, dropout=args.dropout, use_conv2d=args.conv2d)

    model.cuda()
    if args.verbose:
        from ptflops import get_model_complexity_info
        x, _ = train_dataset.__getitem__(0)
        shape = tuple(x.shape)
        macs, params = get_model_complexity_info(model, shape, as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8} M'.format(
            'Computational complexity: ', (macs)))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        if args.get_flops:
            print('X> Will only flops and models size will be saved. Program will be terminated after logging.')
            file_name = '{}_flops_model_size.csv'.format(
                config[args.model_configuration]['Paper_name'])
            if not os.path.isfile(file_name):
                file_hnd = open(file_name, 'w')
                file_hnd.write(
                    'classifier_name,dataset_name,flops,model_size\n')
            else:
                file_hnd = open(file_name, 'a')
            file_hnd.write('{},{},{},{}\n'.format(
                config[args.model_configuration]['Paper_name'], args.dataset, macs, params))
            file_hnd.close()
            exit()
            

    print(model)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    if predict_size > 2:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.BCELoss().cuda()
    # , momentum=args.momentum,
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=np.ceil(
        args.epochs/20.).astype(int), verbose=args.verbose,  min_lr=1e-6, cooldown=np.ceil(args.epochs/40.).astype(int))
    # optimizer, step_size=70, gamma=args.gamma)

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
            '''
            target_state = OrderedDict()
            for k, v in source_state['state_dict'].items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            '''
            model.load_state_dict(source_state['state_dict'])
        else:
            print("=> no weight found at '{}'".format(best_model))

        if args.cnv and args.conv2d:
            print("=> Starting to convert model from Conv2D to Conv1D")
            val_dataset = Pysionet(
                X_val, Y_val, input_size=input_size, conv2d=False)

            val_loader_1d = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

            model_1d = TCN(input_size, args.predict_size, kernel_size,
                           args.sdil, args.input_scaling, channel_sizes,
                           dropout, conv2d=False)

            model_1d.cuda()

            print(model_1d)

            convert_atcn1d_to2d(model_1d, model)

        if args.cnv and args.conv2d:
            print("=> Using ATCN 1D for validation...")
            _, _, res = validate(val_loader_1d, len(
                val_loader_1d), model_1d, criterion)
            print("=> Saving ATCN 1D for validation...")
            atcn_1d_model_file_name = os.path.join(
                args.checkpoint, "model_best_atcn1d.pth.tar")
            torch.save(model_1d.state_dict(), atcn_1d_model_file_name)
            print("=> ATCN 1D model is saved at:{}".format(
                atcn_1d_model_file_name))
        else:
            _, _, res = validate(val_loader, len(val_loader), model, criterion)

        if res is not None:
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
            train_loader), model, criterion, optimizer, epoch, args, predict_size)

        # evaluate on validation set
        val_loss, prec1, res = validate(
            val_loader, len(val_loader), model, criterion, predict_size, args=args)

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
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            best_value = 'Best accuracy:{:3.4f}%.'.format(best_prec1)
            with open(os.path.join(args.checkpoint, 'best_value.txt'), 'w') as file:
                file.write(best_value)
                file.close()
            if res is not None:
                with open(os.path.join(args.checkpoint, 'scores.txt'), 'w') as file:
                    file.write(res)
                    file.close()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        lr_scheduler.step(val_loss)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()


if __name__ == "__main__":
    args = argparser()
    main(args)
