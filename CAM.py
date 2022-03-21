# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import numpy as np
import cv2
import torch
from torch.nn import functional as F
from torch.nn import Upsample
from model.net import get_network
from model.config import config
from UCR_DataLoader.UCR_Archive_Dataset import UCR_DataSet
from UCR_DataLoader.UCR_Archive_Dataset import get_data
from utils import argparser
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import UCR_DataLoader.time_series_augmentation.utils.augmentation as aug

# hook the feature extractor
features_blobs = []


def plot_data_aug(x):
    aug_list = ['Jittering', 'Magnitude_Warping', 'Window_Wraping', 'Scaling']
    plt.rcParams["figure.figsize"] = 5, 2.75
    x_trn = x.transpose(1, 2).cpu().numpy()
    x_train_max = np.max(x_trn)
    x_train_min = np.min(x_trn)
    x_train = 2. * (x_trn - x_train_min) / (x_train_max - x_train_min) - 1.
    for aug_type in aug_list:
        if aug_type == 'Jittering':
            x_aug = aug.jitter(x_train, sigma=0.02)
        elif aug_type == 'Scaling':
            x_aug = aug.scaling(x_train, sigma=0.1)
        elif aug_type == 'Magnitude_Warping':
            x_aug = aug.magnitude_warp(x_train, sigma=0.1, knot=4)
        else:
            x_aug = aug.window_warp(x_train, window_ratio=0.1, scales=[0.5, 2.])
        x_tmp = x_train.squeeze(0).squeeze(1)
        x_aug_tmp = np.transpose(x_aug, (0, 2, 1)).squeeze(
            0).squeeze(0)
        t = range(len(x_tmp))

        plt.plot(t, x_tmp, 'b-', label='$X$')
        plt.plot(t, x_aug_tmp, 'r-', label='$\^{X}$')
        plt.xlabel('Sequence')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.extra_tag,
                                 'AUG_{}.pdf'.format(aug_type)))
        plt.clf()
    print('-> Plotting augmented X is done.')


def upsampler(input):
    upsampler = Upsample(scale_factor=2, mode='linear', align_corners=True)
    inp_tensor = torch.from_numpy(input).unsqueeze(0)
    output = upsampler(inp_tensor)
    output = output.squeeze(0).squeeze(0).data.cpu().numpy()
    return output


def plot(x, CAM, model):
    plt.rcParams["figure.figsize"] = 5, 2.75

    fig = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=1, nrows=7, figure=fig)

    ax1 = fig.add_subplot(spec2[0:6, 0])
    ax2 = fig.add_subplot(spec2[6:7, 0])

    x_tmp = x.squeeze(0).squeeze(0).squeeze(0)
    x_tmp = x_tmp - torch.min(x_tmp)
    x_tmp = x_tmp / torch.max(x_tmp)
    seq = range(len(x_tmp))
    extent = [seq[0]-(seq[1]-seq[0])/2., seq[-1]+(seq[1]-seq[0])/2., 0, 1]

    ax1.plot(seq, x_tmp)
    # ax1.set_title('{}'.format(model))
    ax1.set_yticks([])
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_xlabel('Sequence')

    ax2.imshow(CAM[0][np.newaxis, :], cmap="jet",
               aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax2.set_xlim(extent[0], extent[1])

    plt.tight_layout()
    plt.savefig(os.path.join(args.extra_tag, 'CAM_{}.pdf'.format(model)))
    plt.clf()


def hook_feature(module, input, output):
    features_blobs.append(input[0].data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
    # generate the class activation maps upsample to 256x256
    #size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = upsampler(cam)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # cam_img = 2*(cam_img)-1
        #cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam


def extend_4d(x, y):

    x = x.reshape(x.shape[1], x.shape[0])
    # This function is for making a 1D input time series to 4D.
    x = torch.unsqueeze(x, 0)
    return x, y


def main(args):

    global features_blobs

    finalconv_name = 'downsampler'

    transf = extend_4d if args.conv2d else None

    _, _, x_test, y_test, nb_class = get_data(args=args)

    val_dataset = UCR_DataSet(
        x=x_test, y=y_test, nb_class=nb_class, args=None, train=False, transform=transf)

    batch_size = val_dataset.x.shape[0]

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    input_size = val_dataset.length
    predict_size = val_dataset.num_class

    # create model

    print("=> creating {} model.".format(args.model_configuration))
    # Load model for training and inference
    net = get_network(args.model_configuration, input_size,
                      prediction_size=predict_size, dropout=args.dropout, use_conv2d=args.conv2d)

    best_model = os.path.join(args.checkpoint, "model_best.pth.tar")
    source_state = torch.load(best_model)
    net.load_state_dict(source_state['state_dict'])

    net.cpu().eval()

    X, Y = next(iter(val_loader))

    plot_data_aug(X[0])

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())

    for i in range(-3, -1):
        if i == -3:
            weight_softmax = torch.matmul(
                torch.transpose(params[i], 0, 1), torch.transpose(params[i+1], 0, 1))  # np.squeeze(.numpy())
        else:
            weight_softmax = torch.matmul(
                weight_softmax, torch.transpose(params[i+1], 0, 1))  # np.squeeze(.numpy())

    weight_softmax = torch.transpose(weight_softmax, 0, 1).data.cpu().numpy()

    for j in range(predict_size):
        true_positive = False
        temp_i = 0
        features_blobs = []
        while not true_positive:
            idx_class = np.argwhere(Y == j)
            if temp_i >= len(idx_class[0]):
                print('X> Class {} cannot be recognized at all.'.format(j))
                break
            x = torch.unsqueeze(X[idx_class[0][temp_i]], 0)
            logit = net(x)

            if predict_size > 2:
                h_x = F.softmax(logit, dim=1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                probs = probs.numpy()
                idx = idx.numpy()

                true_positive = (idx[0] == j)
                tmp_id = idx[0]
            else:
                if j == 0:
                    true_positive = True if (logit < 0.5) else False
                else:
                    true_positive = True if (logit >= 0.5) else False
                tmp_id = 0

            if true_positive:
                print('!> Plotting CAM for class {}.'.format(j))
                # generate class activation mapping for the top1 prediction
                CAMs = returnCAM(features_blobs[0], weight_softmax,
                                 [tmp_id], size_upsample=(1, input_size))
                plot(x, CAMs, 'db_{}_cls_{}_config_{}'.format(
                    args.dataset, j, config[args.model_configuration]['Paper_name']))
            else:
                temp_i += 1

    '''
    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % idx[0])
    #img = cv2.imread('test.jpg')
    #height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (1, input_size)), cv2.COLORMAP_JET)
    #result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', heatmap)
    '''


if __name__ == "__main__":
    args = argparser()
    main(args)
