import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import pandas as pd
import h5py
import scipy as sc
from scipy.signal import hilbert, butter, lfilter
from biosppy.signals.tools import analytic_signal
from biosppy.signals import ecg
import random
'''
code borrowed from
https://github.com/ismorphism/DeepECG/blob/master/Conv1D_ECG.py
'''


def get_data(file_name):
    hf = h5py.File(file_name, 'r')

    X_train = np.array(hf.get('X_train'))
    Y_train = np.array(hf.get('Y_train'))
    X_val = np.array(hf.get('X_val'))
    Y_val = np.array(hf.get('Y_val'))
    Weights = np.array(hf.get('Weights'))

    return X_train, Y_train, X_val, Y_val, Weights


# Helper functions needed for data augmentation
def stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def fit_tolength(source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target


def random_resample(signals, upscale_factor=1):
    n_signals = 1
    length = signals.shape[0]
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length*80/120),
        high=int(length*80/60),
        size=[n_signals, upscale_factor])
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [stretch_squeeze(s, l)
            for s, nl in zip(signals, new_length) for l in nl]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


def plot(signal, fileName):
    import matplotlib.pyplot as plt
    size = len(signal)
    for i in range(size):
        plt.plot(signal[i], label='Signal: {:d}'.format(i))
    plt.legend()
    file = './tmp/{}'.format(fileName)
    print("File is saved in {}".format(file))
    plt.savefig(file)
    plt.close()


def save_data(folder_path, file_name, fs=300, big=1800, samples=1800, training_size=0.7, seed=7, hilbert=False):
    mypath = folder_path  # Training directory
    onlyfiles = [f for f in listdir(mypath) if (
        isfile(join(mypath, f)) and f[0] == 'A')]
    bats = [f for f in onlyfiles if f[7] == 'm']
    check = 100
    mats = [f for f in bats if (
        np.shape(sio.loadmat(mypath + f)['val'])[1] >= check)]
    size = len(mats)
    print('Total training size is ', size)
    if hilbert:
        input_size = samples
    else:
        input_size = big

    X_tmp = []

    print("Generating data for input size = {}".format(big))

    for i in range(size):
        dummy = sio.loadmat(mypath + mats[i])['val'][0, :]
        X_tmp.append(dummy)
        print('=> Reading data {:.2f}'.format((i+1)*100/size), end='\r')
    print()
    target_train = np.zeros((size, 1))
    Train_data = pd.read_csv(mypath + 'REFERENCE.csv',
                             sep=',', header=None, names=None)
    for i in range(size):
        if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
            target_train[i] = 0
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
            target_train[i] = 1
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
            target_train[i] = 2
        else:
            target_train[i] = 3
        print('=> Processed {:.2f}'.format((i+1)*100/size), end='\r')
    print()

    xNormal = []
    xAfib = []
    xOthers = []
    xNoisy = []

    for i in range(size):
        # for i in range(200):
        dummy = X_tmp[i]
        dummy = butter_bandpass_filter(dummy, 3, 45, fs, order=5)
        #res = ecg.ecg(dummy, sampling_rate=fs, show=False)
        # idx = res[2]  # IDX of R-Peack
        #mean_pr = dummy[idx].mean()
        #std_pr = dummy[idx].std()
        #if abs(dummy.min()) > dummy.max():
            # Reverse prob
        #    dummy = -1*dummy
        if (len(dummy) - big) >= 0:
            howMany = len(dummy) // input_size
            for j in range(howMany):
                b = dummy[j*input_size:(j+1)*input_size]
                #b = (b - b.min())/b.ptp()
                if target_train[i] == 0:
                    xNormal.append(b)
                elif target_train[i] == 1:
                    xAfib.append(b)
                else:  # target_train[i] == 2:
                    xOthers.append(b)
                # else:
                #    xNoisy.append(b)
        else:
            resd = big-len(dummy)
            dummy = np.pad(dummy, (0, resd), 'constant', constant_values=(0))
            #dummy = (dummy - dummy.min())/dummy.ptp()
            if target_train[i] == 0:
                xNormal.append(dummy)
            elif target_train[i] == 1:
                xAfib.append(dummy)
            else:  # if target_train[i] == 2:
                xOthers.append(b)
            # else:
            #    xNoisy.append(b)
        print('=> Catagorized {:.2f}'.format((i+1)*100/size), end='\r')
    print()

    np.random.seed(seed)
    xNormalSize = [i for i in range(len(xNormal))]
    xAfibSize = [i for i in range(len(xAfib))]
    xOthersSize = [i for i in range(len(xOthers))]
    #xNoisySize = [i for i in range(len(xNoisy))]

    total_size = len(xNormal) + len(xAfib) + len(xOthers)  # + len(xNoisy)
    weights = [total_size/len(xNormal), total_size /
               len(xAfib), total_size/len(xOthers)]  # , total_size/len(xNoisy)]
    #

    permutationsNormal = np.random.permutation(xNormalSize)
    permutationsAfib = np.random.permutation(xAfibSize)
    permutationsOthers = np.random.permutation(xOthersSize)
    #permutationsNoisy = np.random.permutation(xNoisySize)

    xNormal_tmp = np.array(xNormal)
    xAfib_tmp = np.array(xAfib)
    xOthers_tmp = np.array(xOthers)
    #xNoisy_tmp = np.array(xNoisy)

    if hilbert:
        xNormal = np.zeros((xNormal_tmp.shape[0], input_size))
        xAfib = np.zeros((xAfib_tmp.shape[0], input_size))
        xOthers = np.zeros((xOthers_tmp.shape[0], input_size))
        #xNoisy = np.zeros((xNoisy_tmp.shape[0], input_size))
        for i in range(xNormal.shape[0]):
            xNormal[i] = insfreq(xNormal_tmp[i], N=samples)
            print('=> Transformed normal data {:.2f}'.format(
                (i+1)*100/xNormal.shape[0]), end='\r')
        print()
        for i in range(xAfib.shape[0]):
            xAfib[i] = insfreq(xAfib_tmp[i], N=samples)
            print('=> Transformed AF data {:.2f}'.format(
                (i+1)*100/xAfib.shape[0]), end='\r')
        print()
        for i in range(xOthers.shape[0]):
            xOthers[i] = insfreq(xOthers_tmp[i], N=samples)
            print('=> Transformed other data {:.2f}'.format(
                (i+1)*100/xOthers.shape[0]), end='\r')
        print()
        for i in range(xNoisy.shape[0]):
            xNoisy[i] = insfreq(xNoisy_tmp[i], N=samples)
            print('=> Transformed noisy data {:.2f}'.format(
                (i+1)*100/xNoisy.shape[0]), end='\r')
        print()
    else:
        xNormal = xNormal_tmp
        xAfib = xAfib_tmp
        xOthers = xOthers_tmp
        #xNoisy = xNoisy_tmp

    #xNormal = (xNormal - np.min(xNormal))/np.ptp(xNormal)
    #xAfib = (xAfib - np.min(xAfib))/np.ptp(xAfib)
    #xOthers = (xOthers - np.min(xOthers))/np.ptp(xOthers)

    xNormal = xNormal[permutationsNormal, :]
    xAfib = xAfib[permutationsAfib, :]
    xOthers = xOthers[permutationsOthers, :]
    #xNoisy = xOthers[permutationsNoisy, :]

    XNormal_train = xNormal[:int(training_size * xNormal.shape[0]), :]
    XAfib_train = xAfib[:int(training_size * xAfib.shape[0]), :]
    XOther_train = xOthers[:int(training_size * xOthers.shape[0]), :]
    #xNoisy_train = xNoisy[:int(training_size * xNoisy.shape[0]), :]

    XNormal_val = xNormal[int(training_size * xNormal.shape[0]):, :]
    XAfib_val = xAfib[int(training_size * xAfib.shape[0]):, :]
    XOther_val = xOthers[int(training_size * xOthers.shape[0]):, :]
    #xNoisy_val = xNoisy[int(training_size * xNoisy.shape[0]):, :]

    '''
    xAfibOverSampled_train = np.zeros(
        (len(XNormal_train), input_size))
    for i in range(len(XNormal_train)):
        data = random.choice(XAfib_train)
        #new_data = random_resample(data)
        xAfibOverSampled_train[i] = (data)

    xOthersOverSampled_train = np.zeros(
        (len(XNormal_train), input_size))
    for i in range(len(XNormal_train)):
        data = random.choice(XOther_train)
        #data = random_resample(data)
        xOthersOverSampled_train[i] = (data)
    
    xNoisySampled_train = np.zeros(
        (len(XNormal_train), big))
    for i in range(len(XNormal_train)):
        data = random.choice(xNoisy_train)
        xNoisySampled_train[i] = (data)
    
    
    xAfibOverSampled_val = np.zeros(
        (len(XNormal_val), input_size))
    for i in range(len(XNormal_val)):
        data = random.choice(XAfib_val)
        xAfibOverSampled_val[i] = (data)

    xOthersOverSampled_val = np.zeros(
        (len(XNormal_val), input_size))
    for i in range(len(XNormal_val)):
        data = random.choice(XOther_val)
        xOthersOverSampled_val[i] = (data)

    xNoisyOverSampled_val = np.zeros(
        (len(XNormal_val), big))
    for i in range(len(XNormal_val)):
        data = random.choice(xNoisy_val)
        xNoisyOverSampled_val[i] = (data)
    '''

    Xtrain = np.concatenate(
        (XNormal_train,
         XAfib_train,
         XOther_train
         # , xNoisy_train
         )
    )
    print("=> Train specification:")
    print("\t=> XNormal_train:{}".format(len(XNormal_train)))
    print("\t=> XAfib_train:{}".format(len(XAfib_train)))
    print("\t=> XOther_train:{}".format(len(XOther_train)))


    '''
    Xtrain = np.concatenate(
        (XNormal_train,
         xAfibOverSampled_train,
         xOthersOverSampled_train,
         xNoisySampled_train)
    )
    '''
    Ytrain = np.concatenate((
        np.zeros((1, XNormal_train.shape[0])),
        np.ones((1, XAfib_train.shape[0])),
        2*np.ones((1, XOther_train.shape[0]))  # ,
        #3*np.ones((1, xNoisySampled_train.shape[0]))
    ), axis=1)

    XVal = np.concatenate(
        (
            XNormal_val,
            XAfib_val,
            XOther_val  # ,
            # xNoisy_val
        )
    )

    print("=> Validation specification:")
    print("\t=> XNormal_train:{}".format(len(XNormal_val)))
    print("\t=> XAfib_train:{}".format(len(XAfib_val)))
    print("\t=> XOther_train:{}".format(len(XOther_val)))

    # XVal = np.concatenate(
    #    (XNormal_val, xAfibOverSampled_val, xOthersOverSampled_val, xNoisyOverSampled_val))
    YVal = np.concatenate((
        np.zeros((1, XNormal_val.shape[0])),
        np.ones((1, XAfib_val.shape[0])),
        2*np.ones((1, XOther_val.shape[0]))  # ,
        #3*np.ones((1, xNoisyOverSampled_val.shape[0])),
    ), axis=1)

    # HXtrain = np.zeros((Xtrain.shape[0], original))
    # HXVal = np.zeros((XVal.shape[0], original))

    assert(XVal.shape[0] == YVal.shape[1])
    assert(Xtrain.shape[0] == Ytrain.shape[1])

    print('=> Writing data in {}'.format(file_name))

    hf = h5py.File(file_name, 'w')

    hf.create_dataset('X_train', data=Xtrain)
    hf.create_dataset('Y_train', data=Ytrain)
    hf.create_dataset('X_val', data=XVal)
    hf.create_dataset('Y_val', data=YVal)
    hf.create_dataset('Weights', data=weights)

    hf.close()
    print("Finished")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def insfreq(signal, N=16, fs=300):
    #signal = np.pad(signal, (1, 0), 'constant', constant_values=(0))
    a, _ = analytic_signal(signal)  # , N)  # hilbert(signal, N)
    #instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(p) /
    #                           (2.0*np.pi) * fs)
    return a


if __name__ == "__main__":
    training_path = '/mnt/4tb/ECG/raw/training2017/'
    fs = 300
    input_size_scnd = 60
    sample = big = fs * input_size_scnd
    file_name = '/mnt/4tb/ECG/raw/data_weighted_3way_{}i.h5'.format(input_size_scnd)
    load_data = False
    if not load_data:
        save_data(training_path, file_name, big=big, samples=sample)
    else:
        get_data(file_name)
