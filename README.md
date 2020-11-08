# ATCN: Agile Temporal Convolutional Network
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

Agile Temporal Convolutional Network (ATCN) is a model for high-accurate fast classification and time series prediction in resource-constrained embedded systems. ATCN is primarily designed for mobile embedded systems with performance and memory constraints such as wearable biomedical devices and real-time reliability monitoring systems. It makes fundamental improvements over the mainstream temporal convolutional neural networks, including the incorporation of separable depth-wise convolution to reduce the computational complexity of the model and residual connections as time attention machines to increase the network depth and accuracy.
  
## Installation
You only need to clone the Deep RACE repository:
```bash
git clone https://github.com/TeCSAR-UNCC/ATCN.git --recurse-submodules
```
The following instructions are for the MNIST dataset. For `DeepRACE` or `ECG` datasets, you should switch the branch to `DeepRACE` ot `ECG` by:
```bash
cd DeepRace
git checkout <branch_name>
```
`DeepRACE` or `ECG` branches will be pushed soon. Please follow the instruction in `README.md` of the `DeepRACE` or `ECG` branch to train and run inference. 

## Prerequisites
First make sure you have already installed pip3, Tkinter, and git tools:
``` bash
sudo apt install git python3-pip python3-tk
```
You should also install the following python packages:
```bash
sudo -H pip3 install scipy matplotlib seaborn numpy torch sklearn glob shutil
pip3 install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
```

## Training the network models
Change the path to the `ATCN` directory and run the `train.py`:
```bash
cd ATCN
python3 train.py --data ./data/mnist/ --conv2d -c ./checkpoints/trained_model
```

## Evaluating the models
```bash
python3 train.py --data ./data/mnist/ --conv2d --evaluate  -c ./checkpoints/trained_model
```

## Authors
* Reza Baharani - [Personal webpage](https://rbaharani.com/)

## License
Copyright (c) 2020, the University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/ATCN/MNIST/LICENSE) file for details.