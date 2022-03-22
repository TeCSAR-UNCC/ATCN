# ATCN: Resource-Efficient Processing of Time Series on Edge
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

This Git repo presents the source code of the research paper titled "ATCN: Resource-Efficient Processing of Time Series on Edge". ATCN enables fast time-series prediction and accurate classification in resource-constrained embedded systems. It is a family of compact networks with formalized hyperparameters that enable application-specific adjustments to be made to the model architecture. It is designed primarily for embedded edge devices with limited performance and memory, such as wearable biomedical devices and real-time reliability monitoring systems. 

## Installation
You only need to clone the Deep RACE repository:

```bash
git clone https://github.com/TeCSAR-UNCC/ATCN
```

## Dataset

You can download the 2018 UCR Time series classification archive [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

## Training the network models
In order to train the model for the entire 70 benchmarks mentioned in the paper, you should run `trainAll.sh` as follows:
```bash
./trainAll.sh
```

Ensure that the `DATA_DIR` defined in `trainAll.sh` points to the downloaded UCR time series archive folder. You can also specify the model network (T0, and T1) in the mentioned file.

## Custom network
A custom network should be defined in the file named `./model/config.py`. Two examples of configuration are already included in the `config.py` file for T0 and T1. To learn more about how network configuration knobs work, please refer to the ATCN paper.

## Author
* Reza Baharani - [Personal webpage](https://mbaharan.github.io/)

## License
Copyright (c) 2022, the University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/ATCN/master/LICENSE) file for details.

## Acknowledgments

* For more information on time series augmentation source code, please click [here](https://github.com/uchidalab/time_series_augmentation).

* The Git repo for the Class Activation Mapping (CAM) can also be found [here](https://github.com/zhoubolei/CAM).