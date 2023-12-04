# Sparse ISO-FLOP Transformations for Maximizing Training Efficiency

This repository accompanies the research paper titled [Sparse ISO-FLOP Transformations for Maximizing Training Efficiency](https://arxiv.org/abs/2303.11525). The paper discusses novel transformations aimed at maximizing the training efficiency of deep neural networks by introducing sparse Iso-FLOP techniques.

## Dependencies:

- [pytorch](https://pytorch.org) 
- [numpy](https://numpy.org/install/)
-  `tensorboard`

## Prerequisites 
Before running experiments or utilizing the code in this repository, please ensure the following prerequisites are met:

1. Create a conda environment using the provided
   environment configuration file.

```console
conda env create -f sparseift_env.yml
```

Activate the environment:
```console
conda activate sparseift_env
```

2. Set the environment variable `CODE_SOURCE_DIR` to the path of the source code
directory. This can be achieved using the following command:
```console
export CODE_SOURCE_DIR=/path/to/your/source/code
```

## Experiments

To run CIFAR-100/ImageNet experiments, enter the `ComputerVision` directory:
```console
cd ComputerVision/
```

### ResNet-18 on CIFAR-100

##### Dense Baseline

```console
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_dense_baseline --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/dense_baseline.yaml --run-cifar
```

### Sparse-IFT ResNet-18 on CIFAR-100

![alt attribute goes here!](/assets/sparse_ift_family.png)
*Different members of the Sparse-IFT family. Transformation of all members is parameterized by a single hyperparameter (i.e., sparsity level ($s$)). Black and white squares denote sparse and active weights, respectively. Green block indicates a non-linear activation function (e.g., BatchNorm, ReLU, LayerNorm). All transformations are derived with sparsity set to 50% as an example, are Iso-FLOP to the dense feedforward function $f_{θ_l}$, and hence can be used as a drop-in replacement of $f_{θ_l}$. See Section 2.2 of [Sparse ISO-FLOP Transformations for Maximizing Training Efficiency](https://arxiv.org/abs/2303.11525) for more details about each member.*

##### Sparse-Wide + RigL
```console
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsewide --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsewide_rigl.yaml --run-cifar
```

##### Sparse-Parallel + RigL
```console
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparseparallel --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparseparallel_rigl.yaml --run-cifar
```

##### Sparse-Factorized + RigL
```console
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsedoped --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsedoped_rigl.yaml --run-cifar
```

##### Sparse-Doped + RigL
```console
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsefactorized --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsefactorized_rigl.yaml --run-cifar
```

# Citation
If you find this work helpful or use the provided code in your research, please
consider citing our paper:

```bibtex
@inproceedings{
thangarasa2023sparseift,
title={Sparse Iso-{FLOP} Transformations for Maximizing Training Efficiency},
author={Vithursan Thangarasa and Shreyas Saxena and Abhay Gupta and Sean Lie},
booktitle={Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@NeurIPS 2023)},
year={2023},
url={https://openreview.net/forum?id=iP4WcJ4EX0}
}
```

Feel free to adapt the paths, configurations, and commands based on your specific setup.