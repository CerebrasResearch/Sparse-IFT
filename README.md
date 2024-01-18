# Sparse Iso-FLOP Transformations for Maximizing Training Efficiency

This is the official repository used to run the experiments in the paper that proposed Sparse-IFT. The codebase is implemented in PyTorch.

[Sparse ISO-FLOP Transformations for Maximizing Training Efficiency](https://openreview.net/pdf?id=iP4WcJ4EX0)

*Vithursan Thangarasa, Shreyas Saxena, Abhay Gupta, Sean Lie*

The paper discusses novel transformations aimed at maximizing the training efficiency (test accuracy w.r.t training FLOPs) of deep neural networks by introducing a family of Sparse Iso-FLOP Transformations.

![alt attribute goes here!](/assets/sparse_ift_family.png)
*Different members of the Sparse-IFT family. Transformation of all members is parameterized by a single hyperparameter (i.e., sparsity level ($s$)). Black and white squares denote sparse and active weights, respectively. Green block indicates a non-linear activation function (e.g., BatchNorm, ReLU, LayerNorm). All transformations are derived with sparsity set to 50% as an example, are Iso-FLOP to the dense feedforward function $f_{θ_l}$, and hence can be used as a drop-in replacement of $f_{θ_l}$. See Section 2 of [Sparse ISO-FLOP Transformations for Maximizing Training Efficiency](https://openreview.net/pdf?id=iP4WcJ4EX0) for more details about each member.*

## Dependencies:

- [pytorch](https://pytorch.org) 
- [numpy](https://numpy.org/install/)
-  `tensorboard`

## Prerequisites 
Before running experiments or utilizing the code in this repository, please ensure the following prerequisites are met:

1. Set the environment variable `CODE_SOURCE_DIR` to the path of the source code
directory. This can be achieved using the following command:
```bash
export CODE_SOURCE_DIR=/path/to/your/source/code
# e.g., export CODE_SOURCE_DIR=/Users/$USER/Documents/Sparse-IFT/ComputerVision
```

2. Create a conda environment using the provided
   environment configuration file.

```bash
cd Sparse-IFT
conda env create -f sparseift_env.yaml
```

Activate the environment:
```bash
conda activate sparseift_env
```

## Quick Start: CIFAR-100 Experiments

To run CIFAR-100/ImageNet experiments, enter the `ComputerVision` directory:
```bash
cd ComputerVision/
```

### ResNet-18 on CIFAR-100

##### Dense Baseline

```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_dense_baseline --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/dense_baseline.yaml --run-cifar
```

### Sparse-IFT ResNet-18 on CIFAR-100

This section provides instructions for running experiments on CIFAR-100 using the ResNet-18 model with different configurations of the Sparse-IFT family using the dynamic sparsity algorithm RigL. RigL introduces dynamic sparsity to optimize the training efficiency of deep neural networks.

##### Sparse-Wide + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsewide --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsewide_rigl.yaml --run-cifar
```

##### Sparse-Parallel + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparseparallel --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparseparallel_rigl.yaml --run-cifar
```

##### Sparse-Factorized + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsedoped --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsedoped_rigl.yaml --run-cifar
```

##### Sparse-Doped + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_cifar100_sparsefactorized --base-dir /path/to/experiment/directory/ --base-cfg CIFAR/configs/resnet18/base.yaml --exp-cfg CIFAR/configs/resnet18/sparseift/sparsefactorized_rigl.yaml --run-cifar
```

## Quick Start: ImageNet Experiments

##### Dense Baseline

```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_imagenet_dense_baseline --base-dir /path/to/experiment/directory/ --base-cfg ImageNet/configs/resnet18/base.yaml --exp-cfg ImageNet/configs/resnet18/dense_baseline.yaml 
```

### Sparse-IFT ResNet-18 on ImageNet
This section provides instructions for running experiments on ImageNet using the ResNet-18 model with different configurations of the Sparse-IFT family using the dynamic sparsity algorithm RigL.

##### Sparse-Wide + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_imagenet_sparsewide --base-dir /path/to/experiment/directory/ --base-cfg ImageNet/configs/resnet18/base.yaml --exp-cfg ImageNet/configs/resnet18/sparseift/sparsewide_rigl.yaml 
```

##### Sparse-Parallel + RigL
```bash
python launch_utils/prepare_job_commands.py --job-name resnet18_imagenet_sparseparallel --base-dir /path/to/experiment/directory/ --base-cfg ImageNet/configs/resnet18/base.yaml --exp-cfg ImageNet/configs/resnet18/sparseift/sparseparallel_rigl.yaml 
```

# Citation
If you find this work helpful or use the provided code in your research, please
consider citing our paper:

```bibtex
@inproceedings{thangarasa2023sparseift,
title={Sparse Iso-{FLOP} Transformations for Maximizing Training Efficiency},
author={Vithursan Thangarasa and Shreyas Saxena and Abhay Gupta and Sean Lie},
booktitle={Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@NeurIPS 2023)},
year={2023},
url={https://openreview.net/forum?id=iP4WcJ4EX0}
}
```

Feel free to adapt the paths, configurations, and commands based on your specific setup.