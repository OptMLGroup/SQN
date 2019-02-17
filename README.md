# SQN: Sampled Quasi-Newton Methods for Deep Learning

Authors: [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/) and [Martin Takáč](http://mtakac.com/)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Introduction
This is a Python software package for solving a toy classification problem using neural networks. More specifically, the user can select one of two methods:
- sampled LBFGS (S-LBFGS),
- sampled LSR1 (S-LSR1).

See [paper](https://arxiv.org/abs/1901.09997) for details.

Note, the code is extendible to solving other deep learning problems (see comments below).

## Problem
The problem we are solving is...

## Citation
If you use SQN for your research, please cite:

```
@article{berahas2019quasi,
  title={Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample},
  author={Berahas, Albert S and Jahani, Majid and Tak{\'a}{\v{c}}, Martin},
  journal={arXiv preprint arXiv:1901.09997},
  year={2019}
}
```

## Usage Guide
The algorithms can be run using the syntax:
```bash 
python main.py method
```
where ```method = SLBFGS``` or ```method = SLSR1```

By default, the code runs on a single GPU.

### Dependencies
* Numpy
* [TensorFlow](https://www.tensorflow.org/)>=1.2

### Parameters
The parameters for the problem are:


The hyperparameters for the methods are:
- Random seed
- Maximum number of iterations
- Memory length
- Sampling radius
- Tolerance for updating QN matrices
- TR tolerance
- Initial trust region radius ```init_TR``` (default ```init_TR = 1```)
- Initial step length
- Tolerance of CG Steinhaug
- Armijo sufficient decrease parameter
- Armijo backtracking factor
- Initial sampling SLBFGS

The list of specific parameters are available in the ``parameters.py`` file.

### Functions
In order to run the code, one needs the following functions:

### Logs
All logs are stored in ``.pkl`` files in ``./_saved_log_files`` directory. The default outputs are:
- Iteration counter,
- Function value,

### Output printing

## Example

Here, we provide a working example of how to use the SQN code. We first describe and set-up a toy problem, and illustrate how to run the S-LBFGS and S-LSR1 method. We then describe how one could use our code to solve different problems.

In general, to solve a problem using SQN, the user must...



### Sampled LBFGS (S-LBFGS)

```
[1, 0.6956258550774328, 0.5, 0.0674561314351428, 2, 2, 0, 4, 0.12931609153747559, 1]
[2, 0.6928303728452202, 0.5, 0.009572266947966944, 4, 3, 1, 8, 0.39708590507507324, 1.0]
[3, 0.692176991735801, 0.5, 0.013195938306183187, 6, 4, 2, 12, 0.5926470756530762, 1.0]
[4, 0.6910937151406233, 0.5, 0.06540600186566126, 8, 5, 3, 16, 0.8022611141204834, 1.0]
[5, 0.6869002623690355, 0.5, 0.02262611533759857, 12, 6, 4, 22, 1.0328218936920166, 0.25]
[6, 0.6778780169034967, 0.5, 0.07594391958980883, 23, 7, 5, 35, 1.3343679904937744, 0.000488]
```

### Sampled LSR1 (S-LSR1)

### Other problems

## Paper
[Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample](https://arxiv.org/abs/1901.09997). 

