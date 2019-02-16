# SQN: Sampled Quasi-Newton Methods for Deep Learning

Authors: [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/) and [Martin Takáč](http://mtakac.com/)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Introduction
This is a Python software package for solving Deep Learning problems using sampled quasi-Newton methods. More specifically, the user can select one of two methods:
- sampled LBFGS (S-LBFGS),
- sampled LSR1 (S-LSR1).

See [paper](https://arxiv.org/abs/1901.09997) for details.

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

### Sampled LSR1 (S-LSR1)

### Other problems

## Paper
[Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample](https://arxiv.org/abs/1901.09997). 

