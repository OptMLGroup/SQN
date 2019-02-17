# SQN: Sampled Quasi-Newton Methods for Deep Learning

Authors: [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/) and [Martin Takáč](http://mtakac.com/)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Introduction
This is a Python software package for solving a toy classification problem using neural networks. More specifically, the user can select one of two methods:
- **sampled LBFGS (S-LBFGS)**,
- **sampled LSR1 (S-LSR1)**,

to solve the problem described below. See [paper](https://arxiv.org/abs/1901.09997) for details.

Note, the code is extendible to solving other deep learning problems (see comments below).

## Problem
Consider the following simple classification problem, illustrated in the figure below, consisting of two classes each with 50 data points. We call this the **sin classification problem**. We trained a small fully conncted neural network with sigmoid activation functions and 4 hidden layers with 2 nodes in each layer.

<img src="https://user-images.githubusercontent.com/17861925/52918364-18a5ab00-32c4-11e9-8486-0f5a4cc97178.png" width="450" height="300">

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
In this section we describe all the parameters needed to run the methods on the **sin classification problem**. The list of all parameters is available in the ``parameters.py`` file.

The parameters for the **problem** are:
- ```num_pts```: Number of data points (per class)  (default ```num_pts = 50```)
- ```freq```: Frequency (default ```freq = 8```)
- ```offset```: Offset (default ```offset = 0.8```)
- ```activation```: Activation (default ```activation = "sigmoid"```)
- ```FC1```, ```FC2```,..., ```FC6```: Network size (number of nodes in each hidden layer) (default all equation to ```2```)

All these parameters can be changed in ```parameters.py```. Note that ```FC1``` and ```FC6``` should both be equal to ```2``` since this is the input and output size.

The hyperparameters for the **methods** are:
- ```seed```: Random seed (default ```seed = 67```)
- ```numIter```: Maximum number of iterations (default ```numIter = 1000```)
- ```mmr```: Memory length (default ```mmr = 10```)
- ```radius```: Sampling radius (default ```radius = 1```)
- ```eps```: Tolerance for updating QN matrices  (default ```eps = 1e-8```)
- ```eta```: TR tolerance (default ```eta = 1e-6```)
- ```delta_init```: Initial trust region radius  (default ```delta_init = 1```)
- ```alpha_init```: Initial step length  (default ```alpha_init = 1```)
- ```epsTR```: Tolerance of CG Steinhaug  (default ```epsTR = 1e-10```)
- ```cArmijo```: Armijo sufficient decrease parameter (default ```cArmijo = 1e-4```)
- ```rhoArmijo```: Armijo backtracking factor (default ```rhoArmijo = 0.5```)
- ```init_sampling_SLBFGS```: Initial sampling SLBFGS (default ```init_sampling_SLBFGS = "on"```)

All these parameters can be changed in ```parameters.py```.

### Functions
In this section we describe all the functions needed to run the code. For both methods:
- ```main.py```: This is the main file that runs the code for both methods. For each method: (1) gets the input parameters required (```parameters.py```), (2) gets the data for the **sin classification problem** (```data_generation.py```), (3) constructs the neural network (```network.py```), and (4) runs the method (```S_LBFGS.py``` or ```S_LSR1.py```).
-```parameters.py```: Sets all the parameters.
-```data_generation.py```: Generates the data.
- ```network.py```: Constructs the neural network.
- ```S_LBFGS.py```, ```S_LSR1.py```: Runs the **S-LBFGS** and **S-LSR1** methods, respectively.

Each method has several method specific functions. For **S-LBFGS**:
- ```L_BFGS_two_loop_recursion.py```: LBFGS two-loop recursion for computing the search direction.
- ```sample_pairs_SY_SLBFGS.py```: Function for computing ```S```, ```Y``` curvature pairs.

For **S-LSR1**:
- ```CG_Steinhaug_matFree.py```: CG Steinhaug method for solving the TR subproblem and computing the search direction.
- ```rootFinder.py```: Root finder subroutine used in the CG Steinhaug method.
- ```sample_pairs_SY_SLSR1.py```: Function for computing ```S```, ```Y``` curvature pairs.

The ```sample_pairs_SY_SLBFGS.py``` and ```sample_pairs_SY_SLSR1.py``` functions are in the ```sampleSY.py``` file. The rest of the functions are found in the ```util_func.py```.

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

