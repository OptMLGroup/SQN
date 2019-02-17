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
- ```network.py```: Constructs the neural network (function, gradient, Hessian and Hessian-vector products).
- ```S_LBFGS.py```, ```S_LSR1.py```: Runs the **S-LBFGS** and **S-LSR1** methods, respectively.

Each method has several method specific functions. For **S-LBFGS**:
- ```L_BFGS_two_loop_recursion.py```: LBFGS two-loop recursion for computing the search direction.
- ```sample_pairs_SY_SLBFGS.py```: Function for computing ```S```, ```Y``` curvature pairs.

For **S-LSR1**:
- ```CG_Steinhaug_matFree.py```: CG Steinhaug method for solving the TR subproblem and computing the search direction.
- ```rootFinder.py```: Root finder subroutine used in the CG Steinhaug method.
- ```sample_pairs_SY_SLSR1.py```: Function for computing ```S```, ```Y``` curvature pairs.

The ```sample_pairs_SY_SLBFGS.py``` and ```sample_pairs_SY_SLSR1.py``` functions are in the ```sampleSY.py``` file. The rest of the functions are found in the ```util_func.py```.

### Logs & Printing
All logs are stored in ``.pkl`` files in ``./_saved_log_files`` directory. The default outputs and what is printed at every iteration is:
- Iteration counter,
- Function value,
- Accuracy,
- Norm of the gradient,
- Number of function evaluations,
- Number of gradient evaluations,
- Number of Hessian-vector products,
- Total cost (# function evaluations + # gradient evaluations + # Hessian-vector products)
- Elapsed time,
- Step length (**S-LBFGS**) and TR radius (**S-LSR1**).

## Example

Here, we provide the commands for running the two methods, and the output for the first 10 iterations of both methods. We then describe how one could use our code to solve different problems.

### Sampled LBFGS (S-LBFGS)

To run the **S-LBFGS** method the syntax is: ```bash python main.py SLBFGS```

The output of the first 10 iterations is:
```
[0, 0.7568820772024124, 0.5, 0.3164452574498637, 1, 1, 0, 2, 0.03657197952270508, 2]
[1, 0.6956258550774328, 0.5, 0.0674561314351428, 2, 2, 0, 4, 0.2941138744354248, 1]
[2, 0.6928303728452202, 0.5, 0.009572266947966944, 4, 3, 1, 8, 0.7679529190063477, 1.0]
[3, 0.692176991735801, 0.5, 0.013195938306183187, 6, 4, 2, 12, 1.0290610790252686, 1.0]
[4, 0.6910937151406233, 0.5, 0.06540600186566126, 8, 5, 3, 16, 1.241429090499878, 1.0]
[5, 0.6869002623690355, 0.5, 0.02262611533759857, 12, 6, 4, 22, 1.6189680099487305, 0.25]
[6, 0.6778780169034967, 0.5, 0.07594391958980883, 23, 7, 5, 35, 2.000309944152832, 0.00048828125]
[7, 0.677836543566537, 0.5, 0.07582481736644729, 24, 8, 6, 38, 2.510093927383423, 0.0009765625]
[8, 0.6770420397015698, 0.5, 0.07593946324726104, 25, 9, 7, 41, 2.8690669536590576, 0.001953125]
[9, 0.6769673131763045, 0.5, 0.07570366951301735, 26, 10, 8, 44, 3.1529359817504883, 0.00390625]
[10, 0.6762432711651389, 0.5, 0.07603773166566419, 27, 11, 9, 47, 3.436460018157959, 0.0078125]
```

### Sampled LSR1 (S-LSR1)

To run the **S-LSR1** method the syntax is: ```bash python main.py SLSR1```

The output of the first 10 iterations is:
```
[0, 0.7568820772024124, 0.5, 0.3164452574498637, 1, 1, 1, 3, 10, 0.48604512214660645, 1]
[1, 0.6952455337961233, 0.5, 0.07691431196154719, 2, 2, 2, 6, 10, 0.7635290622711182, 2]
[2, 0.6795686674317041, 0.5, 0.09084038468631689, 3, 3, 3, 9, 10, 1.0299029350280762, 2]
[3, 0.6305025883514083, 0.6, 0.20005533664635822, 4, 4, 4, 12, 10, 1.2226409912109375, 4]
[4, 0.6305025883514083, 0.6, 0.20005533664635822, 5, 5, 5, 15, 10, 1.6073269844055176, 2.0]
[5, 0.6305025883514083, 0.6, 0.20005533664635822, 6, 6, 6, 18, 10, 1.7895889282226562, 1.0]
[6, 0.577640123436474, 0.82, 0.21616021189282156, 7, 7, 7, 21, 10, 1.9722239971160889, 1.0]
[7, 0.49736796460679933, 0.78, 0.14993963430571358, 8, 8, 8, 24, 10, 2.1734681129455566, 1.0]
[8, 0.42248074404077923, 0.87, 0.11864276500865208, 9, 9, 9, 27, 10, 2.385266065597534, 2.0]
[9, 0.33875392059040343, 0.83, 0.07470337441644294, 10, 10, 10, 30, 10, 2.5926640033721924, 4.0]
[10, 0.33875392059040343, 0.83, 0.07470337441644294, 11, 11, 11, 33, 10, 2.7668380737304688, 2.0]
[11, 0.33875392059040343, 0.83, 0.07470337441644294, 12, 12, 12, 36, 10, 3.0181710720062256, 1.0]
```

### Other problems

In order for a user to run the **S-LBFGS** and **S-LSR1** methods on different problems, there are a few things that must be modified: (1) the parameters of the neural network (Network size in ```parameters.py```), (2) the data (in ```data_generation.py```), and (3) the network (in ```network.py```). 

If users have any issues, please contact us.

## Paper
[Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample](https://arxiv.org/abs/1901.09997). 

