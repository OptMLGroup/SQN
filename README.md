# SQN: Sampled Quasi-Newton Methods for Deep Learning

Authors: [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/) and [Martin Takáč](http://mtakac.com/)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Introduction
This is a Python software package for solving Deep Learning problems using sampled quasi-Newton methods. More specifically, the user can select one of two methods:
- sampled LBFGS,
- sampled LSR1.

See [paper](https://arxiv.org/abs/1901.09997) for details.

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




## Dependencies
* Numpy
* [tensorflow](https://www.tensorflow.org/)>=1.2




## How to Run
### Train
By default, the code is running in the training mode on a single gpu. For running the code, one can use the following command:
```bash
python main.py
```


The list of specific parameters are available in the ``parameters.py`` file.


### Logs
All logs are stored in ``.pkl`` file in ``./_saved_log_files`` directory.

## Paper
[Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample](https://arxiv.org/abs/1901.09997). 

