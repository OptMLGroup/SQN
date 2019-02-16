# SQN: Sampled Quasi-Newton Methods for Deep Learning

Authors: [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home), [Majid Jahani](http://coral.ise.lehigh.edu/maj316/) and [Martin Takáč](http://mtakac.com/)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Paper
Implementation of our paper: [Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample](https://arxiv.org/abs/1901.09997). 


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
