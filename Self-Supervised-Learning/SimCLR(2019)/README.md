# SimCLR에 대한 pytorch 구현 코드입니다.

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 
