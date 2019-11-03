

# Voice-Conversion-CycleGAN2

### Paper and Dataset

**Paper：**[CYCLEGAN-VC2: IMPROVED CYCLEGAN-BASED NON-PARALLEL VOICE CONVERSION](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682897)

**Dataset：**[VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211)

<br/>

*Note: the code is not built exactly according to the paper because the GPU-memory is limited (11178MB). And the dimension of the features are 24.*

The filters of the 1D -> 2D of Generator is decreased from 2304 to 512, and the filters of the last conv in the Generator is decreased from 35 to 1. 

<br/>

### Dependencies

Ubuntu 1080ti

python 3.5

tensorflow 1.14.0

PyWorld 0.2.8

numpy 1.15.4

librosa 0.5.1

<br/>

### File Structure

```bash
|--convert.py
|--model.py
|--module.py
|--preprocess.py
|--train.py
|--utils.py
|--data--|vcc2016_training
       --|evaluation_all
```

<br/>

### Usage

#### Preprocess

```python
python preprocess.py
```

It may take 13 minutes to process the data if the same dataset is used.

<br/>

#### Train

```python
python train.py
```

It may take 2 minutes for one epoch.  500 epoch are needed in order to get good voice quality.



If other speakers are involved, please change the directory below.

```bash
train_A_dir_default = './data/vcc2016_training/SF1'
train_B_dir_default = './data/vcc2016_training/TM1'
```

<br/>

#### Inference

```python
python convert.py
```

The converted voices can be found in the directory below.

```bash
|--converted_voices
```

<br/>

### To-DO

- [x] Separate the preprocess from the training steps.
- [x] Add process bar to the code.
- [x] Accelerate the training speed through saving the models by 100 epoch.
- [x] Add the module of saving the last five epoch.
- [ ] Provide some converted samples.

