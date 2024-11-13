posdc
=====

- Bayes decoding of animal's position in an linear environment
- Depending on the customized package [neuralib](https://github.com/ytsimon2004/neuralib), [Documentation](https://neuralib.readthedocs.io/en/latest/index.html)
- Core function [place_bayes](https://neuralib.readthedocs.io/en/latest/api/neuralib.model.bayes_decoding.position.html#neuralib.model.bayes_decoding.position.place_bayes)


------------------------------

How to set up in a local machine?
---------------------------------

### 1. Clone or download the project locally

### 2. Create conda environment

```bash
conda create -n posdc python~=3.10.0 -y
conda activate posdc
```

### 3. Install required packages

- First `cd` to the directory with [pyproject.toml](pyproject.toml)

```bash
pip install -e .
```


------------------------------

How to run in terminal?
-----------------------

- See [main_cli.py](posdc/main_cli.py)

```bash
python -m posdoc -h
```

### Example 

```bash
python -m posdoc -F [HDF_FILE]
```

[//]: # (TODO content)

```text
usage: main_cli.py [-h] [-F FILE] [--deconv] [--length VALUE] [--spatial-bin VALUE] [--temporal-bin VALUE] [--run] [--train-session NAME] [--cv CROSS_VALIDATION] [--no-shuffle] [--repeats N_REPEATS] [--train TRAIN_FRACTION] [--random-neuron NUMBER] [--seed VALUE] [--rastermap] [--rastermap-bin VALUE] [--ignore-foreach]

Bayes decoding of animal's position in an linear environment

options:
  -h, --help            show this help message and exit
  -F FILE, --file FILE  hdf data file path
  --deconv              use deconv activity, otherwise, df_f
  --length VALUE        trial length in cm
  --spatial-bin VALUE   spatial bin size in cm
  --temporal-bin VALUE  temporal bin size in second, CURRENTLY NOT USE, DIRECTLY USE THE SAMPLING RATE OF THE DF/F
  --run                 whether select only the running epoch
  --train-session NAME  train the decoder in which behavioral session
  --cv CROSS_VALIDATION
                        nfold for model cross validation
  --no-shuffle          whether without shuffle the data for non-repeated cv
  --repeats N_REPEATS   run as repeat kfold cv, make number of results to `n_cv * n_repeats`
  --train TRAIN_FRACTION
                        fraction of data for train set if `random_split` in cv, the rest will be utilized in test set
  --random-neuron NUMBER
                        number of random neurons
  --seed VALUE          seed for random number generator
  --rastermap           sort activity heatmap using rastermap
  --rastermap-bin VALUE
                        bin size for number of total neurons
  --ignore-foreach      whether to ignore foreach cv plot, and only plot the summary result

```


------------------------------

Output
------

### figures

### CSV file

#### header

`n_cv,session,n_trials,decode_err`

-------------------------------

Annotations
-----------

### Numpy Array

Numpy array are annotated with following expression `Array[type, shape]`.
For example, `Array[int, [3, 3]]` is a 3x3 int array or `(3, 3)` in short. 
`Array[float, T]` is a 1d floating array with length `T`, or `(T,)` in short.
If `T` refer to a time, then this array is a 1d time series data, 
and all array in the same scope, such as a class, which its annotation mention `T` are shared the same length on certain axis.