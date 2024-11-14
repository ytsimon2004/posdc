posdc
=====

- Bayes decoding of animal's position in an linear environment
- Depending on the customized
  package [neuralib](https://github.com/ytsimon2004/neuralib), [Documentation](https://neuralib.readthedocs.io/en/latest/index.html)
- Core
  function [place_bayes](https://neuralib.readthedocs.io/en/latest/api/neuralib.model.bayes_decoding.position.html#neuralib.model.bayes_decoding.position.place_bayes)

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

### See help

```bash
posdoc -h
```

```text
Bayes decoding of animal's position in an linear environment

options:
  -h, --help            show this help message and exit
  -F FILE, --file FILE  hdf data file path
  --deconv              use deconv activity, otherwise, df_f

Decoding Option:
  --length VALUE        trial length in cm
  --spatial-bin VALUE   spatial bin size in cm
  --temporal-bin VALUE  temporal bin size in second, CURRENTLY NOT USE, DIRECTLY USE THE SAMPLING RATE OF THE DF/F
  --run                 whether select only the running epoch
  --no-position-ds      not interpolate position to neural activity shape, then interpolate neural activity to position
  --invalid-cache       invalid all the cache, and recompute

Train/Test Option:
  --train NAME           train the decoder in which behavioral session
  --cv CROSS_VALIDATION  nfold for model cross validation
  --no-shuffle           whether without shuffle the data for non-repeated cv
  --repeats N_REPEATS    run as repeat kfold cv, make number of results to `n_cv * n_repeats`
  --train-fraction TRAIN_FRACTION 
                         fraction of data for train set if `random_split` in cv, the rest will be utilized in test set.(require for --train=random-split)

Random Option:
  --random-neuron NUMBER number of random neurons
  --seed VALUE           seed for random number generator

Rastermap sorting Options:
  --rastermap            sort activity heatmap using rastermap
  --rastermap-bin VALUE  bin size for number of total neurons

Plotting Options:
  --ignore-foreach       whether to ignore foreach cv plot, and only plot the summary result
  -O PATH, --output PATH output figures to a directory

```

### Examples

#### Example of 5-fold cross validation on light session using 500 neurons, and test on the rest

```bash
posdoc --file [HDF_FILE] --run --train light-cv --cv 5 --random-neuron 500
```

#### Example of 10 repeated 5-fold cross validation on light session using 500 neurons, and test on the rest

```bash
posdoc --file [HDF_FILE] --run --train light-cv --cv 5 --random-neuron 500 --repeats 10
```

#### Remarks
- `--invalid-cache`: To invalid local cache when changing **Decoding Option**
- `--rastermap`: show sorted heatmap by [rastermap](https://github.com/MouseLand/rastermap)
- `--ignore-foreach`: ignore foreach cv plots, and only plot summary
- `--output`: specify output dir for saving the figures
- See other `TRAIN_OPTIONS` if needed: check `PositionDecodeOptions.train_test_split()`


------------------------------

Output
------------------

### figures

- `decode_foreach_cv*.pdf`: Foreach cross validation results
- `decode_summary.pdf`: Summary of cross validation in light versus dark

### CSV file

- Example output from 5-fold non-repeated cross-validation

| n_cv | session | n_trials | decode_err |
|------|---------|----------|------------|
| 0    | light   | 17       | 6.506346   |
| 0    | dark    | 64       | 33.359664  |
| 1    | light   | 16       | 6.211329   |
| 1    | dark    | 64       | 34.97227   |
| 2    | light   | 16       | 5.990199   |
| 2    | dark    | 64       | 35.223511  |
| 3    | light   | 16       | 5.608659   |
| 3    | dark    | 64       | 35.161925  |
| 4    | light   | 16       | 5.813293   |
| 4    | dark    | 64       | 34.579645  |

- `n_cv`: Number of the cross validation
- `session`: Which session of the test dataset
- `n_trials`: Number of trials of the test dataset
- `decode error`: Mean/median decode error for the session

-------------------------------


TODO/BUGFIX
------------

- [ ] `RuntimeError: pr approach the infinite`, usually happened in `--deconv`, try to specify less neuron 
using `--random-neuron`
- [ ] Test for the `--seed` usage
- [ ] Statistic for cv summary if needed


Annotations
-----------

### Numpy Array

Numpy array are annotated with following expression `Array[dtype, shape]`.
For example, `Array[int, [3, 3]]` is a 3x3 int array or `(3, 3)` in short.