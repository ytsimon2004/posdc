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

Output Files
------------------

### Figures

- `decode_foreach_cv*.pdf`: Foreach cross validation results
- `decode_summary.pdf`: Summary of cross validation in light versus dark

### Parquet file

- Example output from 5-fold non-repeated cross-validation

| n_cv (i64) | session (str) | n_trials (i64) | decode_err (f64) | trial_indices (list[i64]) | binned_err (list[f64])           |
|------------|---------------|----------------|------------------|---------------------------|----------------------------------|
| 0          | light         | 10             | 3.635864         | [5, 7, … 47]              | [22.904429, 18.559892, … 17.09…] |
| 0          | dark          | 81             | 8.658199         | [51, 52, … 126]           | [30.071842, 23.799756, … 30.65…] |
| 1          | light         | 10             | 5.738808         | [0, 1, … 47]              | [22.249407, 12.599134, … 17.27…] |
| 1          | dark          | 81             | 9.701644         | [51, 52, … 126]           | [27.65329, 17.79454, … 27.6105…] |
| 2          | light         | 9              | 4.084523         | [3, 9, … 47]              | [16.561472, 19.38851, … 19.172…] |
| 2          | dark          | 81             | 9.151024         | [51, 52, … 126]           | [27.966511, 20.448634, … 28.03…] |
| 3          | light         | 9              | 3.85323          | [4, 6, … 47]              | [38.950859, 13.562917, … 27.73…] |
| 3          | dark          | 81             | 8.98436          | [51, 52, … 126]           | [28.859011, 19.658932, … 27.56…] |
| 4          | light         | 9              | 4.877178         | [2, 24, … 47]             | [26.855452, 5.773164, … 16.729…] |
| 4          | dark          | 81             | 9.153177         | [51, 52, … 126]           | [27.265134, 19.030994, … 28.35…] |



- `n_cv`: Number of the cross validation
- `session`: Which session of the test dataset
- `n_trials`: Number of trials of the test dataset
- `decode_err`: Mean/median decode error for the session
- `trial_indices`: trial indices for test set
- `binned_err`: Binned decoding error, length equal to number of bins

### Cache

- `[FILE]_position_cache.npz`: Position information
- `[FILE]_ratemap_cache.npy`: RateMap (position binned activity)

-------------------------------


TODO/BUGFIX
------------

- [ ] If get an`RuntimeError: pr approach the infinite`, usually happened in `--deconv`, try to specify less neuron
  using `--random-neuron`
- [ ] Test for the `--seed` usage
- [ ] Statistic for cv summary if needed
- [ ] Other behavioral session support

Annotations
-----------

### Numpy Array

Numpy array are annotated with following expression `Array[dtype, shape]`.
For example, `Array[int, [3, 3]]` is a 3x3 int array or `(3, 3)` in short.