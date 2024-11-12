# posdc

- Bayes decoding of animal's position in an linear environment
- Depending on the customized package [neuralib](https://github.com/ytsimon2004/neuralib), [Documentation](https://neuralib.readthedocs.io/en/latest/index.html)
- Core function [place_bayes](https://neuralib.readthedocs.io/en/latest/api/neuralib.model.bayes_decoding.position.html#neuralib.model.bayes_decoding.position.place_bayes)


------------------------------

# How to set up in a local machine?

## 1. Clone or download the project locally

## 2. Create conda environment

```bash
conda create -n posdc python~=3.10.0 -y
conda activate posdc
```

## 3. Install required packages

- First `cd` to the directory with [pyproject.toml](pyproject.toml)

```bash
pip install -e .
```


------------------------------

# How to run as CLI?

- See [main_cli.py](posdc/main_cli.py)

```bash
python -m posdoc.main_cli -h
```

### Example 

```bash
python -m posdoc.main_cli -F [HDF_FILE]
```