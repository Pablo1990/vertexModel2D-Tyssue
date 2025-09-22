# vertexModel2D-Tyssue

## Use it with Google Colab -no longer working :(

Open 'main_colab.ipynib', make a copy, and run it.

## Install tyssue locally in your computer

You can install tyssue from the conda-forge channel or from source.

### Create the environment with python venv
```
python -m venv tyssue-env
source tyssue-env/bin/activate
```

### Install tyssue via pip and additional packages
```
pip install pandas==1.5.3 pathlib==1.0.1 numpy==1.23.5 tyssue==0.8.0 jupyterlab scipy matplotlib ipywidgets vispy quantities
```

## DO NOT INSTALL IPYVOLUME OR CGAL

We really do not care right now about those two requirements.

## Download the code
Via:

```
git clone --recursive https://github.com/damcb/tyssue.git
```
or downloading the zip file: https://github.com/Pablo1990/vertexModel2D-Tyssue/archive/refs/heads/main.zip

## Run the code
Through jupyter notebook or lab. 

```
jupyter notebook
main.ipynb
```
or
```
python main.py
```

## (ONLY LINUX!) Install from source

You can install tyssue from source in case you have a Linux machine.

```
git clone --recursive https://github.com/damcb/tyssue.git
cd tyssue
```

### Create the environment
```
conda env create -f environment.yml
```

### Install tyssue
```
conda activate tyssue
python setup.py install
```
