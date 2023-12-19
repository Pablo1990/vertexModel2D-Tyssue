# vertexModel2D-Tyssue

## Instructions to install tyssue from conda
### Create the environment
```
conda create -n tyssue-env python=3.8
conda activate tyssue-env
```

### Install tyssue 
```
conda install -c conda-forge tyssue
```

## Install from source
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

## Install additional packages
```
pip install jupyterlab
```

## Run
```
main.ipynb
```
or
```
python main.py
```
