# vertexModel2D-Tyssue

## Instructions to install tyssue

You can install tyssue from the conda-forge channel or from source.

### Create the environment
```
conda create -n tyssue-env
conda activate tyssue-env
```

### Install tyssue via conda-forge
```
conda install -c conda-forge tyssue
```

## Install additional packages
```
pip install pandas==1.5.3 pathlib==1.0.1 numpy==1.23.5 jupyterlab
```

## DO NOT INSTALL IPYVOLUME OR CGAL

We really do not care right now about those two requirements.

## Run the code
```
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
