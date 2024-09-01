# ISLP

## Install packages globally

pip install jupyterlab jupytext

## Create virtual environment

python -m venv islpenv

## Set up virtual environment

source islpenv/bin/activate

## Install packages in virtual environment

pip install -r requirements.txt

## Register virtual environment to Jupyter

python -m ipykernel install --user --name=islpenv

## Convert py files to ipynb

./genipynb

## Run Jupyter lab

jupyter lab

## Deactivate virtual environment

deactivate
