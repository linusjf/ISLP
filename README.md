# ISLP

## Create virtual environment

python -m venv islpenv

## Activate virtual environment

source islpenv/bin/activate

## Install packages in virtual environment

python -m pip install -r requirements.txt

## Register virtual environment to Jupyter

python -m ipykernel install --user --name=islpenv

## Convert py files to ipynb

./genipynb

## Run Jupyter lab

python -m jupyter lab

## Deactivate virtual environment

deactivate

## Additional packages installed to help with EDA

<https://www.nb-data.com/p/python-packages-for-automated-eda>

## References

1. <https://www.bitecode.dev/p/relieving-your-python-packaging-pain>
