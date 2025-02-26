# ISLP

## Install LATEX for nbconvert

<https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex>

## Create virtual environment

python -m venv islpenv

## Activate virtual environment

source islpenv/bin/activate

## Install packages in virtual environment

python -m pip install -r .txt

## Register virtual environment to Jupyter

python -m ipykernel install --user --name=islpenv

## Convert py files to ipynb

./genipynb

## Convert ipynb files to pdf

./genpdf <dir> # convert all ipynb files in directory to pdf if the directory is a quarto project

## Install jupyterlab_templates extension
jupyter labextension install jupyterlab_templates

## Enable jupyterlab_templates extension
jupyter server extension enable --py jupyterlab_templates

## Set up a template directory
## Create a directory where you will store your notebooks.
## E.g., ~/.jupyter/templates
## Create the following file (if it does not yet exist)
## ~/.jupyter/jupyter_notebook_config.py
## Add the following line to this file. This tells jupyterLab the full path to your template directory. This must be the full path, do not use the ~ shorthand
`c.JupyterLabTemplates.templates_dir = ['/home/{username}/.jupyter/templates']`

## Run Jupyter lab

python -m jupyter lab

## Deactivate virtual environment

deactivate

## Additional packages installed to help with EDA

<https://www.nb-data.com/p/python-packages-for-automated-eda>

## How to enable code wrap using Latex

<https://github.com/quarto-dev/quarto-cli/discussions/4121>


## References

1. <https://www.bitecode.dev/p/relieving-your-python-packaging-pain>
2. <https://towardsdatascience.com/keep-your-notebooks-consistent-with-jupyterlab-templates-69f72ee25de5>
