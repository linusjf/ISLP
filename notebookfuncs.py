# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: islpenv
#     language: python
#     name: islpenv
# ---

# %% [markdown]
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .qmd
#       format_name: quarto
#       format_version: '1.0'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: islpenv
#     language: python
#     name: islpenv
# ---

# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

## Import up sound alert dependencies
from IPython.display import Audio, display, Markdown, Math, Latex


def allDone():
    """allDone method that plays the bell sound."""
    url = "https://sound.peal.io/ps/audios/000/064/733/original/youtube_64733.mp3"
    display(Audio(url=url, autoplay=True))

def printmd(string):
    display(Markdown(string))

def printlatex(string):
  display(Latex(string))
