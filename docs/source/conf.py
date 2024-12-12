# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NYgrid-python'
author = 'Bo Yuan'
release = 'v0.1.1'

from datetime import datetime
year = datetime.now().year
if year > 2023:
    copyright = f'2023-{year}, {author}'
else:
    copyright = f'{year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',    # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',    # Add links to source code
    'sphinx_rtd_theme',       # ReadTheDocs theme
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
