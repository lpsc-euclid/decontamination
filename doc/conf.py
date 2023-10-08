########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

project = 'Decontamination'
copyright = '2023, IN2P3 / CNRS'
author = 'Gaël ALGUERO, Juan MACIAS-PEREZ, Jérôme ODIER, Martìn RODRIGUEZ-MONROY'
release = '1.0.0'

########################################################################################################################

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
]

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

########################################################################################################################

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

########################################################################################################################

#html_theme = 'furo'
html_theme = 'cloud'
#html_theme = 'sphinx_rtd_theme'

html_css_files = ['custom.css']
html_static_path = ['_static']

########################################################################################################################
