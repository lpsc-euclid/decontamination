########################################################################################################################

import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decontamination', 'metadata.json')), 'r') as f:

    metadata = json.load(f)

########################################################################################################################

project = metadata['name']
release = metadata['version']

author = ', '.join(metadata['author_names'])

copyright = '2023, ' + metadata['credits']

########################################################################################################################

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'docstring': 'class',
    'undoc-members': False,
    'private-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js'

########################################################################################################################

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

########################################################################################################################

html_theme = 'cloud'

html_logo = '_static/logo.png'
html_css_files = ['custom.css']
html_static_path = ['_static']

########################################################################################################################
