########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import json
import typing
import inspect

import numpy as np

import decontamination.algo.selection as selection

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
    'member-order': 'bysource',
    #
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'show-inheritance': True,
    'inherited-members': True,
}

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-svg.min.js'

########################################################################################################################

exclude_patterns = ['_build', '.DS_Store', 'Thumbs.db']

########################################################################################################################

html_js_files = ['custom.js']

html_css_files = ['custom.css']

########################################################################################################################

templates_path = ['_templates']

html_static_path = ['_html_static']

########################################################################################################################

html_theme = 'cloud'

html_theme_options = {
    'max_width': '13in',
    'externalicon': False,
    'borderless_decor': True,
}

html_sidebars = {'**': ['logo.html', 'localtoc.html', 'globaltoc.html', 'searchbox.html', 'sonar.html']}

########################################################################################################################

html_use_modindex = False

########################################################################################################################

# noinspection PyUnusedLocal
def before_process_signature(app, obj, bound_method):

    numeric_type = typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]

    numeric_value = typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]

    ast_node = typing.Union[
        selection.Selection.UnaryOpNode,
        selection.Selection.BinaryOpNode,
        selection.Selection.NumberNode,
        selection.Selection.ColumnNode,
    ]

    if callable(obj):

        for param, annotation in obj.__annotations__.items():

            if annotation == numeric_type:
                obj.__annotations__[param] = '<numeric type>'
            if annotation == numeric_value:
                obj.__annotations__[param] = '<numeric value>'
            if annotation == ast_node:
                obj.__annotations__[param] = '<ast node>'

        if 'return' in obj.__annotations__:

            if obj.__annotations__['return'] == numeric_type:
                obj.__annotations__['return'] = '<numeric type>'
            if obj.__annotations__['return'] == numeric_value:
                obj.__annotations__['return'] = '<numeric value>'
            if obj.__annotations__['return'] == ast_node:
                obj.__annotations__['return'] = '<ast node>'

########################################################################################################################

# noinspection PyUnusedLocal
def process_signature(app, what, name, obj, options, signature, return_annotation):

    if signature:

        signature = signature.replace('typing.', '').replace('numpy.', 'np.')

    if return_annotation:

        return_annotation = return_annotation.replace('typing.', '').replace('numpy.', 'np.')

    return signature, return_annotation

########################################################################################################################

# noinspection PyUnusedLocal
def process_docstring(app, what, name, obj, options, lines):

    ####################################################################################################################

    options['inherited-members'] = getattr(obj, '__module__', '') not in [
        'decontamination.mask.wcs'
    ]

    ####################################################################################################################

    numeric_type1 = 'Type[np.float32 | np.float64 | float | np.int32 | np.int64 | int]'
    numeric_type2 = 'Type[Union[np.float32, np.float64, float, np.int32, np.int64, int]]'

    numeric_value1 = 'np.float32 | np.float64 | float | np.int32 | np.int64 | int'
    numeric_value2 = 'Union[np.float32, np.float64, float, np.int32, np.int64, int]'

    ast_node1 = 'UnaryOpNode | BinaryOpNode | NumberNode | ColumnNode'
    ast_node2 = 'Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]'

    for index, line in enumerate(lines):

        lines[index] = (
            line
            .replace('typing.', '')
            .replace('numpy.', 'np.')
            .replace(numeric_type1, '<numeric type>')
            .replace(numeric_type2, '<numeric type>')
            .replace(numeric_value1, '<numeric value>')
            .replace(numeric_value2, '<numeric value>')
            .replace(ast_node1, '<ast node>')
            .replace(ast_node2, '<ast node>')
        )

########################################################################################################################

def setup(app):

    app.connect('autodoc-before-process-signature', before_process_signature)

    app.connect('autodoc-process-signature', process_signature)

    app.connect('autodoc-process-docstring', process_docstring)

########################################################################################################################
