########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import json
import typing

import numpy as np

from datetime import date

import decontamination.algo.selection as selection

########################################################################################################################

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decontamination', 'metadata.json')), 'r') as f:

    metadata = json.load(f)

########################################################################################################################

project = metadata['name']
release = metadata['version']

author = ', '.join(metadata['author_names'])

copyright = f'2023-{date.today().year}, {metadata["credits"]}'

########################################################################################################################

extensions = [
    'numpydoc',
    #'sphinx_docsearch',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinxawesome_theme',
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

########################################################################################################################

exclude_patterns = ['_build', '.DS_Store', 'Thumbs.db']

########################################################################################################################

html_js_files = ['custom.js']

html_css_files = ['custom.css']

########################################################################################################################

templates_path = ['_templates']

html_static_path = ['_html_static']

########################################################################################################################

html_theme = 'sphinxawesome_theme'

html_permalinks_icon = ''

html_sidebars = {'**': [
    'logo.html',
    'globaltoc.html',
    'sonar.html',
]}

########################################################################################################################

def skip_member(app, what, name, obj, skip, options):

    doc = getattr(obj, '__doc__', None)

    if not doc or ':private:' in doc:

        return True

    return skip

########################################################################################################################

# noinspection PyUnusedLocal
def before_process_signature(app, obj, bound_method):

    numeric_type = typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]

    numeric_value = typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]

    ast_node = typing.Union[
        selection.Selection.UnaryOpNode,
        selection.Selection.BinaryOpNode,
        selection.Selection.FloatNumNode,
        selection.Selection.IntNumNode,
        selection.Selection.IntNumNode,
        selection.Selection.ColNameNode,
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

    app.connect('autodoc-skip-member', skip_member)

    app.connect('autodoc-before-process-signature', before_process_signature)

    app.connect('autodoc-process-signature', process_signature)

    app.connect('autodoc-process-docstring', process_docstring)

########################################################################################################################
