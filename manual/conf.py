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

html_theme_options = {
    'extra_header_link_icons': {
        'repository on GitLab': {
            'link': 'https://gitlab.in2p3.fr/lpsc-euclid/decontamination',
            'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-gitlab" viewBox="0 0 16 16"><path d="m15.734 6.1-.022-.058L13.534.358a.57.57 0 0 0-.563-.356.6.6 0 0 0-.328.122.6.6 0 0 0-.193.294l-1.47 4.499H5.025l-1.47-4.5A.572.572 0 0 0 2.47.358L.289 6.04l-.022.057A4.044 4.044 0 0 0 1.61 10.77l.007.006.02.014 3.318 2.485 1.64 1.242 1 .755a.67.67 0 0 0 .814 0l1-.755 1.64-1.242 3.338-2.5.009-.007a4.05 4.05 0 0 0 1.34-4.668Z"/></svg>',
        },
        'repository on GitHub': {
            'link': 'https://github.com/lpsc-euclid/decontamination',
            'icon': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>',
        },
    },
}

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

    ast_node1 = 'UnaryOpNode | BinaryOpNode | FloatNumNode | IntNumNode | ColNameNode'
    ast_node2 = 'Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]'

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
