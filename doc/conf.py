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
    'show-inheritance': True,
    'member-order': 'bysource',
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
    'externalicon': False,
}

html_sidebars = {'**': ['logo.html', 'localtoc.html', 'globaltoc.html', 'searchbox.html']}

########################################################################################################################

html_use_modindex = False

########################################################################################################################

# noinspection PyUnusedLocal
def skip_undocumented_classes_and_functions(app, what, name, obj, skip, options):

    if not name.startswith('_') and getattr(obj, '__doc__', ''):

        return False

    return True

########################################################################################################################

# noinspection PyUnusedLocal
def process_signature(app, what, name, obj, options, signature, return_annotation):

    numeric_type = '~Type[~np.float32 | ~np.float64 | float | ~np.int32 | ~np.int64 | int]'

    numeric_value = '~np.float32 | ~np.float64 | float | ~np.int32 | ~np.int64 | int'

    if signature:

        signature = signature.replace('typing.', '').replace('numpy.', 'np.').replace(numeric_type, '<numeric type>').replace(numeric_value, '<numeric value>')

    if return_annotation:

        return_annotation = return_annotation.replace('typing.', '').replace('numpy.', 'np.').replace(numeric_type, '<numeric type>').replace(numeric_value, '<numeric value>')

    return signature, return_annotation

########################################################################################################################

# noinspection PyUnusedLocal
def process_docstring(app, what, name, obj, options, lines):

    numeric_type1 = 'Type[np.float32 | np.float64 | float | np.int32 | np.int64 | int]'
    numeric_type2 = 'Type[Union[np.float32, np.float64, float, np.int32, np.int64, int]]'

    numeric_value1 = 'np.float32 | np.float64 | float | np.int32 | np.int64 | int'
    numeric_value2 = 'Union[np.float32, np.float64, float, np.int32, np.int64, int]'

    for index, line in enumerate(lines):

        lines[index] = (
            line
            .replace('typing.', '')
            .replace('numpy.', 'np.')
            .replace(numeric_type1, '<numeric type>')
            .replace(numeric_type2, '<numeric type>')
            .replace(numeric_value1, '<numeric value>')
            .replace(numeric_value2, '<numeric value>')
        )

########################################################################################################################

def setup(app):

    app.connect('autodoc-skip-member', skip_undocumented_classes_and_functions)

    app.connect("autodoc-process-signature", process_signature)

    app.connect("autodoc-process-docstring", process_docstring)

########################################################################################################################
