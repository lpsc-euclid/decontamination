#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import json

from setuptools import setup

########################################################################################################################

if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md'), 'r') as f1:

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'decontamination', 'metadata.json'), 'r') as f2:

            readme = f1.read()
            metadata = json.load(f2)

            setup(
                name = 'decontamination',
                version = metadata['version'],
                author = ', '.join(metadata['author_names']),
                author_email = ', '.join(metadata['author_emails']),
                description = 'A toolbox for performing systematics decontamination in cosmology analyses.',
                long_description = readme,
                long_description_content_type = 'text/markdown',
                keywords = [],
                url = '???',
                license = 'CeCILL-C',
                packages = ['decontamination'],
                data_files = [('decontamination', ['decontamination/metadata.json'])],
                include_package_data = True,
                install_requires = ['h5py', 'tqdm', 'numpy', 'numba'],
                extras_require = {
                    'astropy': ['astropy'],
                    'matplotlib': ['matplotlib'],
                }
            )

########################################################################################################################
