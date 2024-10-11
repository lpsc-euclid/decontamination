.. raw:: html

    <div style="display: flex !important; justify-content: space-around; margin-left: auto; margin-right: auto; width: 80%;">

.. image:: _html_static/logo_lpsc.svg
  :alt: LPSC
  :height: 72
  :target: http://lpsc.in2p3.fr/

.. image:: _html_static/logo_in2p3.svg
  :alt: IN2P3
  :height: 72
  :target: http://www.in2p3.fr/

.. image:: _html_static/logo_uga.svg
  :alt: UGA
  :height: 72
  :target: http://www.univ-grenoble-alpes.fr/

.. raw:: html

   </div>

    <div style="display: flex !important;">
        <a href="http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.txt">
            <img src="https://img.shields.io/badge/license-CeCILL--C-blue.svg" />
        </a>
        &nbsp;
        <a href="https://gitlab.in2p3.fr/lpsc-euclid/decontamination">
            <img src="https://img.shields.io/badge/Main%20Repo-gitlab.in2p3.fr-success" />
        </a>
        &nbsp;
        <a href="https://github.com/lpsc-euclid/decontamination">
            <img src="https://img.shields.io/badge/Alt%20Repo-github.com-success" />
        </a>
        &nbsp;
        <a href="https://sonarqube.in2p3.fr/dashboard?id=decontamination-key">
            <img src="https://sonarqube.in2p3.fr/api/project_badges/measure?project=decontamination-key&metric=coverage&token=sqb_70baaf7c87542fe8555d5bd23fdb95bfaf848b37" crossorigin="anonymous" />
        </a>
    </div>

Decontamination
===============

This software is a toolbox for cosmology analysis with three main purposes: constructing HEALPix masks of the
systematics that affect galaxy detection, including survey properties, instrumental performance, and sky properties;
simulating random galaxy catalogs; and providing multiple methods to compute galaxy-galaxy 2-point correlation functions
(2PCF) decontaminated from the effects of systematics.

Authors:
 * Jérôme ODIER - jerome.odier@lpsc.in2p3.fr
 * Gaël ALGUERO - gael.alguero@lpsc.in2p3.fr
 * Juan MACIAS-PEREZ - juan.macias-perez@lpsc.in2p3.fr

Installing
==========

Installing the last development version from git:

.. code-block:: bash

    pip install git+https://gitlab.in2p3.fr/lpsc-euclid/decontamination.git

or alternatively:

.. code-block:: bash

    pip install git+https://github.com/lpsc-euclid/decontamination.git

Dependencies
============

Mandatory dependencies:

* `h5py <https://www.h5py.org/>`_
* `tqdm <https://tqdm.github.io/>`_
* `numpy <https://numpy.org/>`_
* `numba <https://numba.pydata.org/>`_
* `scipy <https://scipy.org/>`_
* `healpy <https://healpy.readthedocs.io/>`_
* `astropy <https://www.astropy.org/>`_
* `matplotlib <https://matplotlib.org/>`_

Optional dependencies:

* `treecorr <https://rmjarvis.github.io/TreeCorr/_build/html/index.html>`_

Indices and modules
===================

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   Module JIT <module_jit>
   Module HP <module_hp>
   Module Mask <module_mask>
   Module Algorithm <module_algo>
   Module Plotting <module_plotting>
   Module Generator <module_generator>
   Module Correlation <module_correlation>
   Module Decontamination <module_decontamination>

:ref:`genindex`

:ref:`modindex`
