Decontamination
===============

[![][License img]][License]
&nbsp;
[![][MainRepo img]][MainRepo]
<!--
&nbsp;
[![][AltRepo img]][AltRepo]
-->

<a href="http://lpsc.in2p3.fr/"              target="_blank"><img src="./doc/_static/logo_lpsc.svg" alt="LPSC" height="72" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.ijclab.in2p3.fr/"       target="_blank"><img src="./doc/_static/logo_ijclab.svg" alt="IJCLab" height="72" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://www.in2p3.fr/"               target="_blank"><img src="./doc/_static/logo_in2p3.svg" alt="IN2P3" height="72" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://www.univ-grenoble-alpes.fr/" target="_blank"><img src="./doc/_static/logo_uga.svg" alt="UGA" height="72" /></a>

**Decontamination** is a high-performance toolbox, that runs on both CPU and GPU, for performing systematics decontamination in cosmology analyses. See the full documentation [here]().

Installation
------------

Installing the last version:

```bash
pip install decontamination
```

Installing the last development version from git:

```bash
pip install git+https://gitlab.in2p3.fr/lpsc-euclid/decontamination.git
```

Dependencies
------------

* [h5py](https://www.h5py.org/)
* [tqdm](https://tqdm.github.io/)
* [numpy](https://numpy.org/)
* [numba](https://numba.pydata.org/)
* [healpy](https://healpy.readthedocs.io/)
* [matplotlib](https://matplotlib.org/)

[License]:http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.txt
[License img]:https://img.shields.io/badge/license-CeCILL--C-blue.svg

[MainRepo]:https://gitlab.in2p3.fr/lpsc-euclid/decontamination
[MainRepo img]:https://img.shields.io/badge/Main%20Repo-gitlab.in2p3.fr-success

[AltRepo]:https://github.com/odier-xyz/decontamination
[AltRepo img]:https://img.shields.io/badge/Alt%20Repo-github.com-success
