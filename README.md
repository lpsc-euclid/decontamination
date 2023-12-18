Decontamination
===============

[![][License img]][License]
&nbsp;
[![][MainRepo img]][MainRepo]
&nbsp;
[![][AltRepo img]][AltRepo]
&nbsp;
[![][CodeCoverage img]][CodeCoverage]
&nbsp;
[![][CodeQuality img]][CodeQuality]
&nbsp;
[![][CodeLines img]][CodeLines]

<a href="http://lpsc.in2p3.fr/"              target="_blank"><img src="./doc/_html_static/logo_lpsc.svg" alt="LPSC" height="72" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://www.in2p3.fr/"               target="_blank"><img src="./doc/_html_static/logo_in2p3.svg" alt="IN2P3" height="72" /></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://www.univ-grenoble-alpes.fr/" target="_blank"><img src="./doc/_html_static/logo_uga.svg" alt="UGA" height="72" /></a>

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

or alternatively:

```bash
pip install git+https://github.com/lpsc-euclid/decontamination.git
```

Dependencies
------------

* [h5py](https://www.h5py.org/)
* [tqdm](https://tqdm.github.io/)
* [numpy](https://numpy.org/)
* [numba](https://numba.pydata.org/)
* [scipy](https://scipy.org/)
* [healpy](https://healpy.readthedocs.io/)
* [matplotlib](https://matplotlib.org/)

[License]:http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.txt
[License img]:https://img.shields.io/badge/license-CeCILL--C-blue.svg

[MainRepo]:https://gitlab.in2p3.fr/lpsc-euclid/decontamination
[MainRepo img]:https://img.shields.io/badge/private%20repo-gitlab.in2p3.fr-success

[AltRepo]:https://github.com/lpsc-euclid/decontamination
[AltRepo img]:https://img.shields.io/badge/public%20repo-github.com-success

[CodeCoverage]:https://sonarqube.in2p3.fr/dashboard?id=decontamination-key
[CodeCoverage img]:https://sonarqube.in2p3.fr/api/project_badges/measure?project=decontamination-key&metric=coverage&token=sqb_70baaf7c87542fe8555d5bd23fdb95bfaf848b37

[CodeQuality]:https://sonarqube.in2p3.fr/dashboard?id=decontamination-key
[CodeQuality img]:https://sonarqube.in2p3.fr/api/project_badges/measure?project=decontamination-key&metric=alert_status&token=sqb_70baaf7c87542fe8555d5bd23fdb95bfaf848b37

[CodeLines]:https://sonarqube.in2p3.fr/dashboard?id=decontamination-key
[CodeLines img]:https://sonarqube.in2p3.fr/api/project_badges/measure?project=decontamination-key&metric=ncloc&token=sqb_70baaf7c87542fe8555d5bd23fdb95bfaf848b37
