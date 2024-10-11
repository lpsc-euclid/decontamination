Module decontamination.algo
===========================

This module provides a set of machine learning algorithms (Self-Organizing Maps (SOM), linear regressions, ...) designed to run at constant memory usage on both CPU and GPU.

Self-Organizing Maps
--------------------

Self Organizing Maps (SOMs) are neural networks that transform high-dimensional data into spatially organized low-dimensional clusters.

.. image:: _html_static/som.png
    :alt: Self Organizing Maps
    :width: 50%
    :align: center

.. automodule:: decontamination.algo.som_abstract
   :members:

.. automodule:: decontamination.algo.som_pca
   :members:

.. automodule:: decontamination.algo.som_online
   :members:

.. automodule:: decontamination.algo.som_batch
   :members:

Clustering
----------

.. automodule:: decontamination.algo.clustering
   :members:

Data selection
--------------

.. automodule:: decontamination.algo.selection
   :members:
