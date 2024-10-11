Module decontamination.jit
==========================

This module provides an abstraction layer for Numba, enabling a single code to run on both CPU and GPU.

* | *Numba*
  | Open source JIT compiler
    (`documentation <https://numba.readthedocs.io/en/stable/>`_)

.. automodule:: decontamination.jit
   :members:

Example (new syntax)
--------------------

.. code-block:: python

    @jit()
    def foo_xpu(a, b):

        return a + b

    @jit(kernel = True, parallel = True)
    def foo_kernel(result, a, b):

        if jit.is_gpu:

            ####################################################################
            # GPU CODE                                                         #
            ####################################################################

            i = jit.grid(1)
            if i < result.shape[0]:

                result[i] = foo_xpu(a[i], b[i])

            ####################################################################

        else:

            ####################################################################
            # CPU CODE                                                         #
            ####################################################################

            for i in nb.prange(result.shape[0]):

                result[i] = foo_xpu(a[i], b[i])

            ####################################################################

    use_gpu = True
    threads_per_block = 32

    A = np.random.randn(100_000).astype(np.float32)
    B = np.random.randn(100_000).astype(np.float32)

    result = device_array_empty(100_000, dtype = np.float32)

    foo_kernel[use_gpu, threads_per_block, result.shape[0]](result, A, B)

    print(result.copy_to_host())

Example (alt. syntax)
---------------------

.. code-block:: python

    @jit()
    def foo_xpu(a, b):

        return a + b

    @jit(kernel = True, parallel = True)
    def foo_kernel(result, a, b):

        ########################################################################
        # !--BEGIN-GPU--

        i = jit.grid(1)
        if i < result.shape[0]:

            result[i] = foo_xpu(a[i], b[i])

        # !--END-GPU--
        ########################################################################
        # !--BEGIN-CPU--

        for i in nb.prange(result.shape[0]):

            result[i] = foo_xpu(a[i], b[i])

        # !--END-CPU--
        ########################################################################

    use_gpu = True
    threads_per_block = 32

    A = np.random.randn(100_000).astype(np.float32)
    B = np.random.randn(100_000).astype(np.float32)

    result = device_array_empty(100_000, dtype = np.float32)

    foo_kernel[use_gpu, threads_per_block, result.shape[0]](result, A, B)

    print(result.copy_to_host())
