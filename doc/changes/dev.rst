====================
Changes: Development
====================

..
   STYLE GUIDELINES:
   * Capitalize sentences
   * Use imperative, e.g., Add, Improve, Change, etc.
   * Don't use a period (.) at the end of entries
   * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

.. rubric:: General:

.. rubric:: Configure/Build:

.. rubric:: Sys:

- Add `PetscDeviceContextGetStreamHandle()` to return a handle to the stream the current device context is using

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``VecCreateMatDense()``
- Remove ``MATSOLVERSPARSEELEMENTAL`` since it is no longer functional
- Add MATSELLCUDA. It supports fast ``MatMult()``, ``MatMultTranspose()`` and ``MatMultAdd()`` on GPUs
- Add support for ``MAT_FACTOR_LU`` and ``MAT_FACTOR_CHOLESKY`` with ``MATSOLVERMUMPS`` for ``MATNEST``
- ``MatGetFactor()`` can now return ``NULL`` for some combinations of matrices and solvers types. This is to support those combinations that can only be inspected at runtime (i.e. MatNest with AIJ blocks vs MatNest with SHELL blocks).
- Remove ``MatSetValuesDevice()``, ``MatCUSPARSEGetDeviceMatWrite()``, ``MatKokkosGetDeviceMatWrite``
- Add ``MAT_TRIANGULAR_STORAGE_HERMITIAN`` to be used to describe the behavior of ``MATSBAIJ`` so that ``MAT_HERMITIAN`` can be purely descriptive.

.. rubric:: MatCoarsen:

.. rubric:: PC:

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

- Add ``DMNetworkViewSetShowRanks()``, ``DMNetworkViewSetViewRanks()``, ``DMNetworkViewSetShowGlobal()``, ``DMNetworkViewSetShowVertices()``, ``DMNetworkViewSetShowNumbering()``

- Add ``-dmnetwork_view_all_ranks`` ``-dmnetwork_view_rank_range`` ``-dmnetwork_view_no_vertices`` ``-dmnetwork_view_no_numbering`` for viewing DMNetworks with the Matplotlib viewer

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
