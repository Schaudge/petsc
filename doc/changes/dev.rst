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

- Add ``PetscOptionsBegin()``, ``PetscOptionsEnd()``, ``PetscOptionsInt()``, ``PetscOptionsBool()``, ``PetscOptionsIntArray()``,
  ``PetscOptionsReal()``, ``PetscOptionsRealArray()``, and ``PetscOptionsScalar()`` for Fortran
- Add ``PetscAssertPointer()`` as a replacement for ``PetscValidPointer()``, ``PetscValidCharPointer()``, ``PetscValidIntPointer()``, ``PetscValidInt64Pointer()``, ``PetscValidCountPointer()``, ``PetscValidBoolPointer()``, ``PetscvalidScalarPointer()``, and ``PetscValidRealPointer()``

.. rubric:: Configure/Build:

- Add support for external-packages to prefer tarball download for regular use - as currently all packages  prefer git clones. MPICH is a package using this feature. Here MPICH tarball will be downloaded for regular use. However on providing ``--download-mpich-commit=main`` option - configure download and builds from MPICH git repository

.. rubric:: Sys:

- Add ``PetscDeviceContextGetStreamHandle()`` to return a handle to the stream the current device context is using
- Add ``PetscStrcmpAny()`` to compare against multiple non-empty strings
- Change arguments 4 and 5 of ``PetscViewerBinaryReadAll()`` and ``PetscViewerBinaryWriteAll()`` to ``PetscInt64``
- Add ``PetscIsCloseAtTolScalar()``
- Remove ``PetscTellMyCell()`` and related runtime option
- Remove ``PetscTextBelt()`` and related runtime option
- Remove deprecated ``-malloc [no]`` startup option
- Remove deprecated ``-malloc_info`` startup option
- Remove deprecated ``-log_summary`` option
- Remove ``PetscURLShorten()``, it has not worked since 2019

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

- Add ``PetscViewerASCIIOpenWithFileUnit()`` and ``PetscViewerASCIISetFileUnit()``

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecErrorWeightedNorms()`` to unify weighted local truncation error norms used in ``TS``
- Add CUDA/HIP implementations for ``VecAbs()``, ``VecSqrt()``, ``VecExp()``, ``VecLog()``, ``VecPointwiseMax()``, ``VecPointwiseMaxAbs()``, and ``VecPointwiseMin()``
- Add ``VecMAXPBY()``

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

- Add ``MatCreateDenseFromVecType()``
- Add support for calling ``MatDuplicate()`` on a matirx preallocated via ``MatSetPreallocationCOO()``, and then ``MatSetValuesCOO()`` on the new matrix
- Remove ``MATSOLVERSPARSEELEMENTAL`` since it is no longer functional
- Add MATSELLCUDA. It supports fast ``MatMult()``, ``MatMultTranspose()`` and ``MatMultAdd()`` on GPUs
- Add support for ``MAT_FACTOR_LU`` and ``MAT_FACTOR_CHOLESKY`` with ``MATSOLVERMUMPS`` for ``MATNEST``
- ``MatGetFactor()`` can now return ``NULL`` for some combinations of matrices and solvers types. This is to support those combinations that can only be inspected at runtime (i.e. MatNest with AIJ blocks vs MatNest with SHELL blocks).
- Remove ``MatSetValuesDevice()``, ``MatCUSPARSEGetDeviceMatWrite()``, ``MatKokkosGetDeviceMatWrite``
- Add ``MatDenseCUDASetPreallocation()`` and ``MatDenseHIPSetPreallocation()``
- Add support for KOKKOS in ``MATH2OPUS``
- Add ``-pc_precision single`` option for use with ``MATSOLVERSUPERLU_DIST``
- Add ``MATDIAGONAL`` which can be created with ``MatCreateDiagonal()``
- Add ``MatDiagonalGetDiagonal()``, ``MatDiagonalRestoreDiagonal()``, ``MatDiagonalGetInverseDiagonal()``, and ``MatDiagonalRestoreInverseDiagonal()``
- Add support for ``MatLoad()`` and ``MatView()`` to load and store ``MPIAIJ`` matrices that have more than ``PETSC_INT_MAX`` nonzeros, so long as each rank has fewer than ``PETSC_INT_MAX``
- Add ``MatLRCSetMats()`` and register creation routine for ``MatLRC``
- Add CUDA/HIP implementation for ``MatGetDiagonal()``
- Add a Boolean parameter to ``MatEliminateZeros()`` to force the removal of zero diagonal coefficients
- Expose ``MatComputeVariableBlockEnvelope()`` in public headers
- Add ``MatEliminateZeros()`` implementations for ``MatBAIJ`` and ``MatSBAIJ``

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add ``PCMatGetApplyOperation()`` and ``PCMatSetApplyOperation()``
- Add ``PCReduceFailedReason()``
- Add ``PCIsSymmetric()``
- Add ``PCSetUseSymmetricForm()`` for use by `KSPCG` and friends to ensure the preconditioner is symmetric

.. rubric:: KSP:

- Add ``KSPSetMinimumIterations()`` and ``KSPGetMinimumIterations()``

.. rubric:: SNES:

- Add a convenient, developer-level ``SNESConverged()`` function that runs the convergence test and updates the internal converged reason.
- Swap the order of monitor and convergence test. Now monitors are always called after a convergence test.
- Deprecate option ``-snes_ms_norms``. Use ``-snes_norm_schedule always``.

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Remove ``TSErrorWeightedNormInfinity()``, ``TSErrorWeightedNorm2()``, ``TSErrorWeightedENormInfinity()``, ``TSErrorWeightedENorm2()`` since the same functionality can be obtained with ``VecErrorWeightedNorms()``
- Add support for time-dependent solvers with varying solution size using ``TSSetResize()``

.. rubric:: TAO:

- Add ``TaoADMMGetRegularizerCoefficient()``
- Add ``TAOBNCG``, ``TaoBNCGGetType()`` and ``TaoBNCGSetType()``

.. rubric:: DM/DA:

- Add support for ``DMDAGetElements()`` for Fortran
- Add support for clearing named vectors with ``DMClearNamedGlobalVectors()`` and ``DMClearNamedLocalVectors()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMPlexTransformExtrudeGetPeriodic()`` and ``DMPlexTransformExtrudeSetPeriodic()``
- Replace ``DMPlexGetGhostCellStratum()`` with ``DMPlexGetCellTypeStratum()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

- Add ``DMNetworkViewSetShowRanks()``, ``DMNetworkViewSetViewRanks()``, ``DMNetworkViewSetShowGlobal()``, ``DMNetworkViewSetShowVertices()``, ``DMNetworkViewSetShowNumbering()``

- Add ``-dmnetwork_view_all_ranks`` ``-dmnetwork_view_rank_range`` ``-dmnetwork_view_no_vertices`` ``-dmnetwork_view_no_numbering`` for viewing DMNetworks with the Matplotlib viewer

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:

- Add ``PetscCheck()`` and ``PetscCheckA()`` for Fortran
- Change ``PETSC_HAVE_FORTRAN`` to ``PETSC_USE_FORTRAN_BINDINGS`` to indicate if PETSc is built with Fortran bindings
