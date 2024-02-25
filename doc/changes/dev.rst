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

.. rubric:: Event Logging:

.. rubric:: PetscViewer:

.. rubric:: PetscDraw:

.. rubric:: AO:

.. rubric:: IS:

.. rubric:: VecScatter / PetscSF:

.. rubric:: PF:

.. rubric:: Vec:

- Add ``VecGhostGetGhostIS()`` to get the ghost indices of a ghosted vector
- Add ``-vec_mdot_use_gemv`` to let ``VecMDot()``, ``VecMTDot()``  use BLAS2 ``gemv()`` instead of custom unrolled kernel. Default is on
- Add ``-vec_maxpy_use_gemv`` to let ``VecMAXPY()`` use BLAS2 ``gemv()`` instead of custom unrolled kernel. Default is off
- ``VecReplaceArray()`` on the first Vec obtained from ``VecDuplicateVecs()`` with either of the two above \*_use_gemv options won't work anymore. If needed, turn them off or use ``VecDuplicateVec()`` instead
- ``VecScale()`` is now a logically collective operation
- Add ``VecISShift()`` to shift a part of the vector
- ``VecISSet()`` does no longer accept NULL as index set
- Add ``VecLocalFormGetRead()``, ``VecLocalFormRestoreRead()``, ``VecLocalFormGetWrite()``, ``VecLocalFormRestoreWrite()``, ``VecLocalFormSetVec()``, ``VecLocalFormSetIS()``, ``VecLocalFormSetUpdateRead()``, and ``VecLocalFormSetUpdateWrite()``
- Deprecate ``VecGhostGetGhostIS()``, ``VecGhostGetLocalForm()``, ``VecGhostRestoreLocalForm()``, ``VecGhostIsLocalForm()``, ``VecGhostUpdateBegin()``, ``VecGhostUpdateEnd()``, ``VecCreateGhost()``, ``VecCreateGhostWithArray()``, ``VecCreateGhostBlock()``, and ``VecCreateGhostBlockWithArray()``

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add support in ``PCFieldSplitSetFields()`` including with ``-pc_fieldsplit_%d_fields fields`` for ``MATNEST``,  making it possible to
  utilize multiple levels of ``PCFIELDSPLIT`` with ``MATNEST`` from the command line
- Add ``PCCompositeSpecialSetAlphaMat()`` API to use a matrix other than the identity in
  preconditioners based on an alternating direction iteration, e.g., setting :math:`M` for
  :math:`P = (A + alpha M) M^{-1} (alpha M + B)`

- Change the option database keys for coarsening for ``PCGAMG`` to use the prefix ``-pc_gamg_``, for example ``-pc_gamg_mat_coarsen_type``

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

- Add Rosenbrock-W methods from :cite:`rang2015improved` with :math:`B_{PR}` stability: ``TSROSWR34PRW``, ``TSROSWR3PRL2``, ``TSROSWRODASPR``, and ``TSROSWRODASPR2``

.. rubric:: TAO:

.. rubric:: DM/DA:

- Add ``DMGetSparseLocalize()`` and ``DMSetSparseLocalize()``
- Add ``DMGeomModelRegister()``, ``DMGeomModelRegisterAll()``, ``DMGeomModelRegisterDestroy()``, ``DMSnapToGeomModel()``, ``DMSetSnapToGeomModel()`` to support registering geometric models
- Add ``DMGetOutputSequenceLength()``

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

- Add ``DMLabelGetValueBounds()``
- Add ``DMPlexOrientLabel()``
- Add an argument to ``DMPlexLabelCohesiveComplete()`` in order to change behavior at surface boundary
- Remove ``DMPlexSnapToGeomModel()``
- Add refinement argument to ``DMPlexCreateHexCylinderMesh()``
- Now ``DMPlexComputeBdIntegral()`` takes one function per field
- Add ``DMPlexCreateEdgeNumbering()``

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
