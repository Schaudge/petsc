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

- Add ``PetscCtxDestroyFn`` as the prototype for all context destroy functions. It is ``PetscErrorCode ()(void **)``. Previously some context destructor
  setters took ``PetscErrorCode ()(void *)``. But these would not work directly with PETSc objects as contexts and having two different
  context destructor models added unneeded complexity to the library. This change is not backward compatible
- Deprecate ``PetscContainerSetUserDestroy()`` with ``PetscContainerSetCtxDestroy()``, updating will require a small change in calling code
- Deprecate ``PetscContainerCtxDestroyDefault`` with ``PetscCtxDestroyDefault()``

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

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

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

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
