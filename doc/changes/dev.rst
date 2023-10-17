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

.. rubric:: PetscSection:

.. rubric:: PetscPartitioner:

.. rubric:: Mat:

.. rubric:: MatCoarsen:

.. rubric:: PC:

- Add ``PCGAMGSetLowMemoryFilter()`` with corresponding option ``-pc_gamg_low_memory_threshold_filter``. Use the system ``MatFilter`` graph/matrix filter, without a temporary copy of the graph, otherwise use method that can be faster

.. rubric:: KSP:

.. rubric:: SNES:

.. rubric:: SNESLineSearch:

.. rubric:: TS:

.. rubric:: TAO:

- Add in petsc4py an experimental class ``torchbridge.TAOtorch`` to mimic PyTorch.Optimizer objects by using customized TAO optimizers. The feature is demonstrated on a canonical MNIST classification problem.

.. rubric:: DM/DA:

.. rubric:: DMSwarm:

.. rubric:: DMPlex:

.. rubric:: FE/FV:

.. rubric:: DMNetwork:

.. rubric:: DMStag:

.. rubric:: DT:

.. rubric:: Fortran:
