cdef extern from * nogil:

    ctypedef const char* PetscRegressorType
    PetscRegressorType PETSCREGRESSORLINEAR
    PetscRegressorType PETSCREGRESSORPYTHON

    PetscErrorCode PetscRegressorCreate(MPI_Comm,PetscRegressor*)
    PetscErrorCode PetscRegressorReset(PetscRegressor)
    PetscErrorCode PetscRegressorDestroy(PetscRegressor*)
    PetscErrorCode PetscRegressorSetType(PetscRegressor,PetscRegressorType)
    PetscErrorCode PetscRegressorSetUp(PetscRegressor)
    PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor)
    PetscErrorCode PetscRegressorView(PetscRegressor,PetscViewer)
    PetscErrorCode PetscRegressorFit(PetscRegressor,PetscMat,PetscVec)
    PetscErrorCode PetscRegressorPredict(PetscRegressor,PetscMat,PetscVec)
