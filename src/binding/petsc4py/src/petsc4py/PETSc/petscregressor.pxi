cdef extern from * nogil:

    ctypedef const char* PPetscRegressorType "PetscRegressorType"
    PPetscRegressorType PETSCREGRESSORLINEAR

    PetscErrorCode PetscRegressorCreate(MPI_Comm,PPetscRegressor*)
    PetscErrorCode PetscRegressorReset(PPetscRegressor)
    PetscErrorCode PetscRegressorDestroy(PPetscRegressor*)
    PetscErrorCode PetscRegressorSetType(PPetscRegressor,PPetscRegressorType)
    PetscErrorCode PetscRegressorSetUp(PPetscRegressor)
    PetscErrorCode PetscRegressorSetFromOptions(PPetscRegressor)
    PetscErrorCode PetscRegressorView(PPetscRegressor,PetscViewer)
    PetscErrorCode PetscRegressorFit(PPetscRegressor,PetscMat,PetscVec)
    PetscErrorCode PetscRegressorPredict(PPetscRegressor,PetscMat,PetscVec)
