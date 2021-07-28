cdef extern from * nogil:

    ctypedef const char* PetscMLRegressorType "MLRegressorType"
    PetscMLRegressorType MLREGRESSORLINEAR

    int MLRegressorCreate(MPI_Comm,PetscMLRegressor*)
    int MLRegressorReset(PetscMLRegressor)
    int MLRegressorDestroy(PetscMLRegressor*)
    int MLRegressorSetType(PetscMLRegressor,PetscMLRegressorType)
    int MLRegressorSetUp(PetscMLRegressor)
    int MLRegressorSetFromOptions(PetscMLRegressor)
    int MLRegressorView(PetscMLRegressor,PetscViewer)
    int MLRegressorFit(PetscMLRegressor,PetscMat,PetscVec)
    int MLRegressorPredict(PetscMLRegressor,PetscMat,PetscVec)
