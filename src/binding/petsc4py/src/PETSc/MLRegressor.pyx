class MLRegressorType(object):
    LINEAR = S_(MLREGRESSORLINEAR)

cdef class MLRegressor(Object):

    Type = MLRegressorType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.mlregressor
        self.mlregressor = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( MLRegressorView(self.mlregressor, cviewer) )

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscMLRegressor newmlregressor = NULL
        CHKERR( MLRegressorCreate(ccomm, &newmlregressor) )
        PetscCLEAR(self.obj); self.mlregressor = newmlregressor
        return self

    def setUp(self):
        CHKERR( MLRegressorSetUp(self.mlregressor) )

    def fit(self, Mat X, Vec y):
        CHKERR( MLRegressorFit(self.mlregressor, X.mat, y.vec) )

    def predict(self, Mat X, Vec y):
        CHKERR( MLRegressorPredict(self.mlregressor, X.mat, y.vec) )

    def reset(self):
        CHKERR( MLRegressorReset(self.mlregressor) )

    def destory(self):
        CHKERR( MLRegressorDestroy(&self.mlregressor) )
        return self

    def setType(self, mlregressor_type):
        cdef PetscMLRegressorType cval = NULL
        mlregressor_type = str2bytes(mlregressor_type, &cval)
        CHKERR( MLRegressorSetType(self.mlregressor, cval) )

del MLRegressorType
