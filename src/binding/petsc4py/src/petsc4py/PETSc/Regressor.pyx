# --------------------------------------------------------------------

class RegressorType:
    LINEAR = S_(PETSCREGRESSORLINEAR)

cdef class Regressor(Object):

    Type = RegressorType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.regressor
        self.regressor = NULL

    def view(self, Viewer viewer=None) -> None:
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( PetscRegressorView(self.regressor, cviewer) )

    def create(self, comm: Comm | None = None) -> Self:
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor newregressor = NULL
        CHKERR( PetscRegressorCreate(ccomm, &newregressor) )
        CHKERR( PetscCLEAR(self.obj) ); self.regressor = newregressor
        return self

    def setUp(self) -> None:
        CHKERR( PetscRegressorSetUp(self.regressor) )

    def fit(self, Mat X, Vec y) -> None:
        CHKERR( PetscRegressorFit(self.regressor, X.mat, y.vec) )

    def predict(self, Mat X, Vec y) -> None:
        CHKERR( PetscRegressorPredict(self.regressor, X.mat, y.vec) )

    def reset(self) -> None:
        CHKERR( PetscRegressorReset(self.regressor) )

    def destory(self) -> Self:
        CHKERR( PetscRegressorDestroy(&self.regressor) )
        return self

    def setType(self, regressor_type: Type | str) -> None:
        cdef PetscRegressorType cval = NULL
        regressor_type = str2bytes(regressor_type, &cval)
        CHKERR( PetscRegressorSetType(self.regressor, cval) )

del RegressorType
