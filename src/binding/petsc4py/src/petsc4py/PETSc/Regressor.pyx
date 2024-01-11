# --------------------------------------------------------------------

class RegressorType:
    LINEAR = S_(PETSCREGRESSORLINEAR)
    PYTHON = S_(PETSCREGRESSORPYTHON)

cdef class Regressor(Object):

    Type = RegressorType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.regressor
        self.regressor = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the solver.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscRegressorView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( PetscRegressorView(self.regressor, cviewer) )

    def create(self, comm: Comm | None = None) -> Self:
        """Create a PetscRegressor solver.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscRegressorCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor newregressor = NULL
        CHKERR( PetscRegressorCreate(ccomm, &newregressor) )
        CHKERR( PetscCLEAR(self.obj) ); self.regressor = newregressor
        return self

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorSetUp

        """
        CHKERR( PetscRegressorSetUp(self.regressor) )

    def fit(self, Mat X, Vec y) -> None:
        """Fit the regression problem.

        Collective.

        Parameters
        ----------
        x
            The starting vector
        y
            The resulting vector

        See Also
        --------
        petsc.PetscRegressorFit

        """
        CHKERR( PetscRegressorFit(self.regressor, X.mat, y.vec) )

    def predict(self, Mat X, Vec y) -> None:
        """Predict the regression problem.

        Collective.

        Parameters
        ----------
        X
            The predict matrix
        y
            The predicted vector

        See Also
        --------
        petsc.PetscRegressorPredict

        """
        CHKERR( PetscRegressorPredict(self.regressor, X.mat, y.vec) )

    def reset(self) -> None:
        """Reset the regressor

        Collective.

        See Also
        --------
        petsc.PetscRegressorReset

        """
        CHKERR( PetscRegressorReset(self.regressor) )

    def destory(self) -> Self:
        """Destroy the regression object.

        Collective.

        See Also
        --------
        petsc.PetscRegressorDestroy

        """
        CHKERR( PetscRegressorDestroy(&self.regressor) )
        return self

    def setType(self, regressor_type: Type | str) -> None:
        """Set the type of the regression.

        Logically collective.

        Parameters
        ----------
        regressor_type
            The type of the solver.

        See Also
        --------
        getType, petsc.PetscRegressorSetType

        """
        cdef PetscRegressorType cval = NULL
        regressor_type = str2bytes(regressor_type, &cval)
        CHKERR( PetscRegressorSetType(self.regressor, cval) )

    def createPython(self, context: Any = None, comm: Comm | None = None) -> Self:
        """Create an optimization solver of Python type.

        Collective.

        Parameters
        ----------
        context
            An instance of the Python class implementing the required methods.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_regressor, setType, setPythonContext, Type.PYTHON

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor reg = NULL
        CHKERR( PetscRegressorCreate(ccomm, &reg) )
        CHKERR( PetscCLEAR(self.obj) ); self.reg = reg
        CHKERR( PetscRegressorSetType(self.reg, PETSCREGRESSORPYTHON) )
        CHKERR( PetscRegressorPythonSetContext(self.tao, <void*>context) )
        return self

    def setPythonContext(self, context: Any) -> None:
        """Set the instance of the class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_tao, getPythonContext

        """
        CHKERR( PetscRegressorPythonSetContext(self.reg, <void*>context) )

    def getPythonContext(self) -> Any:
        """Return the instance of the class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_tao, setPythonContext

        """
        cdef void *context = NULL
        CHKERR( PetscRegressorPythonGetContext(self.reg, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_tao, setPythonContext, getPythonType
        petsc.TaoPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( PetscRegressorPythonSetType(self.reg, cval) )

    def getPythonType(self) -> str:
        """Return the fully qualified Python name of the class used by the solver.

        Not collective.

        See Also
        --------
        petsc_python_tao, setPythonContext, setPythonType
        petsc.TaoPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR( PetscRegressorPythonGetType(self.reg, &cval) )
        return bytes2str(cval)
del RegressorType
