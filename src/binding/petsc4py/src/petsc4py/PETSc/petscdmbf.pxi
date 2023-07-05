
cdef extern from "petscdmbf.h" nogil:
 
    int DMBFSetBlockSize(PetscDM,PetscInt*);
    int DMBFGetBlockSize(PetscDM,PetscInt*);
    int DMBFSetCellDataSize(PetscDM,PetscInt*,PetscInt,PetscInt*,PetscInt);
    int DMBFGetCellDataSize(PetscDM,PetscInt**,PetscInt*,PetscInt**,PetscInt*);
    
    int DMBFGetLocalSize(PetscDM,PetscInt*);
    int DMBFGetGlobalSize(PetscDM,PetscInt*);
    int DMBFGetGhostSize(PetscDM,PetscInt*);
    
    int DMBFCoarsenInPlace(PetscDM,PetscInt);
    int DMBFRefineInPlace(PetscDM,PetscInt);
    
    ctypedef int (*PetscDMBF_CellCallback)(PetscDM,
                                           DM_BF_Cell*,
                                           void*) except PETSC_ERR_PYTHON
    
    int DMBFIterateOverCellsVectors(PetscDM,PetscDMBF_CellCallback,void*,PetscVec*,PetscInt,PetscVec*,PetscInt);
    int DMBFIterateOverCells(PetscDM,PetscDMBF_CellCallback,void*);
    
    ctypedef struct DM_BF_Cell:
        PetscReal         corner[8*3], volume, sidelength[3], dummy1[4];
        PetscInt          indexLocal, indexGlobal;
        PetscInt          level, dummy2;
        const PetscScalar **vecViewRead;
        PetscScalar       **vecViewReadWrite;
        const PetscScalar *dataRead;
        PetscScalar       *dataReadWrite;
    
    ctypedef int (*PetscDMBF_FaceCallback)(PetscDM,
                                           DM_BF_Face*,
                                           void*) except PETSC_ERR_PYTHON
    
    int DMBFIterateOverFacesVectors(PetscDM,PetscDMBF_FaceCallback,void*,PetscVec*,PetscInt,PetscVec*,PetscInt);
    int DMBFIterateOverFaces(PetscDM,PetscDMBF_FaceCallback,void*);
    
    ctypedef struct DM_BF_Face:
        PetscInt    nCellsL, nCellsR;
        DM_BF_Cell  *cellL[4], *cellR[4];

    int DMBFSetCellData(PetscDM,PetscVec*,PetscVec*);
    int DMBFGetCellData(PetscDM,PetscVec*,PetscVec*);
    int DMBFCommunicateGhostCells(PetscDM);
    
    #int DMBFGetP4est(DM,void*);
    #int DMBFGetGhost(DM,void*);
    
    int DMBFVTKWriteAll(PetscObject,PetscViewer);

cdef inline DMBF ref_DMBF(PetscDM dm):
    cdef DMBF ob = <DMBF> DMBF()
    ob.dm = dm
    PetscINCREF(ob.obj)
    return ob

cdef int DMBF_CellCallback(
        PetscDM dm, 
        DM_BF_Cell*cell, 
        void*ctx
        ) except PETSC_ERR_PYTHON with gil:
    cdef DMBF Dm = ref_DMBF(dm)
    cdef object context = Dm.get_attr('__cellf__')
    cdef PyDM_BF_Cell PyCell = PyCell_Create(cell)
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (cellf, args, kargs) = context
    cellf(Dm, PyCell,  *args, **kargs)
    return 0


cdef int DMBF_FaceCallback(
        PetscDM dm, 
        DM_BF_Face*face, 
        void*ctx
        ) except PETSC_ERR_PYTHON with gil:
    cdef DMBF Dm = ref_DMBF(dm)
    cdef object context = Dm.get_attr('__facef__')
    cdef PyDM_BF_Face PyFace = PyFace_Create(face)
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (facef, args, kargs) = context
    facef(Dm, PyFace,  *args, **kargs)
    return 0

