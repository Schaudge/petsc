from sympy.physics.units.definitions.unit_definitions import dm
import numpy as np

cdef class DMBF(DM):
    
    def setBlockSize(self,blockSize):
        cdef PetscInt *blockSizec
        iarray_i(blockSize,NULL,&blockSizec)
        CHKERR( DMBFSetBlockSize(self.dm,blockSizec) )
    
    def getBlockSize(self):
        cdef PetscInt blockSize[3]
        CHKERR( DMBFGetBlockSize(self.dm,blockSize) )      
        return array_i(3,blockSize)
    
    def getLocalSize(self):
        cdef PetscInt localSize
        CHKERR( DMBFGetLocalSize(self.dm,&localSize) )
        return localSize
    
    def getGlobalSize(self):
        cdef PetscInt globalSize
        CHKERR( DMBFGetGlobalSize(self.dm,&globalSize) )
        return globalSize
    
    def getGhostSize(self):
        cdef PetscInt ghostSize
        CHKERR( DMBFGetGhostSize(self.dm,&ghostSize) )
        return ghostSize
    
    def iterateOverCells(self,cellf,args=None,kargs=None):        
        if args  is None: args  = ()
        if kargs is None: kargs = {}        
        context = (cellf, args, kargs)
        self.set_attr('__cellf__', context)
        CHKERR( DMBFIterateOverCells(self.dm,DMBF_CellCallback,<void*>context) )
        return args
        
    def iterateOverFaces(self,facef,args=None,kargs=None):           
        if args  is None: args  = ()
        if kargs is None: kargs = {}        
        context = (facef, args, kargs)
        self.set_attr('__facef__', context)
        CHKERR( DMBFIterateOverFaces(self.dm,DMBF_FaceCallback,<void*>context) )
    
    def setCellDataSize(self,valsPerElemRead,nValsPerElemRead,valsPerElemReadWrite,nValsPerElemReadWrite):
        cdef PetscInt  *valsPerElemReadc
        cdef PetscInt  nValsPerElemReadc
        cdef PetscInt  *valsPerElemReadWritec
        cdef PetscInt  nValsPerElemReadWritec
        
        valsPerElemRead      = iarray_i(valsPerElemRead,&nValsPerElemReadc,&valsPerElemReadc)
        valsPerElemReadWrite = iarray_i(valsPerElemReadWrite,&nValsPerElemReadWritec,&valsPerElemReadWritec)
        CHKERR( DMBFSetCellDataSize(self.dm,valsPerElemReadc,nValsPerElemReadc,valsPerElemReadWritec,nValsPerElemReadWritec) )
    
    def setCellData(self,int nVecRead,Vec[:] vecRead or None,int nVecReadWrite, Vec[:] vecReadWrite or None):
        cdef int i   
        cdef PetscVec *PetscVecRead, *PetscVecReadWrite  
        if vecRead is None:
            PetscVecRead  = NULL
        else:
            PetscVecRead  = <PetscVec*> malloc(sizeof(PetscVec)*nVecRead)
        if vecReadWrite is None:
            PetscVecReadWrite  = NULL
        else:
            PetscVecReadWrite  = <PetscVec*> malloc(sizeof(PetscVec)*nVecReadWrite)
        
        for i in range(nVecRead):
            PetscVecRead[i] = vecRead[i].vec
        for i in range(nVecReadWrite):
            PetscVecReadWrite[i] = vecReadWrite[i].vec
                
        CHKERR( DMBFSetCellData(self.dm,PetscVecRead,PetscVecReadWrite) )
    
        #self.__p__setCellData(nVecRead,nVecWrite,VecRead,VecWrite)
   
    def getCellDataSize(self):
        cdef PetscInt  *valsPerElemRead
        cdef PetscInt  nValsPerElemRead
        cdef PetscInt  *valsPerElemReadWrite
        cdef PetscInt  nValsPerElemReadWrite
                
        CHKERR( DMBFGetCellDataSize(self.dm,&valsPerElemRead,&nValsPerElemRead,&valsPerElemReadWrite,&nValsPerElemReadWrite) )
        return (array_i(nValsPerElemRead,     valsPerElemRead),     nValsPerElemRead,
                array_i(nValsPerElemReadWrite,valsPerElemReadWrite),nValsPerElemReadWrite)
    
    cdef getCellDataSizeRead(self):
        cdef PetscInt  *valsPerElemRead
        cdef PetscInt  nValsPerElemRead
                
        CHKERR( DMBFGetCellDataSize(self.dm,&valsPerElemRead,&nValsPerElemRead,NULL,NULL) )
        return (array_i(nValsPerElemRead,     valsPerElemRead),    nValsPerElemRead)
    
    cdef getCellDataSizeReadWrite(self):
        cdef PetscInt  *valsPerElemReadWrite
        cdef PetscInt  nValsPerElemReadWrite
                
        CHKERR( DMBFGetCellDataSize(self.dm,NULL,NULL,&valsPerElemReadWrite,&nValsPerElemReadWrite) )
        return (array_i(nValsPerElemReadWrite,valsPerElemReadWrite),nValsPerElemReadWrite)
    
    def getCellData(self,int nVecReadWrite,Vec[:] vecReadWrite):
        cdef PetscVec *PetscVecReadWrite = <PetscVec*> malloc(sizeof(PetscVec)*nVecReadWrite)
        for i in range(nVecReadWrite):
            PetscVecReadWrite[i] = vecReadWrite[i].vec
        CHKERR( DMBFGetCellData(self.dm,NULL,PetscVecReadWrite) )

    def communicateGhostCells(self):
         CHKERR( DMBFCommunicateGhostCells(self.dm) )
    
    '''
    def iterateOverCellsVectors(self,cellf,vec_in,vec_out,args=None,kargs=None):        
        if args  is None: args  = ()
        if kargs is None: kargs = {}        
        context = (cellf, args, kargs)
        self.set_attr('__cellf__', context)
        CHKERR( DMBFIterateOverCellsVectors(self.dm, DMBF_CellCallback, &(vec_in[0]), len(vec_in) <void*>context) )
    '''
    def matCreateFromCellFunction(self,cell_function):
        ctx = CellMatCtx(self,cell_function)
        K = Mat().createPython([self.getGlobalSize(), self.getGlobalSize()], comm=self.comm)
        K.setPythonContext(ctx)
        K.setUp()
        return K

class CellMatCtx(object):

    def __init__(self, dm, cell_function):
        self.dm = dm
        self.cellf = cell_function
    
    def mult(self, K, x, b):
        b.set(0.0)
        self.dm.setCellData(x,b)
        self.dm.communicateGhostCells()
        self.dm.iterateOverCells(self.cellf)
        self.dm.getCellData(b)
        

cdef class PyDM_BF_Cell:
    cdef DM_BF_Cell *cell
    
    def __cinit__(self):
        self.cell = NULL

    cdef _setup(self, DM_BF_Cell* cell):
        self.cell = cell
        return self
    
    def getDataRead(self,DMBF dm):
        valsPerElemRead,_ = dm.getCellDataSizeRead()
        return array_r(sum(valsPerElemRead),self.cell.dataRead)
    
    def getDataReadWrite(self,DMBF dm):
        ValsPerElemReadWrite,_ = dm.getCellDataSizeReadWrite()
        return array_r(sum(ValsPerElemReadWrite),self.cell.dataReadWrite)
    
    def setDataReadWrite(self,DMBF dm,double[:] data,mode=0): # 0 is add values      
        cdef PetscInt nVals, i       
        ValsPerElemReadWrite,_ = dm.getCellDataSizeReadWrite()
        nVals = sum(ValsPerElemReadWrite)
        if mode:
            for i in range(nVals):
                self.cell.dataReadWrite[i] = data[i]
        else:
            for i in range(nVals):
                self.cell.dataReadWrite[i] += data[i]
    
    property corner:
        def __get__(self):
            return array_r(8*3,self.cell.corner)
    
    property volume:
        def __get__(self):
            return self.cell.volume
    
    property sidelength:
        def __get__(self):
            return array_r(3,self.cell.sidelength)
    
    property indexLocal:
        def __get__(self):
            return self.cell.indexLocal
    
    property indexGlobal:
        def __get__(self):
            return self.cell.indexGlobal
    
    property level:
        def __get__(self):
            return self.cell.level
    
cdef PyCell_Create(DM_BF_Cell* cell):
    return PyDM_BF_Cell()._setup(cell)    


cdef class PyDM_BF_Face:
    cdef DM_BF_Face *face
    
    def __cinit__(self):
        self.face = NULL

    cdef _setup(self, DM_BF_Face* face):
        self.face = face
        return self
    
    def getNCellsL(self):
        return self.face.nCellsL
    
    def getNCellsR(self):
        return self.face.nCellsR
    
    def getCellsL(self):
        cdef int i
        cellsL = [PyCell_Create(self.face.cellL[i]) for i in range(self.face.nCellsL)] 
        return cellsL
    
    def getCellsR(self):
        cdef int i
        cellsR = [PyCell_Create(self.face.cellR[i]) for i in range(self.face.nCellsR)] 
        return cellsR

    
cdef PyFace_Create(DM_BF_Face* face):
    return PyDM_BF_Face()._setup(face)            
