# --------------------------------------------------------------------

cdef class Section(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sec
        self.sec  = NULL

    def __dealloc__(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        self.sec = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSectionView(self.sec, vwr) )

    def destroy(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSection newsec = NULL
        CHKERR( PetscSectionCreate(ccomm, &newsec) )
        PetscCLEAR(self.obj); self.sec = newsec
        return self

    def clone(self):
        cdef Section sec = <Section>type(self)()
        CHKERR( PetscSectionClone(self.sec, &sec.sec) )
        return sec

    def setUp(self):
        CHKERR( PetscSectionSetUp(self.sec) )

    def reset(self):
        CHKERR( PetscSectionReset(self.sec) )

    def getNumFields(self):
        cdef PetscInt numFields = 0
        CHKERR( PetscSectionGetNumFields(self.sec, &numFields) )
        return toInt(numFields)

    def setNumFields(self,numFields):
        cdef PetscInt cnumFields = asInt(numFields)
        CHKERR( PetscSectionSetNumFields(self.sec, cnumFields) )

    def getFieldName(self,field):
        cdef PetscInt cfield = asInt(field)
        cdef const char *fieldName = NULL
        CHKERR( PetscSectionGetFieldName(self.sec,cfield,&fieldName) )
        return bytes2str(fieldName)

    def setFieldName(self,field,fieldName):
        cdef PetscInt cfield = asInt(field)
        cdef const char *cname = NULL
        fieldName = str2bytes(fieldName, &cname)
        CHKERR( PetscSectionSetFieldName(self.sec,cfield,cname) )

    def getFieldComponents(self,field):
        cdef PetscInt cfield = asInt(field), cnumComp = 0
        CHKERR( PetscSectionGetFieldComponents(self.sec,cfield,&cnumComp) )
        return toInt(cnumComp)

    def setFieldComponents(self,field,numComp):
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumComp = asInt(numComp)
        CHKERR( PetscSectionSetFieldComponents(self.sec,cfield,cnumComp) )

    def getChart(self):
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( PetscSectionGetChart(self.sec, &pStart, &pEnd) )
        return toInt(pStart), toInt(pEnd)

    def setChart(self, pStart, pEnd):
        cdef PetscInt cStart = asInt(pStart)
        cdef PetscInt cEnd   = asInt(pEnd)
        CHKERR( PetscSectionSetChart(self.sec, cStart, cEnd) )

    def getPermutation(self):
        cdef IS perm = IS()
        CHKERR( PetscSectionGetPermutation(self.sec, &perm.iset))
        PetscINCREF(perm.obj)
        return perm

    def setPermutation(self, IS perm):
        CHKERR( PetscSectionSetPermutation(self.sec, perm.iset))

    def getCount(self,point):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetCount(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setCount(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetCount(self.sec,cpoint,cnumDof) )

    def addCount(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddCount(self.sec,cpoint,cnumDof) )

    def getFieldCount(self,point,field):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldCount(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldCount(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldCount(self.sec,cpoint,cfield,cnumDof) )

    def addFieldCount(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldCount(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintCount(self,point):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetConstraintCount(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setConstraintCount(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetConstraintCount(self.sec,cpoint,cnumDof) )

    def addConstraintCount(self,point,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddConstraintCount(self.sec,cpoint,cnumDof) )

    def getFieldConstraintCount(self,point,field):
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldConstraintCount(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldConstraintCount(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldConstraintCount(self.sec,cpoint,cfield,cnumDof) )

    def addFieldConstraintCount(self,point,field,numDof):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldConstraintCount(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintIndices(self,point):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetConstraintCount(self.sec, cpoint, &nindex) )
        CHKERR( PetscSectionGetConstraintIndices(self.sec, cpoint, &indices) )
        return array_i(nindex, indices)

    def setConstraintIndices(self,point,indices):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetConstraintCount(self.sec,cpoint,nindex) )
        CHKERR( PetscSectionSetConstraintIndices(self.sec,cpoint,cindices) )

    def getFieldConstraintIndices(self,point,field):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetFieldConstraintCount(self.sec,cpoint,cfield,&nindex) )
        CHKERR( PetscSectionGetFieldConstraintIndices(self.sec,cpoint,cfield,&indices) )
        return array_i(nindex, indices)

    def setFieldConstraintIndices(self,point,field,indices):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetFieldConstraintCount(self.sec,cpoint,cfield,nindex) )
        CHKERR( PetscSectionSetFieldConstraintIndices(self.sec,cpoint,cfield,cindices) )

    def getMaxCount(self):
        cdef PetscInt maxDof = 0
        CHKERR( PetscSectionGetMaxCount(self.sec,&maxDof) )
        return toInt(maxDof)

    def getStorageSize(self):
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetStorageSize(self.sec,&size) )
        return toInt(size)

    def getConstrainedStorageSize(self):
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetConstrainedStorageSize(self.sec,&size) )
        return toInt(size)

    def getOffset(self,point):
        cdef PetscInt cpoint = asInt(point), offset = 0
        CHKERR( PetscSectionGetOffset(self.sec,cpoint,&offset) )
        return toInt(offset)

    def setOffset(self,point,offset):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetOffset(self.sec,cpoint,coffset) )

    def getFieldOffset(self,point,field):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt offset = 0
        CHKERR( PetscSectionGetFieldOffset(self.sec,cpoint,cfield,&offset) )
        return toInt(offset)

    def setFieldOffset(self,point,field,offset):
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetFieldOffset(self.sec,cpoint,cfield,coffset) )

    def getOffsetRange(self):
        cdef PetscInt oStart = 0, oEnd = 0
        CHKERR( PetscSectionGetOffsetRange(self.sec,&oStart,&oEnd) )
        return toInt(oStart),toInt(oEnd)

    def createGlobalSection(self, SF sf):
        cdef Section gsec = Section()
        CHKERR( PetscSectionCreateGlobalSection(self.sec,sf.sf,PETSC_FALSE,PETSC_FALSE,&gsec.sec) )
        return gsec
