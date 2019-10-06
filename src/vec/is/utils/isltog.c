
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/
#include <petsc/private/hashmapi.h>
#include <petscsf.h>
#include <petscviewer.h>
#include <petscctable.h>
#include <petscbt.h>

PetscClassId IS_LTOGM_CLASSID;
static PetscErrorCode  ISLocalToGlobalMappingSetUpBlockInfo_Private(ISLocalToGlobalMapping);

typedef struct {
  PetscInt *globals;
} ISLocalToGlobalMapping_Basic;

typedef struct {
  PetscHMapI globalht;
} ISLocalToGlobalMapping_Hash;


PetscErrorCode ISGetPointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt       numCells, step = 1;
  PetscBool      isStride;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  *pStart = 0;
  *points = NULL;
  ierr = ISGetLocalSize(pointIS, &numCells);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pointIS, ISSTRIDE, &isStride);CHKERRQ(ierr);
  if (isStride) {ierr = ISStrideGetInfo(pointIS, pStart, &step);CHKERRQ(ierr);}
  *pEnd   = *pStart + numCells;
  if (!isStride || step != 1) {ierr = ISGetIndices(pointIS, points);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode ISRestorePointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt       step = 1;
  PetscBool      isStride;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscObjectTypeCompare((PetscObject) pointIS, ISSTRIDE, &isStride);CHKERRQ(ierr);
  if (isStride) {ierr = ISStrideGetInfo(pointIS, pStart, &step);CHKERRQ(ierr);}
  if (!isStride || step != 1) {ierr = ISGetIndices(pointIS, points);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode ISGetPointSubrange(IS subpointIS, PetscInt pStart, PetscInt pEnd, const PetscInt *points)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (points) {
    ierr = ISSetType(subpointIS, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(subpointIS, pEnd-pStart, &points[pStart], PETSC_USE_POINTER);CHKERRQ(ierr);
  } else {
    ierr = ISSetType(subpointIS, ISSTRIDE);CHKERRQ(ierr);
    ierr = ISStrideSetStride(subpointIS, pEnd-pStart, pStart, 1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------------*/

/*
    Creates the global mapping information in the ISLocalToGlobalMapping structure

    If the user has not selected how to handle the global to local mapping then use HASH for "large" problems
*/
static PetscErrorCode ISGlobalToLocalMappingSetUp(ISLocalToGlobalMapping mapping)
{
  PetscInt       i,*idx = mapping->indices,n = mapping->n,end,start;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mapping->data) PetscFunctionReturn(0);
  end   = 0;
  start = PETSC_MAX_INT;

  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;
  if (!((PetscObject)mapping)->type_name) {
    if ((end - start) > PetscMax(4*n,1000000)) {
      ierr = ISLocalToGlobalMappingSetType(mapping,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
    } else {
      ierr = ISLocalToGlobalMappingSetType(mapping,ISLOCALTOGLOBALMAPPINGBASIC);CHKERRQ(ierr);
    }
  }
  ierr = (*mapping->ops->globaltolocalmappingsetup)(mapping);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Basic(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  PetscInt                    i,*idx = mapping->indices,n = mapping->n,end,start,*globals;
  ISLocalToGlobalMapping_Basic *map;

  PetscFunctionBegin;
  start            = mapping->globalstart;
  end              = mapping->globalend;
  ierr             = PetscNew(&map);CHKERRQ(ierr);
  ierr             = PetscMalloc1(end-start+2,&globals);CHKERRQ(ierr);
  map->globals     = globals;
  for (i=0; i<end-start+1; i++) globals[i] = -1;
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }
  mapping->data = (void*)map;
  ierr = PetscLogObjectMemory((PetscObject)mapping,(end-start+1)*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Hash(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  PetscInt                    i,*idx = mapping->indices,n = mapping->n;
  ISLocalToGlobalMapping_Hash *map;

  PetscFunctionBegin;
  ierr = PetscNew(&map);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&map->globalht);CHKERRQ(ierr);
  for (i=0; i<n; i++ ) {
    if (idx[i] < 0) continue;
    ierr = PetscHMapISet(map->globalht,idx[i],i);CHKERRQ(ierr);
  }
  mapping->data = (void*)map;
  ierr = PetscLogObjectMemory((PetscObject)mapping,2*n*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Basic(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  ISLocalToGlobalMapping_Basic *map  = (ISLocalToGlobalMapping_Basic *)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(0);
  ierr = PetscFree(map->globals);CHKERRQ(ierr);
  ierr = PetscFree(mapping->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Hash(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  ISLocalToGlobalMapping_Hash *map  = (ISLocalToGlobalMapping_Hash*)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(0);
  ierr = PetscHMapIDestroy(&map->globalht);CHKERRQ(ierr);
  ierr = PetscFree(mapping->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define GTOLTYPE _Basic
#define GTOLNAME _Basic
#define GTOLBS mapping->bs
#define GTOL(g, local) do {                  \
    local = map->globals[g/bs - start];      \
    local = bs*local + (g % bs);             \
  } while (0)

#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Basic
#define GTOLNAME Block_Basic
#define GTOLBS 1
#define GTOL(g, local) do {                  \
    local = map->globals[g - start];         \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME _Hash
#define GTOLBS mapping->bs
#define GTOL(g, local) do {                         \
    (void)PetscHMapIGet(map->globalht,g/bs,&local); \
    local = bs*local + (g % bs);                    \
   } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME Block_Hash
#define GTOLBS 1
#define GTOL(g, local) do {                         \
    (void)PetscHMapIGet(map->globalht,g,&local);    \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

/*@
    ISLocalToGlobalMappingDuplicate - Duplicates the local to global mapping object

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   nltog - the duplicated local to global mapping

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingDuplicate(ISLocalToGlobalMapping ltog,ISLocalToGlobalMapping* nltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)ltog),ltog->bs,ltog->n,ltog->indices,PETSC_COPY_VALUES,nltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping, ISLocalToGlobalMappingGetIndices() returns an array of this length

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidIntPointer(n,2);
  *n = mapping->bs*mapping->n;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,PetscViewer viewer)
{
  PetscInt       i;
  PetscMPIInt    rank;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mapping),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mapping),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)mapping,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<mapping->n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not collective

    Input Parameter:
.   is - index set containing the global numbers for each local number

    Output Parameter:
.   mapping - new mapping data structure

    Notes:
    the block size of the IS determines the block size of the mapping
    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetFromOptions()
@*/
PetscErrorCode  ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       n,bs;
  const PetscInt *indices;
  MPI_Comm       comm;
  PetscBool      isblock;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(mapping,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&isblock);CHKERRQ(ierr);
  if (!isblock) {
    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,1,n,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  } else {
    ierr = ISGetBlockSize(is,&bs);CHKERRQ(ierr);
    ierr = ISBlockGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,bs,n/bs,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISBlockRestoreIndices(is,&indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingCreateSF - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Collective

    Input Parameter:
+   sf - star forest mapping contiguous local indices to (rank, offset)
-   start - first global index on this process

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingSetFromOptions()
@*/
PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF sf,PetscInt start,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       i,maxlocal,nroots,nleaves,*globals,*ltog;
  const PetscInt *ilocal;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(mapping,3);

  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  if (ilocal) {
    for (i=0,maxlocal=0; i<nleaves; i++) maxlocal = PetscMax(maxlocal,ilocal[i]+1);
  }
  else maxlocal = nleaves;
  ierr = PetscMalloc1(nroots,&globals);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxlocal,&ltog);CHKERRQ(ierr);
  for (i=0; i<nroots; i++) globals[i] = start + i;
  for (i=0; i<maxlocal; i++) ltog[i] = -1;
  ierr = PetscSFBcastBegin(sf,MPIU_INT,globals,ltog);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,globals,ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,1,maxlocal,ltog,PETSC_OWN_POINTER,mapping);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingResetBlockInfo_Private(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mapping->info_procs);CHKERRQ(ierr);
  ierr = PetscFree(mapping->info_numprocs);CHKERRQ(ierr);
  if (mapping->info_indices) {
    PetscInt i;

    ierr = PetscFree(mapping->info_indices[0]);CHKERRQ(ierr);
    for (i=1; i<mapping->info_nproc; i++) {
      ierr = PetscFree(mapping->info_indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mapping->info_indices);CHKERRQ(ierr);
  }
  if (mapping->info_nodei) {
    ierr = PetscFree(mapping->info_nodei[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(mapping->info_nodec,mapping->info_nodei);CHKERRQ(ierr);
  mapping->info_cached = PETSC_FALSE;
  mapping->info_free = PETSC_FALSE;
  mapping->info_nfree = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingSetBlockSize - Sets the blocksize of the mapping

    Not collective

    Input Parameters:
+   mapping - mapping data structure
-   bs - the blocksize

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingSetBlockSize(ISLocalToGlobalMapping mapping,PetscInt bs)
{
  PetscInt       *nid;
  const PetscInt *oid;
  PetscInt       i,cn,on,obs,nn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (bs < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid block size %D",bs);
  if (bs == mapping->bs) PetscFunctionReturn(0);
  on  = mapping->n;
  obs = mapping->bs;
  oid = mapping->indices;
  nn  = (on*obs)/bs;
  if ((on*obs)%bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block size %D is inconsistent with block size %D and number of block indices %D",bs,obs,on);

  ierr = PetscMalloc1(nn,&nid);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(mapping,&oid);CHKERRQ(ierr);
  for (i=0;i<nn;i++) {
    PetscInt j;
    for (j=0,cn=0;j<bs-1;j++) {
      if (oid[i*bs+j] < 0) { cn++; continue; }
      if (oid[i*bs+j] != oid[i*bs+j+1]-1) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block sizes %D and %D are incompatible with the block indices: non consecutive indices %D %D",bs,obs,oid[i*bs+j],oid[i*bs+j+1]);
    }
    if (oid[i*bs+j] < 0) cn++;
    if (cn) {
      if (cn != bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block sizes %D and %D are incompatible with the block indices: invalid number of negative entries in block %D",bs,obs,cn);
      nid[i] = -1;
    } else {
      nid[i] = oid[i*bs]/bs;
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(mapping,&oid);CHKERRQ(ierr);

  mapping->n           = nn;
  mapping->bs          = bs;
  ierr                 = PetscFree(mapping->indices);CHKERRQ(ierr);
  mapping->indices     = nid;
  mapping->globalstart = 0;
  mapping->globalend   = 0;

  /* reset the cached information */
  ierr = ISLocalToGlobalMappingResetBlockInfo_Private(mapping);CHKERRQ(ierr);

  if (mapping->ops->destroy) {
    ierr = (*mapping->ops->destroy)(mapping);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@
    ISLocalToGlobalMappingGetBlockSize - Gets the blocksize of the mapping
    ordering and a global parallel ordering.

    Not Collective

    Input Parameters:
.   mapping - mapping data structure

    Output Parameter:
.   bs - the blocksize

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockSize(ISLocalToGlobalMapping mapping,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  *bs = mapping->bs;
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   bs - the block size
.   n - the number of local elements divided by the block size, or equivalently the number of block indices
.   indices - the global index for each local element, these do not need to be in increasing order (sorted), these values should not be scaled (i.e. multiplied) by the blocksize bs
-   mode - see PetscCopyMode

    Output Parameter:
.   mapping - new mapping data structure

    Notes:
    There is one integer value in indices per block and it represents the actual indices bs*idx + j, where j=0,..,bs-1

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingSetFromOptions(), ISLOCALTOGLOBALMAPPINGBASIC, ISLOCALTOGLOBALMAPPINGHASH
          ISLocalToGlobalMappingSetType(), ISLocalToGlobalMappingType
@*/
PetscErrorCode  ISLocalToGlobalMappingCreate(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt indices[],PetscCopyMode mode,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       *in;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices,3);
  PetscValidPointer(mapping,4);

  *mapping = NULL;
  ierr = ISInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(*mapping,IS_LTOGM_CLASSID,"ISLocalToGlobalMapping","Local to global mapping","IS",
                           comm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);CHKERRQ(ierr);
  (*mapping)->n             = n;
  (*mapping)->bs            = bs;
  (*mapping)->info_cached   = PETSC_FALSE;
  (*mapping)->info_free     = PETSC_FALSE;
  (*mapping)->info_nfree    = PETSC_FALSE;
  (*mapping)->info_procs    = NULL;
  (*mapping)->info_numprocs = NULL;
  (*mapping)->info_indices  = NULL;
  (*mapping)->info_nodec    = NULL;
  (*mapping)->info_nodei    = NULL;

  (*mapping)->ops->globaltolocalmappingapply      = NULL;
  (*mapping)->ops->globaltolocalmappingapplyblock = NULL;
  (*mapping)->ops->destroy                        = NULL;
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&in);CHKERRQ(ierr);
    ierr = PetscArraycpy(in,indices,n);CHKERRQ(ierr);
    (*mapping)->indices = in;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (mode == PETSC_OWN_POINTER) {
    (*mapping)->indices = (PetscInt*)indices;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  } else SETERRQ(comm,PETSC_ERR_SUP,"Cannot currently use PETSC_USE_POINTER");
  PetscFunctionReturn(0);
}

PetscFunctionList ISLocalToGlobalMappingList = NULL;


/*@
   ISLocalToGlobalMappingSetFromOptions - Set mapping options from the options database.

   Not collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

@*/
PetscErrorCode ISLocalToGlobalMappingSetFromOptions(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode             ierr;
  char                       type[256];
  ISLocalToGlobalMappingType defaulttype = "Not set";
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)mapping);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-islocaltoglobalmapping_type","ISLocalToGlobalMapping method","ISLocalToGlobalMappingSetType",ISLocalToGlobalMappingList,(char*)(((PetscObject)mapping)->type_name) ? ((PetscObject)mapping)->type_name : defaulttype,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISLocalToGlobalMappingSetType(mapping,type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Note Collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mapping) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*mapping),IS_LTOGM_CLASSID,1);
  if (--((PetscObject)(*mapping))->refct > 0) {*mapping = 0;PetscFunctionReturn(0);}
  ierr = PetscFree((*mapping)->indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingResetBlockInfo_Private(*mapping);CHKERRQ(ierr);
  if ((*mapping)->ops->destroy) {
    ierr = (*(*mapping)->ops->destroy)(*mapping);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(mapping);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingApplyIS - Creates from an IS in the local numbering
    a new index set using the global numbering defined in an ISLocalToGlobalMapping
    context.

    Collective on is

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in local numbering

    Output Parameters:
.   newis - index set in global numbering

    Notes:
    The output IS will have the same communicator of the input IS.

    Level: advanced

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
PetscErrorCode  ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidPointer(newis,3);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mapping,n,idxin,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Notes:
   The in and out array parameters may be identical.

   Level: advanced

.seealso: ISLocalToGlobalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

@*/
PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt i,bs,Nmax;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  bs   = mapping->bs;
  Nmax = bs*mapping->n;
  if (bs == 1) {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  } else {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]/bs]*bs + (in[i] % bs);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingApplyBlock - Takes a list of integers in a local block numbering and converts them to the global block numbering

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local block numbering

   Output Parameter:
.  out - indices in global block numbering

   Notes:
   The in and out array parameters may be identical.

   Example:
     If the index values are {0,1,6,7} set with a call to ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,2,2,{0,3}) then the mapping applied to 0
     (the first block) would produce 0 and the mapping applied to 1 (the second block) would produce 3.

   Level: advanced

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

@*/
PetscErrorCode ISLocalToGlobalMappingApplyBlock(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  {
    PetscInt i,Nmax = mapping->n;
    const PetscInt *idx = mapping->indices;

    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local block index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  }
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApply - Provides the local numbering for a list of integers
    specified with a global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApply() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->data) {
    ierr = ISGlobalToLocalMappingSetUp(mapping);CHKERRQ(ierr);
  }
  ierr = (*mapping->ops->globaltolocalmappingapply)(mapping,type,n,idx,nout,idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApplyIS - Creates from an IS in the global numbering
    a new index set using the local numbering defined in an ISLocalToGlobalMapping
    context.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
-   is - index set in global numbering

    Output Parameters:
.   newis - index set in local numbering

    Notes:
    The output IS will be sequential, as it encodes a purely local operation

    Level: advanced

.seealso: ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyIS(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,nout,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  PetscValidPointer(newis,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  if (type == IS_GTOLM_MASK) {
    ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  } else {
    ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nout,&idxout);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nout,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApplyBlock - Provides the local block numbering for a list of integers
    specified with a block global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - replaces global indices with no local value with -1
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApplyBlock() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApplyBlock()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyBlock(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,
                                                 PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->data) {
    ierr = ISGlobalToLocalMappingSetUp(mapping);CHKERRQ(ierr);
  }
  ierr = (*mapping->ops->globaltolocalmappingapplyblock)(mapping,type,n,idx,nout,idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetBlockInfo - Gets the neighbor information for each processor and
     each index shared with other processors

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one (including self)
.   proc - neighboring processor ranks (self is first of the list)
.   numproc - number of block indices for each of the identified processors
-   indices - block indices (in local numbering of this processor) shared with neighbors (sorted by global numbering)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreBlockInfo() when the data is no longer needed.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreBlockInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingSetUpBlockInfo_Private(mapping);CHKERRQ(ierr);
  if (nproc)    *nproc    = mapping->info_nproc;
  if (procs)    *procs    = mapping->info_procs;
  if (numprocs) *numprocs = mapping->info_numprocs;
  if (indices)  *indices  = mapping->info_indices;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetBlockNodeInfo - Gets the neighbor information for each local block index

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nnodes - number of local block indices
.   count - number of neighboring processors per local block index (including self)
-   indices - indices of processes sharing the node (sorted, may contain duplicates)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreBlockNodeInfo() when the data is no longer needed.
           The information returned by this function complements that of ISLocalToGlobalMappingGetBlockInfo().
           The latter only provides local information, and the neighboring information
           cannot be inferred in the general case, unless the mapping is locally one-to-one on each process.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetBlockInfo(), ISLocalToGlobalMappingRestoreBlockNodeInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *n,PetscInt *n_procs[],PetscInt **procs[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingSetUpBlockInfo_Private(mapping);CHKERRQ(ierr);
  if (n)       *n       = mapping->n;
  if (n_procs) *n_procs = mapping->info_nodec;
  if (procs)   *procs   = mapping->info_nodei;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreBlockNodeInfo - Frees the memory allocated by ISLocalToGlobalMappingGetBlockNodeInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
+   mapping - the mapping from local to global indexing
.   nnodes - number of block local nodes
.   count - number of neighboring processors per block node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetBlockNodeInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *nnodes,PetscInt *count[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_nfree) {
    if (count) {
      ierr = PetscFree(*count);CHKERRQ(ierr);
    }
    if (indices) {
      if (*indices) {
        ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);
      }
      ierr = PetscFree(*indices);CHKERRQ(ierr);
    }
  }
  mapping->info_nfree = PETSC_FALSE;
  if (nnodes)  *nnodes  = 0;
  if (count)   *count   = NULL;
  if (indices) *indices = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISLocalToGlobalMappingSetUpBlockInfo_Private(ISLocalToGlobalMapping mapping)
{
  PetscSF            sf;
  MPI_Comm           comm;
  const PetscSFNode *sfnode;
  PetscSFNode        *newsfnode;
  PetscTable         table;
  PetscTablePosition pos;
  PetscLayout        layout;
  const PetscInt     *gidxs,*rootdegree;
  PetscInt           *mask,*mrootdata,*leafdata,*newleafdata,*leafrd,*tmpg;
  PetscInt           nroots,nleaves,newnleaves,bs,i,j,m,mnroots,p,cum;
  PetscMPIInt        rank,size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mapping->info_cached) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)mapping,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Get mapping indices */
  ierr = ISLocalToGlobalMappingGetBlockSize(mapping,&bs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(mapping,&gidxs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(mapping,&nleaves);CHKERRQ(ierr);
  nleaves /= bs;

  /* Create layout for global indices */
  for (i=0,m=0;i<nleaves;i++) m = PetscMax(m,gidxs[i]);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&m,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(layout,m+1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);

  /* Create SF to share global indices */
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf,layout,nleaves,NULL,PETSC_OWN_POINTER,gidxs);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);

  /* communicate root degree to leaves */
  ierr = PetscSFGetGraph(sf,&nroots,NULL,NULL,&sfnode);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(sf,&rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&rootdegree);CHKERRQ(ierr);
  for (i=0,mnroots=0;i<nroots;i++) mnroots += rootdegree[i];
  ierr = PetscMalloc3(2*PetscMax(mnroots,nroots),&mrootdata,2*nleaves,&leafdata,nleaves,&leafrd);CHKERRQ(ierr);
  for (i=0,m=0;i<nroots;i++) {
    mrootdata[2*i+0] = rootdegree[i];
    mrootdata[2*i+1] = m;
    m += rootdegree[i];
  }
  ierr = PetscSFBcastBegin(sf,MPIU_2INT,mrootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_2INT,mrootdata,leafdata);CHKERRQ(ierr);

  /* allocate enough space to store ranks */
  for (i=0,newnleaves=0;i<nleaves;i++) {
    newnleaves += leafdata[2*i];
    leafrd[i] = leafdata[2*i];
  }

  /* create new SF nodes to collect multi-root data at leaves */
  ierr = PetscMalloc1(newnleaves,&newsfnode);CHKERRQ(ierr);
  for (i=0,m=0;i<nleaves;i++) {
    for (j=0;j<leafrd[i];j++) {
      newsfnode[m].rank = sfnode[i].rank;
      newsfnode[m].index = leafdata[2*i+1]+j;
      m++;
    }
  }

  /* gather ranks at multi roots */
  for (i=0;i<mnroots;i++) mrootdata[i] = -1;
  for (i=0;i<nleaves;i++) leafdata[i] = (PetscInt)rank;

  ierr = PetscSFGatherBegin(sf,MPIU_INT,leafdata,mrootdata);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf,MPIU_INT,leafdata,mrootdata);CHKERRQ(ierr);

  /* set new multi-leaves graph into the SF */
  ierr = PetscSFSetGraph(sf,mnroots,newnleaves,NULL,PETSC_OWN_POINTER,newsfnode,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);

  /* broadcoast multi-root data to multi-leaves */
  ierr = PetscMalloc1(newnleaves,&newleafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,mrootdata,newleafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,mrootdata,newleafdata);CHKERRQ(ierr);

  /* sort sharing ranks */
  for (i=0,m=0;i<nleaves;i++) {
    ierr = PetscSortInt(leafrd[i],newleafdata + m);CHKERRQ(ierr);
    m += leafrd[i];
  }

  /* Number of neighbors and their ranks */
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<newnleaves; i++) {
    ierr = PetscTableAdd(table,newleafdata[i]+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&mapping->info_nproc);CHKERRQ(ierr);
  ierr = PetscMalloc1(mapping->info_nproc+1,&mapping->info_procs);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0, cum=1; i<mapping->info_nproc; i++) {
    PetscInt dummy,newr;

    ierr = PetscTableGetNext(table,&pos,&newr,&dummy);CHKERRQ(ierr);
    newr--;
    if (newr == rank) mapping->info_procs[0] = newr; /* info on myself first */
    else mapping->info_procs[cum++] = newr;
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);

  /* collect info data */
  ierr = PetscMalloc1(mapping->info_nproc+1,&mapping->info_numprocs);CHKERRQ(ierr);
  ierr = PetscMalloc1(mapping->info_nproc+1,&mapping->info_indices);CHKERRQ(ierr);
  for (i=0;i<mapping->info_nproc+1;i++) mapping->info_indices[i] = NULL;

  ierr = PetscMalloc1(nleaves,&mask);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&tmpg);CHKERRQ(ierr);
  for (p=0;p<mapping->info_nproc;p++) {
    PetscInt *tmp, trank = mapping->info_procs[p];

    ierr = PetscMemzero(mask,nleaves*sizeof(*mask));CHKERRQ(ierr);
    for (i=0,m=0;i<nleaves;i++) {
      for (j=0;j<leafrd[i];j++) {
        if (newleafdata[m] == trank) mask[i]++;
        if (!p && newleafdata[m] != rank) mask[i]++;
        m++;
      }
    }
    for (i=0,m=0;i<nleaves;i++)
      if (mask[i] > (!p ? 1 : 0)) m++;

    ierr = PetscMalloc1(m,&tmp);CHKERRQ(ierr);
    for (i=0,m=0;i<nleaves;i++)
      if (mask[i] > (!p ? 1 : 0)) {
        tmp[m]  = i;
        tmpg[m] = gidxs[i];
        m++;
     }
    ierr = PetscSortIntWithArray(m,tmpg,tmp);CHKERRQ(ierr);
    mapping->info_indices[p]  = tmp;
    mapping->info_numprocs[p] = m;
  }

  /* Node info */
  ierr = PetscMalloc2(nleaves,&mapping->info_nodec,nleaves+1,&mapping->info_nodei);CHKERRQ(ierr);
  ierr = PetscArraycpy(mapping->info_nodec,leafrd,nleaves);CHKERRQ(ierr);
  ierr = PetscMalloc1(newnleaves,&mapping->info_nodei[0]);CHKERRQ(ierr);
  for (i=0;i<nleaves-1;i++) {
    mapping->info_nodei[i+1] = mapping->info_nodei[i] + mapping->info_nodec[i];
  }
  ierr = PetscArraycpy(mapping->info_nodei[0],newleafdata,newnleaves);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingRestoreBlockIndices(mapping,&gidxs);CHKERRQ(ierr);
  ierr = PetscFree(tmpg);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree3(mrootdata,leafdata,leafrd);CHKERRQ(ierr);
  ierr = PetscFree(newleafdata);CHKERRQ(ierr);

  /* save info for reuse */
  mapping->info_cached = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreBlockInfo - Frees the memory allocated by ISLocalToGlobalMappingGetBlockInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
+   mapping - the mapping from local to global indexing
.   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of block indices for each processor
-   indices - block indices shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetBlockInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_free) {
    if (numprocs) {
      ierr = PetscFree(*numprocs);CHKERRQ(ierr);
    }
    if (indices && *indices) {
      PetscInt i;

      ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);
      for (i=1; i<*nproc; i++) {
        ierr = PetscFree((*indices)[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(*indices);CHKERRQ(ierr);
    }
  }
  mapping->info_free = PETSC_FALSE;
  if (nproc)    *nproc    = 0;
  if (procs)    *procs    = NULL;
  if (numprocs) *numprocs = NULL;
  if (indices)  *indices  = NULL;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nproc - number of processors that are connected to this one (including self)
.   proc - neighboring processor ranks (self is first of the list)
.   numproc - number of indices for each of the identified processors
-   indices - indices (in local numbering of this processor) shared with neighbors (sorted by global numbering)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreInfo() when the data is no longer needed.

    Fortran Usage:
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.


.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscInt       **bindices = NULL,*bnumprocs = NULL,bs = mapping->bs,i,j,k,n,*bprocs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingGetBlockInfo(mapping,&n,&bprocs,&bnumprocs,&bindices);CHKERRQ(ierr);
  if (bs > 1) { /* we need to expand the cached info */
    if (indices) {
      ierr = PetscCalloc1(n,indices);CHKERRQ(ierr);
    }
    if (numprocs) {
      ierr = PetscCalloc1(n,numprocs);CHKERRQ(ierr);
    }
    if (indices || numprocs) {
      for (i=0; i<n; i++) {
        if (indices) {
          ierr = PetscMalloc1(bs*bnumprocs[i],&(*indices)[i]);CHKERRQ(ierr);
          for (j=0; j<bnumprocs[i]; j++) {
            for (k=0; k<bs; k++) {
              (*indices)[i][j*bs+k] = bs*bindices[i][j] + k;
            }
          }
        }
        if (numprocs) (*numprocs)[i] = bnumprocs[i]*bs;
      }
    }
    mapping->info_free = PETSC_TRUE;
  } else {
    if (numprocs) *numprocs = bnumprocs;
    if (indices)  *indices  = bindices;
  }
  if (nproc)    *nproc    = n;
  if (procs)    *procs    = bprocs;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by ISLocalToGlobalMappingGetInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
+   mapping - the mapping from local to global indexing
.   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingRestoreBlockInfo(mapping,nproc,procs,numprocs,indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetNodeInfo - Gets the neighbor information for each local index

    Collective on ISLocalToGlobalMapping

    Input Parameters:
.   mapping - the mapping from local to global indexing

    Output Parameter:
+   nnodes - number of local indices (same ISLocalToGlobalMappingGetSize())
.   count - number of neighboring processors per local index (including self)
-   indices - indices of processes sharing the node (sorted, may contain duplicates)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreNodeInfo() when the data is no longer needed.
           The information returned by this function complements that of ISLocalToGlobalMappingGetInfo().
           The latter only provides local information, and the neighboring information
           cannot be inferred in the general case, unless the mapping is locally one-to-one on each process.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo(), ISLocalToGlobalMappingRestoreNodeInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *nnodes,PetscInt *count[],PetscInt **indices[])
{
  PetscInt       **bindices = NULL,*bcount = NULL,bs = mapping->bs,i,j,k,bn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingGetBlockNodeInfo(mapping,&bn,&bcount,&bindices);CHKERRQ(ierr);
  if (bs > 1) { /* we need to expand the cached info */
    PetscInt *tcount;
    PetscInt c;

    ierr = PetscMalloc1(bn*bs,&tcount);CHKERRQ(ierr);
    for (i=0,c=0; i<bn; i++) {
      for (k=0; k<bs; k++) {
        tcount[i*bs+k] = bcount[i];
      }
      c += bs*bcount[i];
    }
    if (nnodes) *nnodes = bn*bs;
    if (indices) {
      PetscInt **tindices;
      PetscInt tn = bn*bs;

      ierr = PetscMalloc1(tn,&tindices);CHKERRQ(ierr);
      if (tn) {
        ierr = PetscMalloc1(c,&tindices[0]);CHKERRQ(ierr);
      }
      for (i=0;i<tn-1;i++) {
        tindices[i+1] = tindices[i] + tcount[i];
      }
      for (i=0; i<bn; i++) {
        for (k=0; k<bs; k++) {
          for (j=0;j<bcount[i];j++) {
            tindices[i*bs+k][j] = bindices[i][j];
          }
        }
      }
      *indices = tindices;
    }
    if (count) *count = tcount;
    else {
      ierr = PetscFree(tcount);CHKERRQ(ierr);
    }
    mapping->info_nfree = PETSC_TRUE;
  } else {
    if (nnodes)  *nnodes  = bn;
    if (count)   *count   = bcount;
    if (indices) *indices = bindices;
  }
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreNodeInfo - Frees the memory allocated by ISLocalToGlobalMappingGetNodeInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameters:
+   mapping - the mapping from local to global indexing
.   nnodes - number of local nodes
.   count - number of neighboring processors per node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *nnodes,PetscInt *count[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingRestoreBlockNodeInfo(mapping,nnodes,count,indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingGetIndices - Get global indices for every local point that is mapped

   Not Collective

   Input Arguments:
. ltog - local to global mapping

   Output Arguments:
. array - array of indices, the length of this array may be obtained with ISLocalToGlobalMappingGetSize()

   Level: advanced

   Notes:
    ISLocalToGlobalMappingGetSize() returns the length the this array

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreIndices(), ISLocalToGlobalMappingGetBlockIndices(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1) {
    *array = ltog->indices;
  } else {
    PetscInt       *jj,k,i,j,n = ltog->n, bs = ltog->bs;
    const PetscInt *ii;
    PetscErrorCode ierr;

    ierr = PetscMalloc1(bs*n,&jj);CHKERRQ(ierr);
    *array = jj;
    k    = 0;
    ii   = ltog->indices;
    for (i=0; i<n; i++)
      for (j=0; j<bs; j++)
        jj[k++] = bs*ii[i] + j;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingRestoreIndices - Restore indices obtained with ISLocalToGlobalMappingGetIndices()

   Not Collective

   Input Arguments:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1 && *array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");

  if (ltog->bs > 1) {
    PetscErrorCode ierr;
    ierr = PetscFree(*(void**)array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingGetBlockIndices - Get global indices for every local block

   Not Collective

   Input Arguments:
. ltog - local to global mapping

   Output Arguments:
. array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  *array = ltog->indices;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingRestoreBlockIndices - Restore indices obtained with ISLocalToGlobalMappingGetBlockIndices()

   Not Collective

   Input Arguments:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (*array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");
  *array = NULL;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingConcatenate - Create a new mapping that concatenates a list of mappings

   Not Collective

   Input Arguments:
+ comm - communicator for the new mapping, must contain the communicator of every mapping to concatenate
. n - number of mappings to concatenate
- ltogs - local to global mappings

   Output Arguments:
. ltogcat - new mapping

   Note: this currently always returns a mapping with block size of 1

   Developer Note: If all the input mapping have the same block size we could easily handle that as a special case

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode ISLocalToGlobalMappingConcatenate(MPI_Comm comm,PetscInt n,const ISLocalToGlobalMapping ltogs[],ISLocalToGlobalMapping *ltogcat)
{
  PetscInt       i,cnt,m,*idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have a non-negative number of mappings, given %D",n);
  if (n > 0) PetscValidPointer(ltogs,3);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(ltogs[i],IS_LTOGM_CLASSID,3);
  PetscValidPointer(ltogcat,4);
  for (cnt=0,i=0; i<n; i++) {
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = PetscMalloc1(cnt,&idx);CHKERRQ(ierr);
  for (cnt=0,i=0; i<n; i++) {
    const PetscInt *subidx;
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    ierr = PetscArraycpy(&idx[cnt],subidx,m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = ISLocalToGlobalMappingCreate(comm,1,cnt,idx,PETSC_OWN_POINTER,ltogcat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      ISLOCALTOGLOBALMAPPINGBASIC - basic implementation of the ISLocalToGlobalMapping object. When ISGlobalToLocalMappingApply() is
                                    used this is good for only small and moderate size problems.

   Options Database Keys:
.   -islocaltoglobalmapping_type basic - select this method

   Level: beginner

.seealso:  ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetType(), ISLOCALTOGLOBALMAPPINGHASH
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Basic(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Basic;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Basic;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Basic;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Basic;
  PetscFunctionReturn(0);
}

/*MC
      ISLOCALTOGLOBALMAPPINGHASH - hash implementation of the ISLocalToGlobalMapping object. When ISGlobalToLocalMappingApply() is
                                    used this is good for large memory problems.

   Options Database Keys:
.   -islocaltoglobalmapping_type hash - select this method


   Notes:
    This is selected automatically for large problems if the user does not set the type.

   Level: beginner

.seealso:  ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetType(), ISLOCALTOGLOBALMAPPINGHASH
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Hash(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Hash;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Hash;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Hash;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Hash;
  PetscFunctionReturn(0);
}


/*@C
    ISLocalToGlobalMappingRegister -  Adds a method for applying a global to local mapping with an ISLocalToGlobalMapping

   Not Collective

   Input Parameters:
+  sname - name of a new method
-  routine_create - routine to create method context

   Notes:
   ISLocalToGlobalMappingRegister() may be called multiple times to add several user-defined mappings.

   Sample usage:
.vb
   ISLocalToGlobalMappingRegister("my_mapper",MyCreate);
.ve

   Then, your mapping can be chosen with the procedural interface via
$     ISLocalToGlobalMappingSetType(ltog,"my_mapper")
   or at runtime via the option
$     -islocaltoglobalmapping_type my_mapper

   Level: advanced

.seealso: ISLocalToGlobalMappingRegisterAll(), ISLocalToGlobalMappingRegisterDestroy(), ISLOCALTOGLOBALMAPPINGBASIC, ISLOCALTOGLOBALMAPPINGHASH

@*/
PetscErrorCode  ISLocalToGlobalMappingRegister(const char sname[],PetscErrorCode (*function)(ISLocalToGlobalMapping))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&ISLocalToGlobalMappingList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingSetType - Builds ISLocalToGlobalMapping for a particular global to local mapping approach.

   Logically Collective on ISLocalToGlobalMapping

   Input Parameters:
+  ltog - the ISLocalToGlobalMapping object
-  type - a known method

   Options Database Key:
.  -islocaltoglobalmapping_type  <method> - Sets the method; use -help for a list
    of available methods (for instance, basic or hash)

   Notes:
   See "petsc/include/petscis.h" for available methods

  Normally, it is best to use the ISLocalToGlobalMappingSetFromOptions() command and
  then set the ISLocalToGlobalMapping type from the options database rather than by using
  this routine.

  Level: intermediate

  Developer Note: ISLocalToGlobalMappingRegister() is used to add new types to ISLocalToGlobalMappingList from which they
  are accessed by ISLocalToGlobalMappingSetType().

.seealso: PCSetType(), ISLocalToGlobalMappingType, ISLocalToGlobalMappingRegister(), ISLocalToGlobalMappingCreate()

@*/
PetscErrorCode  ISLocalToGlobalMappingSetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType type)
{
  PetscErrorCode ierr,(*r)(ISLocalToGlobalMapping);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)ltog,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(ISLocalToGlobalMappingList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)ltog),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested ISLocalToGlobalMapping type %s",type);
  /* Destroy the previous private LTOG context */
  if (ltog->ops->destroy) {
    ierr              = (*ltog->ops->destroy)(ltog);CHKERRQ(ierr);
    ltog->ops->destroy = NULL;
  }
  ierr = PetscObjectChangeTypeName((PetscObject)ltog,type);CHKERRQ(ierr);
  ierr = (*r)(ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscBool ISLocalToGlobalMappingRegisterAllCalled = PETSC_FALSE;

/*@C
  ISLocalToGlobalMappingRegisterAll - Registers all of the local to global mapping components in the IS package.

  Not Collective

  Level: advanced

.seealso:  ISRegister(),  ISLocalToGlobalRegister()
@*/
PetscErrorCode  ISLocalToGlobalMappingRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ISLocalToGlobalMappingRegisterAllCalled) PetscFunctionReturn(0);
  ISLocalToGlobalMappingRegisterAllCalled = PETSC_TRUE;

  ierr = ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGBASIC, ISLocalToGlobalMappingCreate_Basic);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGHASH, ISLocalToGlobalMappingCreate_Hash);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

