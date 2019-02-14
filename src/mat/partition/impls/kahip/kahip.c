
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

/*
   An interface to kihap's parallel interface.

   Author: Fande Kong (fdkong.jd@gmail.com)
*/

#include <parhip_interface.h>

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  PetscInt  cuts;         /* number of cuts made (output) */
  PetscInt  foldfactor;
  PetscInt  parallel;     /* use parallel partitioner for coarse problem */
  PetscInt  indexing;     /* 0 indicates C indexing, 1 Fortran */
  PetscInt  printout;     /* indicates if one wishes Metis to print info */
  PetscBool repartition;
} MatPartitioning_Kahip;


static PetscErrorCode MatPartitioningApply_Kahip_Private(MatPartitioning part, PetscBool useND, IS *partitioning)
{
  MatPartitioning_Kahip    *kahip = (MatPartitioning_Kahip*)part->data;
  PetscErrorCode           ierr;
  PetscInt                 *locals = NULL;
  Mat                      mat     = part->adj,amat,pmat;
  PetscBool                flg;
  PetscInt                 bs = 1;
  double                   imbalance = 0.03;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (flg) {
    amat = mat;
    ierr = PetscObjectReference((PetscObject)amat);CHKERRQ(ierr);
  } else {
    /* bs indicates if the converted matrix is "reduced" from the original and hence the
       resulting partition results need to be stretched to match the original matrix */
    ierr = MatConvert(mat,MATMPIADJ,MAT_INITIAL_MATRIX,&amat);CHKERRQ(ierr);
    if (amat->rmap->n > 0) bs = mat->rmap->n/amat->rmap->n;
  }
  ierr = MatMPIAdjCreateNonemptySubcommMat(amat,&pmat);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)part));CHKERRQ(ierr);

  if (pmat) {
    MPI_Comm   pcomm,comm;
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    PetscInt   *vtxdist = pmat->rmap->range;
    PetscInt   *xadj    = adj->i;
    PetscInt   *adjncy  = adj->j;
    PetscInt   *NDorder = NULL;
    PetscInt   itmp     = 0,wgtflag=0, nparts=part->n, options[24], i, j;

    ierr = PetscObjectGetComm((PetscObject)pmat,&pcomm);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    /* check that matrix has no diagonal entries */
    {
      PetscInt rstart;
      ierr = MatGetOwnershipRange(pmat,&rstart,NULL);CHKERRQ(ierr);
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          if (adjncy[j] == i+rstart) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row %d has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
        }
      }
    }
#endif

    ierr = PetscMalloc1(pmat->rmap->n,&locals);CHKERRQ(ierr);

    if (adj->values && !part->vertex_weights)
      wgtflag = 1;
    if (part->vertex_weights && !adj->values)
      wgtflag = 2;
    if (part->vertex_weights && adj->values)
      wgtflag = 3;

    if (PetscLogPrintInfo) {itmp = kahip->printout; kahip->printout = 127;}

    /* This sets the defaults */
    options[0] = 0;
    for (i=1; i<24; i++) {
      options[i] = -1;
    }
    /* Duplicate the communicator to be sure that ParMETIS attribute caching does not interfere with PETSc. */
    ierr = MPI_Comm_dup(pcomm,&comm);CHKERRQ(ierr);
    /*PetscStackCallParmetis(ParMETIS_V3_PartKway,((idx_t*)vtxdist,(idx_t*)xadj,(idx_t*)adjncy,(idx_t*)part->vertex_weights,(idx_t*)adj->values,(idx_t*)&wgtflag,(idx_t*)&numflag,(idx_t*)&ncon,(idx_t*)&nparts,tpwgts,ubvec,(idx_t*)options,(idx_t*)&kahip->cuts,(idx_t*)locals,&comm));*/
    ParHIPPartitionKWay((idxtype*)vtxdist, (idxtype*)xadj, (idxtype*)adjncy, (idxtype*)part->vertex_weights, (idxtype*)adj->values,(int*)&nparts, &imbalance, PETSC_FALSE, 0, 0, (int *)&kahip->cuts, (idxtype*)locals,&comm);

    ierr = MPI_Comm_free(&comm);CHKERRQ(ierr);

    if (PetscLogPrintInfo) kahip->printout = itmp;

    if (bs > 1) {
      PetscInt i,j,*newlocals;
      ierr = PetscMalloc1(bs*pmat->rmap->n,&newlocals);CHKERRQ(ierr);
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=0; j<bs; j++) {
          newlocals[bs*i + j] = locals[i];
        }
      }
      ierr = PetscFree(locals);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),bs*pmat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    }
    if (useND) {
      IS ndis;

      if (bs > 1) {
        ierr = ISCreateBlock(PetscObjectComm((PetscObject)part),bs,pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis);CHKERRQ(ierr);
      } else {
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis);CHKERRQ(ierr);
      }
      ierr = ISSetPermutation(ndis);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis);CHKERRQ(ierr);
      ierr = ISDestroy(&ndis);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,partitioning);CHKERRQ(ierr);
    if (useND) {
      IS ndis;

      if (bs > 1) {
        ierr = ISCreateBlock(PetscObjectComm((PetscObject)part),bs,0,NULL,PETSC_COPY_VALUES,&ndis);CHKERRQ(ierr);
      } else {
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,&ndis);CHKERRQ(ierr);
      }
      ierr = ISSetPermutation(ndis);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis);CHKERRQ(ierr);
      ierr = ISDestroy(&ndis);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(&amat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
static PetscErrorCode MatPartitioningApply_Kahip(MatPartitioning part, IS *partitioning)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatPartitioningApply_Kahip_Private(part, PETSC_FALSE, partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Kahip(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Kahip   *kahip = (MatPartitioning_Kahip*)part->data;
  PetscErrorCode           ierr;
  int                      rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (kahip->parallel == 2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using parallel coarse grid partitioner\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using sequential coarse grid partitioner\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %d fold factor\n",kahip->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %d\n",rank,kahip->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



PetscErrorCode MatPartitioningSetFromOptions_Kahip(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  PetscErrorCode ierr;
  /*PetscBool      flag = PETSC_FALSE;*/

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set Kahip partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningDestroy_Kahip(MatPartitioning part)
{
  MatPartitioning_Kahip    *kahip = (MatPartitioning_Kahip*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFree(kahip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   MATPARTITIONINGPARMETIS - Creates a partitioning context via the external package PARMETIS.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_parmetis_coarse_sequential - use sequential PARMETIS coarse partitioner

   Level: beginner

   Notes:
    See http://www-users.cs.umn.edu/~karypis/metis/

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Kahip(MatPartitioning part)
{
  PetscErrorCode           ierr;
  MatPartitioning_Kahip   *kahip;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&kahip);CHKERRQ(ierr);
  part->data = (void*)kahip;

  kahip->cuts       = 0;   /* output variable */
  kahip->foldfactor = 150; /*folding factor */
  kahip->parallel   = 2;   /* use parallel partitioner for coarse grid */
  kahip->indexing   = 0;   /* index numbering starts from 0 */
  kahip->printout   = 0;   /* print no output while running */
  kahip->repartition      = PETSC_FALSE;

  part->ops->apply          = MatPartitioningApply_Kahip;
  part->ops->view           = MatPartitioningView_Kahip;
  part->ops->destroy        = MatPartitioningDestroy_Kahip;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Kahip;
  PetscFunctionReturn(0);
}

