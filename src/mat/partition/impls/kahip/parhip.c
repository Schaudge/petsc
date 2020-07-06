
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

#include <stdbool.h>
#include <parhip_interface.h>

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  int edgecut;         /* number of cuts made (output) */
  int mode;
  bool suppress_output;
} MatPartitioning_ParHIP;

/*
   Uses the ParHIP parallel matrix partitioner to partition the matrix in parallel
*/
static PetscErrorCode MatPartitioningApply_ParHIP(MatPartitioning part, IS *partitioning)
{
  MatPartitioning_ParHIP *parhip = (MatPartitioning_ParHIP*)part->data;
  PetscErrorCode ierr;
  Mat            mat, amat, pmat;
  MPI_Comm       comm;
  PetscBool      flg;
  PetscInt       bs = 1;
  idxtype        *locals;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,2);
  mat = part->adj;
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

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)part),&comm);CHKERRQ(ierr);
  {
    Mat_MPIAdj *adj = (Mat_MPIAdj*)pmat->data;
    idxtype *vtxdist, *xadj, *adjncy, *vwgt = NULL, *adjwgt = NULL;
    int nparts = part->n;
    if (sizeof vtxdist[0] == sizeof pmat->rmap->range[0]) {
      vtxdist = (idxtype*)pmat->rmap->range;
      xadj = (idxtype*)adj->i;
      adjncy = (idxtype*)adj->j;
      vwgt = (idxtype*)part->vertex_weights;
    } else {
      PetscInt nlocal_nodes = pmat->rmap->n, nlocal_edges = adj->i[pmat->rmap->n];
      ierr = PetscMalloc3(nparts+1, &vtxdist, nlocal_nodes+1, &xadj, nlocal_edges, &adjncy);CHKERRQ(ierr);
      for (PetscInt i=0; i<nparts+1; i++) vtxdist[i] = pmat->rmap->range[i];
      for (PetscInt i=0; i<nlocal_nodes+1; i++) xadj[i] = adj->i[i];
      for (PetscInt i=0; i<nlocal_edges; i++) adjncy[i] = adj->j[i];
      if (part->vertex_weights) {
        ierr = PetscMalloc1(nlocal_nodes, &vwgt);CHKERRQ(ierr);
        for (PetscInt i=0; i<nlocal_nodes; i++) vwgt[i] = part->vertex_weights[i];
      }
      if (adj->values) {
        ierr = PetscMalloc1(nlocal_edges, &adjwgt);CHKERRQ(ierr);
        for (PetscInt i=0; i<nlocal_edges; i++) adjwgt[i] = adj->values[i];
      }
    }
    ierr = PetscMalloc1(pmat->rmap->n, &locals);CHKERRQ(ierr);
    double imbalance = 0.; // unused at this time
    int seed = 1;
    ParHIPPartitionKWay(vtxdist, xadj, adjncy, vwgt, adjwgt, &nparts, &imbalance, parhip->suppress_output, seed, parhip->mode, &parhip->edgecut, locals, &comm);
    if (sizeof vtxdist[0] != sizeof pmat->rmap->range[0]) {
      ierr = PetscFree3(vtxdist, xadj, adjncy);CHKERRQ(ierr);
      ierr = PetscFree(vwgt);CHKERRQ(ierr);
      ierr = PetscFree(adjwgt);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Comm_free(&comm);CHKERRQ(ierr);

  if (bs > 1 || sizeof locals[0] != sizeof(PetscInt)) {
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
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,(PetscInt*)locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pmat);CHKERRQ(ierr);
  ierr = MatDestroy(&amat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_ParHIP(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_ParHIP *parhip = (MatPartitioning_ParHIP*)part->data;
  PetscErrorCode         ierr;
  PetscMPIInt            rank;
  PetscBool              iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] Number of cuts found %d\n",rank,parhip->edgecut);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  MatPartitioningParHIPGetEdgeCut - Returns the number of edge cuts in the vertex partition.

  Input Parameter:
. part - the partitioning context

  Output Parameter:
. cut - the edge cut

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParHIPGetEdgeCut(MatPartitioning part, PetscInt *cut)
{
  MatPartitioning_ParHIP *parhip = (MatPartitioning_ParHIP*) part->data;

  PetscFunctionBegin;
  *cut = parhip->edgecut;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_ParHIP(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  MatPartitioning_ParHIP *parhip = (MatPartitioning_ParHIP*) part->data;
  const char *const modelist[] = {"ULTRAFASTMESH", "FASTMESH", "ECOMESH", "ULTRAFASTSOCIAL", "FASTSOCIAL", "ECOSOCIAL", "ParHIPPartitioningMode", "PARHIP_", NULL};
  PetscEnum mode = parhip->mode;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set ParHIP partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mat_partitioning_parhip_mode","ParHIP mode",NULL,modelist,mode,&mode,NULL);CHKERRQ(ierr);
  parhip->mode = mode;
  ierr = PetscOptionsBool("-mat_partitioning_parhip_view","View ParHIP output",NULL,(flag = !parhip->suppress_output),&flag,NULL);CHKERRQ(ierr);
  parhip->suppress_output = !flag;
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningDestroy_ParHIP(MatPartitioning part)
{
  MatPartitioning_ParHIP *parhip = (MatPartitioning_ParHIP*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFree(parhip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGPARHIP - Creates a partitioning context via the external package ParHIP

   Collective

   Input Parameter:
.  part - the partitioning context

   Level: beginner

   Notes:
    See https://parhip.github.io

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_ParHIP(MatPartitioning part)
{
  PetscErrorCode           ierr;
  MatPartitioning_ParHIP *parhip;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&parhip);CHKERRQ(ierr);
  part->data = (void*)parhip;

  parhip->edgecut         = -1; // output
  parhip->mode            = ULTRAFASTMESH;
  parhip->suppress_output = true;

  part->ops->apply          = MatPartitioningApply_ParHIP;
  part->ops->view           = MatPartitioningView_ParHIP;
  part->ops->destroy        = MatPartitioningDestroy_ParHIP;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_ParHIP;
  PetscFunctionReturn(0);
}
