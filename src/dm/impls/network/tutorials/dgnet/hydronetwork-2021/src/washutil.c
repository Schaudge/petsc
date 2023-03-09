/*
   Utility subroutines for WASH
*/
#include "wash.h"

static PetscErrorCode VecArrayPrint_private_Wash(PetscInt n, const PetscScalar *xv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    u;
  PetscReal      tol = 1.e-8;

  PetscFunctionBegin;
  for (i = 0; i < n; i += 2) {
    if (PetscAbsScalar(xv[i + 1]) < tol) {
      u = 0.0;
    } else u = xv[i] / xv[i + 1];

    ierr = PetscPrintf(PETSC_COMM_SELF, "  %10.2f %10.2f %10.2f\n", xv[i], xv[i + 1], u);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Network_Seq_Wash(Wash wash, DM networkdm, Vec X)
{
  PetscErrorCode     ierr;
  PetscInt           e, v, Start, End, offset, id, type;
  const PetscScalar *xv;
  River              river;
  Junction           junction;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &xv);
  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "        Q          H          U(Q/H)\n");
  CHKERRQ(ierr);

  /* iterate over edges */
  ierr = DMNetworkGetEdgeRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  for (e = Start; e < End; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &offset);
    CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalEdgeIndex(networkdm, e, &id);
    CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF, "  River %d\n", id);
    CHKERRQ(ierr);
    ierr = VecArrayPrint_private_Wash(2 * river->ncells, xv + offset);
    CHKERRQ(ierr);
  }

  /* iterate over vertices */
  ierr = DMNetworkGetVertexRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  for (v = Start; v < End; v++) {
    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction, NULL);
    CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm, v, ALL_COMPONENTS, &offset);
    CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm, v, &id);
    CHKERRQ(ierr);

    if (junction->type == JUNCTION) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Junction %d:\n", id);
      CHKERRQ(ierr);
    } else if (junction->type == RESERVOIR) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Reservoir %d:\n", id);
      CHKERRQ(ierr);
    } else if (junction->type == VALVE) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Valve %d:\n", id);
      CHKERRQ(ierr);
    } else if (junction->type == INFLOW) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  Inflow %d:\n", id);
      CHKERRQ(ierr);
    }
    ierr = VecArrayPrint_private_Wash(2, xv + offset);
    CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &xv);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Customized from VecView_Network() */
PetscErrorCode WashVecView(Wash wash, Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  DM             dm = wash->dm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
  CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecView_Network_Seq_Wash(wash, dm, v);
    CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not done yet");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Display Junction information
*/
PetscErrorCode WashJunctionView(Wash wash)
{
  PetscErrorCode  ierr;
  DM              dm = wash->dm;
  PetscInt        i, gv, subnet, nv, ne, nsubnet = wash->nsubnet;
  const PetscInt *vtx, *edges;
  Junction        junction;
  MPI_Comm        comm;
  PetscMPIInt     rank;
  PetscBool       ghost;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(comm, "[%d] Vertex:\n", rank);
  for (subnet = 0; subnet < nsubnet; subnet++) {
    ierr = DMNetworkGetSubnetwork(dm, subnet, &nv, &ne, &vtx, &edges);
    CHKERRQ(ierr);
    for (i = 0; i < nv; i++) {
      ierr = DMNetworkGetComponent(dm, vtx[i], 0, NULL, (void **)&junction, NULL);
      CHKERRQ(ierr);
      ierr = DMNetworkGetGlobalVertexIndex(dm, vtx[i], &gv);
      CHKERRQ(ierr);
      ierr = DMNetworkIsGhostVertex(dm, vtx[i], &ghost);
      CHKERRQ(ierr);

      switch (junction->type) {
      case 0:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: NONE; ghost %d\n", subnet, gv, ghost);
        break;
      case 1:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: JUNCTION; ghost %d\n", subnet, gv, ghost);
        break;
      case 2:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: RESERVOIR; ghost %d\n", subnet, gv, ghost);
        break;
      case 3:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: VALVE; ghost %d\n", subnet, gv, ghost);
        break;
      case 4:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: DEMAND; ghost %d\n", subnet, gv, ghost);
        break;
      case 5:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: INFLOW; ghost %d\n", subnet, gv, ghost);
        break;
      case 6:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: STAGE; ghost %d\n", subnet, gv, ghost);
        break;
      case 7:
        ierr = PetscSynchronizedPrintf(comm, "  subnet[%d].%d, type: TANK; ghost %d\n", subnet, gv, ghost);
        break;
      default:
        PetscCheck(junction->type >= 0 && junction->type < 8, PETSC_COMM_SELF, PETSC_ERR_SUP, "junction->type %" PetscInt_FMT " not done yet", junction->type);
      }
    }
  }
  ierr = PetscSynchronizedFlush(comm, PETSC_STDOUT);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}
