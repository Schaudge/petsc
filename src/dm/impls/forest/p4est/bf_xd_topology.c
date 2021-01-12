#if defined(PETSC_HAVE_P4EST)

#include "bf_xd.h"

/* declare "virtual" functions that need to be implemented */
static PetscErrorCode DMBF_XD_ConnectivityCreate(DM,p4est_connectivity_t**);

static PetscErrorCode DMBF_XD_ConnectivityDestroy(DM dm, p4est_connectivity_t *connectivity)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_connectivity_destroy,(connectivity));
  PetscFunctionReturn(0);
}

struct _p_DM_BF_XD_Topology {
  p4est_connectivity_t *connectivity;
  PetscInt             connectivityRefct;
};

PetscErrorCode DMBF_XD_TopologyCreate(DM dm, DM_BF_XD_Topology **topology)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(dm,topology);CHKERRQ(ierr);
  ierr = DMBF_XD_ConnectivityCreate(dm,&(*topology)->connectivity);CHKERRQ(ierr);
  (*topology)->connectivityRefct = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyDestroy(DM dm, DM_BF_XD_Topology *topology)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  topology->connectivityRefct -= 1;
  if (!topology->connectivityRefct) {
    ierr = DMBF_XD_ConnectivityDestroy(dm,topology->connectivity);CHKERRQ(ierr);
  }
  ierr = PetscFree(topology);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyClone(DM_BF_XD_Topology *srcTopology, DM_BF_XD_Topology **trgTopology, DM trgDm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(trgDm,trgTopology);CHKERRQ(ierr);
  (*trgTopology)->connectivity = srcTopology->connectivity;
  srcTopology->connectivityRefct += 1;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyGetConnectivity(DM_BF_XD_Topology *topology, void *connectivity)
{
  PetscFunctionBegin;
  *(void**)connectivity = topology->connectivity;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
