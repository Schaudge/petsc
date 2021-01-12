#include "bf_3d_topology.h"

#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h> /* convert to p8est for 3D domains */

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_ConnectivityCreate      DMBF_3D_ConnectivityCreate
#define DMBF_XD_ConnectivityDestroy     DMBF_3D_ConnectivityDestroy

#define DM_BF_XD_Topology               DM_BF_3D_Topology
#define _p_DM_BF_XD_Topology            _p_DM_BF_3D_Topology

#define DMBF_XD_TopologyCreate          DMBF_3D_TopologyCreate
#define DMBF_XD_TopologyDestroy         DMBF_3D_TopologyDestroy
#define DMBF_XD_TopologyClone           DMBF_3D_TopologyClone
#define DMBF_XD_TopologyGetConnectivity DMBF_3D_TopologyGetConnectivity

/* include generic functions */
#include "bf_xd_topology.c"

static PetscErrorCode DMBF_3D_ConnectivityCreate(DM dm, p4est_connectivity_t **connectivity)
{
  PetscFunctionBegin;
  //TODO
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
