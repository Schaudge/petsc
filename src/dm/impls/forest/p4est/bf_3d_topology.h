#if !defined(PETSCDMBF_TOPOLOGY_3D_H)
#define PETSCDMBF_TOPOLOGY_3D_H

#include <petscdmbf.h>

typedef struct _p_DM_BF_3D_Topology DM_BF_3D_Topology;

PetscErrorCode DMBF_3D_TopologyCreate(DM,DM_BF_3D_Topology**);
PetscErrorCode DMBF_3D_TopologyDestroy(DM,DM_BF_3D_Topology*);

PetscErrorCode DMBF_3D_TopologyGetConnectivity(DM_BF_3D_Topology*,void*);

#endif /* defined(PETSCDMBF_TOPOLOGY_3D_H) */
