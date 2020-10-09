#if !defined(PETSCDMBF_TOPOLOGY_2D_H)
#define PETSCDMBF_TOPOLOGY_2D_H

#include <petscdmbf.h>

typedef struct _p_DM_BF_2D_Topology DM_BF_2D_Topology;

PetscErrorCode DMBF_2D_TopologyCreate(DM,DM_BF_2D_Topology**);
PetscErrorCode DMBF_2D_TopologyDestroy(DM,DM_BF_2D_Topology*);

PetscErrorCode DMBF_2D_TopologyGetConnectivity(DM_BF_2D_Topology*,void*);

#endif /* defined(PETSCDMBF_TOPOLOGY_2D_H) */
