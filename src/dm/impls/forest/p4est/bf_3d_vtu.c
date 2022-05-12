#include "bf_3d_vtu.h"

#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h> /* convert to p8est for 3D domains */

#define DMBFVTKWriteAll               DMBFVTKWriteAll_3D
#define DMBFGetVTKConnectivity        DMBFGetVTKConnectivity_3D
#define DMBFGetVTKCellOffsets         DMBFGetVTKCellOffsets_3D
#define DMBFGetVTKVertexCoordinates   DMBFGetVTKVertexCoordinates_3D
#define DMBFGetVTKCellTypes           DMBFGetVTKCellTypes_3D
#define DMBFGetVTKTreeIDs             DMBFGetVTKTreeIDs_3D
#define DMBFGetVTKQuadRefinementLevel DMBFGetVTKQuadRefinementLevel_3D
#define DMBFGetVTKMPIRank             DMBFGetVTKMPIRank_3D
#define DMBFVTKWritePiece_VTU         DMBFVTKWritePiece_VTU_3D

#include "bf_xd_vtu.c"

#endif /* defined(PETSC_HAVE_P4EST) */
