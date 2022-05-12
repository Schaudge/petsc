#include "bf_2d_vtu.h"

#if defined(PETSC_HAVE_P4EST)

#define DMBFVTKWriteAll               DMBFVTKWriteAll_2D
#define DMBFGetVTKConnectivity        DMBFGetVTKConnectivity_2D
#define DMBFGetVTKCellOffsets         DMBFGetVTKCellOffsets_2D
#define DMBFGetVTKVertexCoordinates   DMBFGetVTKVertexCoordinates_2D
#define DMBFGetVTKCellTypes           DMBFGetVTKCellTypes_2D
#define DMBFGetVTKTreeIDs             DMBFGetVTKTreeIDs_2D
#define DMBFGetVTKQuadRefinementLevel DMBFGetVTKQuadRefinementLevel_2D
#define DMBFGetVTKMPIRank             DMBFGetVTKMPIRank_2D
#define DMBFVTKWritePiece_VTU         DMBFVTKWritePiece_VTU_2D

#include "bf_xd_vtu.c"

#endif /* defined(PETSC_HAVE_P4EST) */