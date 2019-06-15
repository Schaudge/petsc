
#ifndef __VIEWERHDF5IMPL_H
#define __VIEWERHDF5IMPL_H

#include <petscviewerhdf5.h>
#if !defined(H5_VERSION_GE)
/* H5_VERSION_GE was introduced in HDF5 1.8.7, we support >= 1.8.0 */
/* So beware this will automatically 0 for HDF5 1.8.0 - 1.8.6 */
#define H5_VERSION_GE(a,b,c) 0
#endif

#if defined(PETSC_HAVE_HDF5)

typedef struct PetscViewerHDF5GroupList {
  const char       *name;
  struct PetscViewerHDF5GroupList *next;
} PetscViewerHDF5GroupList;

typedef struct {
  char          *filename;
  PetscFileMode btype;
  hid_t         file_id;
  hid_t         dxpl_id;   /* H5P_DATASET_XFER property list controlling raw data transfer (read/write). Properties are modified using H5Pset_dxpl_* functions. */
  PetscInt      timestep;
  PetscViewerHDF5GroupList *groups;
  PetscBool     basedimension2;  /* save vectors and DMDA vectors with a dimension of at least 2 even if the bs/dof is 1 */
  PetscBool     spoutput;  /* write data in single precision even if PETSc is compiled with double precision PetscReal */
} PetscViewer_HDF5;

#endif
#endif
