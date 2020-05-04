#if !defined (PETSCVIEWERDEF_H)
#define PETSCVIEWERDEF_H

#define PetscViewer type(tPetscViewer)

#define PetscViewerType character*(80)

#define PetscViewerAndFormat PetscFortranAddr
#define PetscViewers PetscFortranAddr
#define PetscFileMode PetscEnum
#define PetscViewerFormat PetscEnum

#define PETSCVIEWERSOCKET       'socket'
#define PETSCVIEWERASCII        'ascii'
#define PETSCVIEWERBINARY       'binary'
#define PETSCVIEWERSTRING       'string'
#define PETSCVIEWERDRAW         'draw'
#define PETSCVIEWERVU           'vu'
#define PETSCVIEWERMATHEMATICA  'mathematica'
#define PETSCVIEWERHDF5         'hdf5'
#define PETSCVIEWERVTK          'vtk'
#define PETSCVIEWERMATLAB       'matlab'
#define PETSCVIEWERSAWS         'saws'
#define PETSCVIEWERGLVIS        'glvis'
#define PETSCVIEWERADIOS        'adios'
#define PETSCVIEWERADIOS2       'adios2'
#define PETSCVIEWEREXODUSII     'exodusii'

#endif
