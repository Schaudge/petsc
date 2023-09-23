
#include <petsctao.h>
#include <petscsf.h>
#include <petscdevice.h>
#include <petscdevice_cupm.h>
#include <petscdevice_hip.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/

typedef struct _Rosenbrock {
  PetscInt  bs; // each block of bs variables is one chained multidimensional rosenbrock problem
  PetscInt  i_start, i_end;
  PetscInt  c_start, c_end;
  PetscReal alpha; // condition parameter
} Rosenbrock;

typedef struct _AppCtx *AppCtx;
struct _AppCtx {
  MPI_Comm   comm;
  PetscInt   n; /* dimension */
  PetscInt   n_local;
  PetscInt   n_local_comp;
  Rosenbrock problem;
  Vec        Hvalues; /* vector for writing COO values of this MPI process */
  Vec        gvalues; /* vector for writing gradient values of this mpi process */
  Vec        fvector;
  PetscSF    off_process_scatter;
  PetscSF    gscatter;
  Vec        off_process_values; /* buffer for off-process values if chained */
  PetscBool  test_lmvm;
};
