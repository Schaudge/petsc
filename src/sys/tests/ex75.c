const char help[] = "Test MPI_Allreduce() forever";

#include <petsc.h>

int main(int argc, char **argv)
{
  PetscInt a,b;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  while (PETSC_TRUE) {
    PetscCall(PetscSleep(1.0));
    PetscCall(MPIU_Allreduce(&a,&b,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  }
  PetscCall(PetscFinalize());
  return 0;
}
