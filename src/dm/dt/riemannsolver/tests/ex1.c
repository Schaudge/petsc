static char help[] = "Tests repeated RiemannSolverSetType().\n\n";

#include <petscriemannsolver.h>

/* Dummy functions to force the RiemannSolver to generate all internal structures */
static PetscErrorCode EigBasis_Dummy(void *ctx,const PetscReal *u, Mat mat)
{
    PetscFunctionReturn(0);
}

static PetscErrorCode RoeMat_Dummy(void *ctx,const PetscReal *uL,const PetscReal *uR, Mat *roe)
{
    PetscFunctionReturn(0);
} 

static PetscErrorCode RoeAvg_Dummy(void *ctx,const PetscReal *uL,const PetscReal *uR,PetscReal *roeavg)
{
    PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       dim = 1, numfields = 5; 
  RiemannSolver  rs; 

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create RiemannSolver */
  ierr = RiemannSolverCreate(PETSC_COMM_SELF,&rs);CHKERRQ(ierr);
  ierr = RiemannSolverSetFluxDim(rs,dim,numfields);CHKERRQ(ierr);
  ierr = RiemannSolverSetEigBasis(rs,EigBasis_Dummy);CHKERRQ(ierr);
  ierr = RiemannSolverSetRoeMatrixFunct(rs,RoeMat_Dummy);CHKERRQ(ierr);
  ierr = RiemannSolverSetRoeAvgFunct(rs,RoeAvg_Dummy);CHKERRQ(ierr);
  ierr = RiemannSolverSetType(rs,"lax");CHKERRQ(ierr); 
  ierr = RiemannSolverSetUp(rs);CHKERRQ(ierr);

    /* Add a different type here when I have another implementation coded up */
  ierr = RiemannSolverSetType(rs,"lax");CHKERRQ(ierr); 
  ierr = RiemannSolverSetUp(rs);CHKERRQ(ierr);

  ierr = RiemannSolverDestroy(&rs);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:

TEST*/
