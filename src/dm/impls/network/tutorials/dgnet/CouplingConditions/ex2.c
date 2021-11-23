static const char help[] = "Test the Characteristic Decomposition of physics provided to the RiemannSolver";

/* 
   
*/
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

int main(int argc,char *argv[])
{
    char              physname[256] = "shallow";
    PetscFunctionList physics = 0;
    MPI_Comm          comm = PETSC_COMM_SELF;
    PetscErrorCode    ierr;
    DGNetwork         dgnet; /* temporarily here as I set up my physics interfaces poorly */
    RiemannSolver     rs;
    PetscInt          numvalues = 10,field,i;
    PetscReal         **u,tol=1e-10;
    PetscRandom       *random;
    PetscViewer       viewer; 

    ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
    ierr = PetscCalloc1(1,&dgnet);CHKERRQ(ierr);
    /* Register physical models to be available on the command line 
       If testing new physics just add your creation function here. */
    ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);

    /* Command Line Options */
    ierr = PetscOptionsBegin(comm,NULL,"Riemann Solver Ex2 Tests Options","");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-numpoints","Number of random points to check the decomposition at","",numvalues,&numvalues,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tol","Tolerance for comparing the Eigenvectors","",tol,&tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    /* Choose the physics from the list of registered models */
    {
        PetscErrorCode (*r)(DGNetwork);
        ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
        if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
        /* Create the physics, will set the number of fields and their names */
        ierr = (*r)(dgnet);CHKERRQ(ierr);
    }

  /* Set up Riemann Solver */
    ierr = RiemannSolverCreate(comm,&rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(rs,dgnet->physics.user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(rs,1,dgnet->physics.dof,dgnet->physics.flux2);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(rs,dgnet->physics.fluxeig);CHKERRQ(ierr);
    ierr = RiemannSolverSetJacobian(rs,dgnet->physics.fluxder);CHKERRQ(ierr);
    ierr = RiemannSolverSetEigBasis(rs,dgnet->physics.eigbasis);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetUp(rs);CHKERRQ(ierr);

    /* Diagnostic Test */
    /* Generate the random number generators to test the eigendecomposition at. Different field variables 
       may have different relevant values (i.e. non-negative pressure/heigh in Euler/SWE) */
    ierr = PetscMalloc1(dgnet->physics.dof,&random);CHKERRQ(ierr);
    for(field=0; field<dgnet->physics.dof; field++)
    {
        ierr = PetscRandomCreate(PETSC_COMM_WORLD,&random[field]);CHKERRQ(ierr);
        ierr = PetscRandomSetInterval(random[field],dgnet->physics.lowbound[field],dgnet->physics.upbound[field]);CHKERRQ(ierr);/* Set a reproducible field for each field variable to generate different numbers for each field */
        ierr = PetscRandomSetSeed(random[field],0x12345678 + 76543*field);CHKERRQ(ierr);
        ierr = PetscRandomSetFromOptions(random[field]);CHKERRQ(ierr);
    }
    /* make the points to test at */
    ierr = PetscCalloc1(numvalues,&u);CHKERRQ(ierr);
    for (i=0; i<numvalues; i++){
        ierr = PetscMalloc1(dgnet->physics.dof,&u[i]);CHKERRQ(ierr);
        for (field=0; field<dgnet->physics.dof; field++) {
            ierr = PetscRandomGetValueReal(random[field],&u[i][field]);CHKERRQ(ierr);
        }
    }
    ierr = PetscViewerCreate(PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
    ierr = RiemannSolverTestEigDecomposition(rs,numvalues,(const PetscReal **)u,tol,NULL,NULL,viewer);CHKERRQ(ierr);
    for (i=0; i<numvalues; i++) {
        ierr = PetscFree(u[i]);CHKERRQ(ierr);
    }
    for (field=0; field<dgnet->physics.dof; field++) {
        ierr = PetscRandomDestroy(&random[field]);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscFree(u);CHKERRQ(ierr);
    ierr = PetscFree(random);CHKERRQ(ierr);
    ierr = DGNetworkDestroyPhysics(dgnet);CHKERRQ(ierr);
    ierr = PetscFree(dgnet);CHKERRQ(ierr);
    ierr = RiemannSolverDestroy(&rs);CHKERRQ(ierr);
    ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
    ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}