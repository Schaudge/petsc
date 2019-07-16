static char help[] = "\n";

#include <petsctao.h>
#include <petscmatlab.h>
#include <petscdmda.h>

PetscMatlabEngine mengine;



static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *targ, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  //PetscScalar    obj;
  
  PetscFunctionBegin;
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(mengine,"[targ,G] = ObjectiveAndGradient(X);");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(mengine,1,1,targ,"targ");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"G");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(mengine,(PetscObject)G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode My_Monitor(Tao tao, void *ctx)
{
   Vec            X;
   TaoGetSolutionVector(tao,&X);
   VecView(X,PETSC_VIEWER_STDOUT_WORLD);
   return(0);
}


int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscBool      viewmat;
  Tao            tao;
  PetscInt       n = 2;
  Vec            X,G;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscMatlabEngineCreate(PETSC_COMM_SELF,NULL,&mengine);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(mengine,"initialize");CHKERRQ(ierr);

  /* Create the TAO objects and set the type */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);

  /* Create starting point and initialize */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X0");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);

  /* Create residuals vector and set residual function */  
  ierr = VecDuplicate(X,&G);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"G");CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,NULL);CHKERRQ(ierr);
  
  /* Solve the problem */
  ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  
  PetscOptionsHasName(NULL,NULL,"-my_monitor",&viewmat);
   if (viewmat){
     TaoSetMonitor(tao,My_Monitor,NULL,NULL);
   }

   /* Check for any tao command line options */
 // TaoSetFromOptions(tao);

   /* SOLVE THE APPLICATION */
  TaoSolve(tao);

  TaoView(tao,PETSC_VIEWER_STDOUT_WORLD);
  //ierr = TaoSolve(tao);CHKERRQ(ierr);
  
  //ierr = PetscMatlabEngineEvaluate(mengine,"Finalize");CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: matlab

TEST*/
