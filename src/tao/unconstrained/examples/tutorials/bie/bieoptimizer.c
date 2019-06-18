static char help[] = "\n";

#include <petsctao.h>
#include <petscmatlab.h>

PetscMatlabEngine mengine;

static PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec F, void *ptr)
{
  PetscErrorCode ierr;
  PetscScalar    obj;

  PetscFunctionBegin;
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(mengine,"[obj,F] = ObjectiveAndGradient(X);");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(mengine,1,1,&obj,'obj');CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(mengine,(PetscObject)F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Tao            tao;
  PetscInt       n = 10;
  Vec            X,F;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscMatlabEngineCreate(PETSC_COMM_SELF,NULL,&mengine);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(mengine,"Initialize");CHKERRQ(ierr);

  /* Create the TAO objects and set the type */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);

  /* Create starting point and initialize */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X0");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);

  /* Create residuals vector and set residual function */  
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  ierr = TaoSetResidualRoutine(tao,F,EvaluateResidual,(void*)user);CHKERRQ(ierr);

  /* Solve the problem */
  ierr = TaoSetType(tao,TAOLMVL);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = PetscMatlabEngineEvaluate(mengine,"Finalize");CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: matlab

TEST*/
