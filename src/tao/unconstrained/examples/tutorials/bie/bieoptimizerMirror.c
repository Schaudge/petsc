static char help[] = "\n";

#include <petsctao.h>
#include <petscmatlab.h>
#include <petscdmda.h>

PetscMatlabEngine mengine;
//Vec               mu;


static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *targ, Vec G, void *ctx)
{
  PetscErrorCode ierr;
  
  
  //PetscScalar    obj;
  
  PetscFunctionBegin;
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(mengine,(PetscObject)X);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)mu,"mu");CHKERRQ(ierr);
//   ierr = PetscMatlabEnginePut(mengine,(PetscObject)mu);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(mengine,"[targ,G] = ObjectiveAndGradientMirror(X);");CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"kjhgf\n" );CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(mengine,1,1,targ,"targ");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"G");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(mengine,(PetscObject)G);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)H,"H");CHKERRQ(ierr);
//   ierr = PetscMatlabEngineGet(mengine,(PetscObject)H);CHKERRQ(ierr);
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
  PetscInt       n = 50; // m = n/2 * (n/2 - 1) /2, ctrl = 0, ctrl2 = 0, k = 0; // nrow = 300, ncol = 50;
  //PetscReal      rc = 1.6;
 // PetscScalar    *c = NULL, *muu;
 // const PetscScalar *x;

  Vec            X,G; //mu; //,CI;
  //Mat            JI;
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

//   ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,ncol,ncol,ncol,NULL,&H);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)H,"H");CHKERRQ(ierr);
  //ierr = TaoSetInequalityConstraintsRoutine(tao, Vec ci, FormConstraints(Tao, Vec, Vec, void*), void *ctx);CHKERRQ(ierr);
   
   ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,NULL);CHKERRQ(ierr);
   
  
  /* Solve the problem */
   ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
   ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  
   

   /* Check for any tao command line options */

   ierr = PetscOptionsHasName(NULL,NULL,"-my_monitor",&viewmat);CHKERRQ(ierr);
   if (viewmat){
     ierr = TaoSetMonitor(tao,My_Monitor,NULL,NULL);CHKERRQ(ierr);
   }
   /* SOLVE THE APPLICATION */
   ierr = TaoSetTolerances(tao,1e-6,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
   ierr = TaoSolve(tao);CHKERRQ(ierr);
  
  

  ierr = TaoView(tao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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
 // ierr = TaoSetHessianRoutine(tao,H, H,FormFunctionGradientHessian,NULL);CHKERRQ(ierr);
 //   TaoSetinEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user);
//   ierr = VecCreateSeq(PETSC_COMM_SELF,nrow,&CI);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)CI,"CI");CHKERRQ(ierr);
//   ierr = TaoSetInequalityConstraintsRoutine(tao,CI,FormInequalityConstraints,NULL);CHKERRQ(ierr);
//   ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,nrow,ncol,ncol,NULL,&JI);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)JI,"JI");CHKERRQ(ierr);
//   ierr = TaoSetJacobianInequalityRoutine(tao,JI,JI,FormInequalityJacobian,NULL);CHKERRQ(ierr);


//   while (ctrl == 0)
//   {
//       
//    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,NULL);CHKERRQ(ierr);
//    
//   
//   /* Solve the problem */
//    ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
//    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
//   
//    
// 
//    /* Check for any tao command line options */
// 
//    ierr = PetscOptionsHasName(NULL,NULL,"-my_monitor",&viewmat);CHKERRQ(ierr);
//    if (viewmat){
//      ierr = TaoSetMonitor(tao,My_Monitor,NULL,NULL);CHKERRQ(ierr);
//    }
//    /* SOLVE THE APPLICATION */
//    ierr = TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
//    ierr = TaoSolve(tao);CHKERRQ(ierr);
//    
//    ierr =TaoGetSolutionVector(tao,&X);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
//    ierr = VecGetArray(mu,&muu);CHKERRQ(ierr);
//    ctrl2 = 0;
//    for (int i = 0; i < n/2; i++)
//    {
//     for (int j = i+1; j < n/2; j++ )
//     { 
//         //ierr = PetscPrintf(PETSC_COMM_SELF,"kjhgf %s %s\n",x[0],x[0] );CHKERRQ(ierr);
//         c[k] = (x[i] - x[j])*((x[i] - x[j])) + (x[i+n/2] - x[j+n/2])*(x[i+n/2] - x[j+n/2]) - (2 * rc + .000001)*(2*rc + .000001);
//         
//         if (c[k] <= 0)
//         {
//           muu[k] = muu[k] * 10.0;
//           ctrl2 = 1;
//         }
//         
//         k = k+1;
//     }
//    }
//    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
//    ierr = VecRestoreArray(mu,&muu);CHKERRQ(ierr);
//    if (ctrl2 == 1)
//    {
//        ctrl = 0;
//    }
//    else
//    {
//        ctrl = 1;
//    }
//   }*/

//   ierr = VecCreateSeq(PETSC_COMM_SELF,m,&mu);CHKERRQ(ierr);
//   ierr = PetscObjectSetName((PetscObject)mu,"mu0");CHKERRQ(ierr);
//   ierr = PetscMatlabEngineGet(mengine,(PetscObject)mu);CHKERRQ(ierr);


// PetscErrorCode FormInequalityConstraints(Tao tao ,Vec X, Vec CI, void *ctx)
// {
//    const PetscScalar *x;
//    PetscScalar       *c;
//    PetscInt           i,j,k=0;
//   
// 
//    VecGetArrayRead(X,&x);
//    VecGetArray(CI,&c);
//    for (i = 0; i < 25; i++)
//    {
//        for (j = i+1; j < 25; j++)
//        {
//            
//             c[k] = (x[i] - x[j])*((x[i] - x[j])) + (x[i+25] - x[j+25])*(x[i+25] - x[j+25]) - (2 * 1.6)*(2*1.6)-.000001;
//             k = k+1;
//        }
//        
//    }
//    VecRestoreArrayRead(X,&x);
//    VecRestoreArray(CI,&c);
//    return(0);
// }
// 
//   PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre,  void *ctx)
// {
//    PetscInt          rows[300];
//    PetscInt          cols[50];
//    PetscScalar       vals[15000] = {0};
//    const PetscScalar *x;
//    //PetscErrorCode    ierr;
//    PetscInt           i,j,k=0;
// 
//    VecGetArrayRead(X,&x);
//    
//    for (i = 0; i < 300; i++)
//    {
//        rows[i] = i;
//    }
//    
//    for (j = 0; j < 50; j++)
//    {  
//             cols[j] = j;
//    }
//    
//    
//    for (i = 0; i<25; i++)
//    {
//        for (j = i+1; j < 25; j++)
//        {    
//                 PetscPrintf(PETSC_COMM_SELF,"blaaaaa/n");
//                 vals[k+i] = 2*(x[i] - x[j]);
//                 vals[k+j] = -2*(x[i] - x[j]);
//                 vals[k+i+25] =2*(x[i+25]-x[j+25]);
//                 vals[k+j+25] = -2*(x[i+25]-x[j+25]);
//                 k = k+50;
//                 
//        }
//    }
//    
//    VecRestoreArrayRead(X,&x);
//    MatSetValues(JI,25,rows,300,cols,vals,INSERT_VALUES);
//    MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY);
//    MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY);
//    return(0);
// }  