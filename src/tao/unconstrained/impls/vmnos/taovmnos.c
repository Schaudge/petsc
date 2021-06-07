#include <../src/tao/unconstrained/impls/vmnos/taovmnos.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petscksp.h>
#include <petsc/private/petscimpl.h>


static PetscBool  cited      = PETSC_FALSE;
static const char citation[] =
  "@inproceedings{pedregosa2018adaptive,\n"
  "   title={Adaptive three operator splitting},\n"
  "   author={Pedregosa, Fabian and Gidel, Gauthier},\n"
  "   booktitle={International Conference on Machine Learning},\n"
  "   pages={4085--4094},\n"
  "   year={2018},\n"
  "   organization={PMLR}\n"
  "}  \n";

static PetscErrorCode Adapt_Shell_Mult(Mat H, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  TAO_VMNOS      *vmnos;

  PetscFunctionBegin;
  ierr = MatShellGetContext(H,&vmnos);CHKERRQ(ierr);
  ierr = VecCopy(X,Y);CHKERRQ(ierr);
  ierr = VecScale(Y,vmnos->stepsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Adapt_Shell_Solve(Mat H, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  TAO_VMNOS      *vmnos;

  PetscFunctionBegin;
  ierr = MatShellGetContext(H,&vmnos);CHKERRQ(ierr);
  ierr = VecCopy(X,Y);CHKERRQ(ierr);
  ierr = VecScale(Y,1/(vmnos->stepsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode TaoVMNOSSetUpdateType_VMNOS(Tao tao, TaoVMNOSUpdateType type)
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;

  PetscFunctionBegin;
  vmnos->vm_update = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoVMNOSGetUpdateType_VMNOS(Tao tao, TaoVMNOSUpdateType *type)
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;

  PetscFunctionBegin;
  *type = vmnos->vm_update;
  PetscFunctionReturn(0);
}

/* Solve f(x) + g(x) + h(x) */
static PetscErrorCode TaoSolve_VMNOS(Tao tao)
{
  TAO_VMNOS      *vmnos = (TAO_VMNOS*)tao->data;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      fk,incrf1dot,fk_xzf1grad,xzH,certificate,Liptemp,fgh;
  PetscReal      ls_tol,quot,step2,rhs;


  PetscFunctionBegin;

  /* VM Update Options */
  switch (vmnos->vm_update) {
  case TAO_VMNOS_ADAPTIVE:
    break;
  case TAO_VMNOS_BB:
    break;
  }

  ierr        = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  tao->reason = TAO_CONTINUE_ITERATING;

  /* subtao[1]->solution = zk */
  /* zk = prox2(xk) */
  ierr = VecCopy(tao->solution,vmnos->zk_old);CHKERRQ(ierr);
  ierr = TaoSolve(vmnos->subtaos[1]);CHKERRQ(ierr);

  ierr = (*vmnos->ops->f1obj)(vmnos->f1subtao,vmnos->zk,&fk,vmnos->f1objP);CHKERRQ(ierr);
  ierr = (*vmnos->ops->f1grad)(vmnos->f1subtao,vmnos->zk,vmnos->f1grad,vmnos->f1gradP);CHKERRQ(ierr);

  /* xk = prox1(z- f1grad). It should be noted that xk is se to z- f1grad, and new xk is computed */
  ierr = VecCopy(tao->solution,vmnos->xk_old);CHKERRQ(ierr);
  ierr = VecCopy(vmnos->f1grad,vmnos->temp);CHKERRQ(ierr);
  ierr = VecScale(vmnos->temp,vmnos->stepsize);CHKERRQ(ierr);
  ierr = VecWAXPY(vmnos->xk,-1.,vmnos->temp,vmnos->zk);CHKERRQ(ierr);
  ierr = TaoSolve(vmnos->subtaos[0]);CHKERRQ(ierr);


  while (tao->reason == TAO_CONTINUE_ITERATING) {
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ierr = VecCopy(vmnos->f1grad,vmnos->f1grad_old);CHKERRQ(ierr);
    ierr = (*vmnos->ops->f1obj)(vmnos->f1subtao,vmnos->zk,&fk,vmnos->f1objP);CHKERRQ(ierr);
    ierr = (*vmnos->ops->f1grad)(vmnos->f1subtao,vmnos->zk,vmnos->f1grad,vmnos->f1gradP);CHKERRQ(ierr);
    if ((vmnos->vm_update == TAO_VMNOS_BB) && (tao->niter > 0)) {
      ierr = MatLMVMUpdate(vmnos->vm,vmnos->xk,vmnos->f1grad);CHKERRQ(ierr);
    }

    /* TODO Lip Hinv bound check */
    ierr = VecCopy(vmnos->xk,vmnos->xk_old);CHKERRQ(ierr);

    /* xk = prox1(zk- (1/H)*(uk + f1grad)) */
    ierr = VecWAXPY(vmnos->temp,1.,vmnos->uk,vmnos->f1grad);CHKERRQ(ierr);

    if ((vmnos->vm_update == TAO_VMNOS_BB) && (tao->niter > 1)) {
      ierr = MatMult(vmnos->vm,vmnos->temp,vmnos->temp2);CHKERRQ(ierr);
      ierr = VecWAXPY(vmnos->subtaos[0]->solution,-1.,vmnos->temp2,vmnos->zk);CHKERRQ(ierr);
    } else {
      ierr = VecScale(vmnos->temp,vmnos->stepsize);CHKERRQ(ierr);
      ierr = VecWAXPY(vmnos->subtaos[0]->solution,-1.,vmnos->temp,vmnos->zk);CHKERRQ(ierr);
    }
    ierr = TaoSolve(vmnos->subtaos[0]);CHKERRQ(ierr);


    /* temp = xk - zk */
    ierr = VecWAXPY(vmnos->temp,-1,vmnos->zk,vmnos->xk);CHKERRQ(ierr);
    ierr = VecNorm(vmnos->temp,NORM_2,&vmnos->resnorm);CHKERRQ(ierr);

    if (vmnos->linesearch && (vmnos->resnorm > 1.E-7)) {
      for (i=0; i<vmnos->lniter; i++) {
        /* incrf1dot = dot(f1grad, x-z) */
        ierr        = VecDot(vmnos->temp,vmnos->f1grad,&incrf1dot);CHKERRQ(ierr);
        fk_xzf1grad = fk + incrf1dot;
        /* Adaptive */
        if (vmnos->vm_update == TAO_VMNOS_ADAPTIVE) rhs = fk_xzf1grad + (vmnos->resnorm*vmnos->resnorm) / (2*vmnos->stepsize);
        else if (vmnos->vm_update == TAO_VMNOS_BB) {
          ierr = MatSolve(vmnos->vm,vmnos->temp,vmnos->temp2);CHKERRQ(ierr);
          ierr = VecDot(vmnos->temp,vmnos->temp2,&xzH);CHKERRQ(ierr);
          rhs  = fk_xzf1grad + 0.5*xzH;
        }
        ierr = (*vmnos->ops->f1obj)(vmnos->f1subtao,vmnos->xk,&fk,vmnos->f1objP);CHKERRQ(ierr);

        ls_tol = fk - rhs;
        if (ls_tol <= vmnos->ls_eps) break;
        else {
          if (vmnos->vm_update == TAO_VMNOS_ADAPTIVE) vmnos->stepsize *= vmnos->bs_factor;
          else {
            ierr = MatScale(vmnos->vm, 1/(vmnos->bs_factor));CHKERRQ(ierr);
          }
        }
      }
    }

    ierr = VecCopy(vmnos->zk,vmnos->zk_old);CHKERRQ(ierr);

    /* zk = prox2(xk + (1/H)*uk) */
    if ((vmnos->vm_update == TAO_VMNOS_BB) && (tao->niter >= 1)) {
      ierr = MatMult(vmnos->vm,vmnos->uk,vmnos->temp);CHKERRQ(ierr);
    } else {
      ierr = MatMult(vmnos->vm,vmnos->uk,vmnos->temp);CHKERRQ(ierr);
    }
    ierr = VecWAXPY(vmnos->subtaos[1]->solution,1.,vmnos->xk,vmnos->temp);CHKERRQ(ierr);
    ierr = TaoSolve(vmnos->subtaos[1]);CHKERRQ(ierr);

    /* uk += (xk - zk)*H */
    ierr = VecWAXPY(vmnos->temp,-1.,vmnos->zk,vmnos->xk);CHKERRQ(ierr);
    if (vmnos->vm_update == TAO_VMNOS_ADAPTIVE) {
      ierr        = VecScale(vmnos->temp,1/(vmnos->stepsize));CHKERRQ(ierr);
      ierr        = VecAXPY(vmnos->uk,1.,vmnos->temp);CHKERRQ(ierr);
      certificate = vmnos->resnorm / vmnos->stepsize;
    } else {
      ierr        = MatSolve(vmnos->vm,vmnos->temp,vmnos->temp2);CHKERRQ(ierr);
      ierr        = VecAXPY(vmnos->uk,1.,vmnos->temp2);CHKERRQ(ierr);
      certificate = vmnos->resnorm / vmnos->stepsize;
    }

    if (vmnos->Lip == 0) vmnos->stepsize *= 1.02;
    else {
      quot            = vmnos->Lip*vmnos->Lip;
      step2           = vmnos->stepsize * vmnos->stepsize;
      Liptemp         = PetscSqrtReal(step2 + ((2*vmnos->stepsize)/quot)*(-vmnos->ls_tol));
      vmnos->stepsize = PetscMin(Liptemp, vmnos->stepsize*1.02);
    }

    tao->niter++;

    ierr = TaoComputeObjective(tao,vmnos->xk,&fgh);CHKERRQ(ierr);
    if (tao->niter == 1) certificate = 1000.;

    ierr = TaoLogConvergenceHistory(tao,fgh,certificate,0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,fgh,certificate,0,1.0);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }

  /* Probably need something like this*/
  for (i=0; i<vmnos->proxnum; i++) {
    ierr = PetscObjectCompose((PetscObject)vmnos->subtaos[i],"TaoGetVMNOSParentTao_VMNOS", NULL);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)vmnos->subtaos[i],"TaoVMNOSGetVMMat_VMNOS", NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoVMNOSSetUpdateType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoVMNOSGetUpdateType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_VMNOS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_VMNOS      *vmnos = (TAO_VMNOS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"VMNOS problem that solves f(x) + g(x) + h(x). ");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_vmnos_linesearch","Sets linesearch routine for inverse diagonal Hessian","",vmnos->linesearch,&vmnos->linesearch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_vmnos_stepsize","Starting stepsize for line search","",vmnos->stepsize,&vmnos->stepsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_vmnos_bb_weight","Weight for calculating BB VM metric","",vmnos->mu,&vmnos->mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_vmnos_bb_linesearch_iteration","Linesearch iteration for BB metric.","",vmnos->lniter,&vmnos->lniter,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_vmnos_vm_update","Variable Metric update policy","TaoVMNOSUpdateType",
                          TaoVMNOSUpdateTypes,(PetscEnum)vmnos->vm_update,(PetscEnum*)&vmnos->vm_update,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_VMNOS(Tao tao,PetscViewer viewer)
{
  TAO_VMNOS      *vmnos = (TAO_VMNOS*)tao->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  for (i=0; i<vmnos->proxnum; i++) {
    ierr = TaoView(vmnos->subtaos[i],viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_VMNOS(Tao tao)
{
  TAO_VMNOS      *vmnos = (TAO_VMNOS*)tao->data;
  PetscErrorCode ierr;
  PetscInt       n,N,i;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);

  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }

  if (!vmnos->zk) {
    ierr = VecDuplicate(tao->solution,&vmnos->zk);CHKERRQ(ierr);
    ierr = VecSet(vmnos->zk,0.0);CHKERRQ(ierr);
  }
  if (!vmnos->zk_old) {
    ierr = VecDuplicate(tao->solution,&vmnos->zk_old);CHKERRQ(ierr);
    ierr = VecSet(vmnos->zk_old,0.0);CHKERRQ(ierr);
  }
  if (!vmnos->f1grad) {
    ierr = VecDuplicate(tao->solution,&vmnos->f1grad);CHKERRQ(ierr);
    ierr = VecSet(vmnos->f1grad,0.0);CHKERRQ(ierr);
  }
  if (!vmnos->f1grad_old) {
    ierr = VecDuplicate(tao->solution,&vmnos->f1grad_old);CHKERRQ(ierr);
    ierr = VecSet(vmnos->f1grad_old,0.0);CHKERRQ(ierr);
  }
  vmnos->xk = tao->solution;
  if (!vmnos->xk_old) {
    ierr = VecDuplicate(tao->solution,&vmnos->xk_old);CHKERRQ(ierr);
    ierr = VecSet(vmnos->xk_old,0.0);CHKERRQ(ierr);
  }

  /* VM Mat */
  if (vmnos->vm_update == TAO_VMNOS_BB) {
    ierr = MatCreate(PetscObjectComm((PetscObject)tao),&vmnos->vm);CHKERRQ(ierr);
    ierr = MatSetType(vmnos->vm,MATLMVMDIAGBB);CHKERRQ(ierr);
    ierr = MatSetSizes(vmnos->vm, n, n, N, N);CHKERRQ(ierr);
    ierr = MatSetUp(vmnos->vm);CHKERRQ(ierr);
    ierr = MatLMVMReset(vmnos->vm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatLMVMSetHistorySize(vmnos->vm, 0);CHKERRQ(ierr);
    ierr = MatLMVMAllocate(vmnos->vm,vmnos->xk,vmnos->f1grad);CHKERRQ(ierr);
  } else {
    ierr = MatCreateShell(PetscObjectComm((PetscObject)tao),n,n,N,N,(void*)vmnos,&vmnos->vm);CHKERRQ(ierr);
    ierr = MatShellSetOperation(vmnos->vm,MATOP_MULT,(void (*)(void))Adapt_Shell_Mult);CHKERRQ(ierr);
    ierr = MatShellSetOperation(vmnos->vm,MATOP_SOLVE,(void (*)(void))Adapt_Shell_Solve);CHKERRQ(ierr);
  }


  ierr = PetscMalloc1(vmnos->proxnum,&vmnos->subtaos);
  /* TODO currently, only support 2 prox operators... */
  for (i=0; i< vmnos->proxnum; i++) {
    char buffer[256];
    ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&(vmnos->subtaos[i]));CHKERRQ(ierr);
    ierr = PetscSNPrintf(buffer,256,"f%d_",i);CHKERRQ(ierr);
    ierr = TaoSetOptionsPrefix(vmnos->subtaos[i],buffer);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)vmnos->subtaos[i],(PetscObject)tao,1);CHKERRQ(ierr);
    ierr = TaoSetType(vmnos->subtaos[i],TAOSHELL);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)vmnos->subtaos[i],"TaoGetVMNOSParentTao_VMNOS", (PetscObject) tao);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)vmnos->subtaos[i],"TaoVMNOSGetVMMat_VMNOS", (PetscObject) vmnos->vm);CHKERRQ(ierr);
  }


  for (i=0; i<vmnos->proxnum; i++) {
    ierr = TaoSetFromOptions(vmnos->subtaos[i]);CHKERRQ(ierr);
  }

  ierr = TaoSetInitialVector(vmnos->subtaos[0], vmnos->xk);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(vmnos->subtaos[1], vmnos->zk);CHKERRQ(ierr);

  for (i=0; i<vmnos->proxnum; i++) {
    ierr = TaoSetUp(vmnos->subtaos[i]);CHKERRQ(ierr);
  }

  if (!vmnos->uk) {
    ierr = VecDuplicate(tao->solution,&vmnos->uk);CHKERRQ(ierr);
    ierr = VecSet(vmnos->uk,0.0);CHKERRQ(ierr);
  }
  if (!vmnos->temp) {
    ierr = VecDuplicate(tao->solution,&vmnos->temp);CHKERRQ(ierr);
    ierr = VecSet(vmnos->temp,0.0);CHKERRQ(ierr);
  }
  if (!vmnos->temp2) {
    ierr = VecDuplicate(tao->solution,&vmnos->temp2);CHKERRQ(ierr);
    ierr = VecSet(vmnos->temp2,0.0);CHKERRQ(ierr);
  }

  /* Save changed tao tolerance for adaptive tolerance */
  if (tao->gatol_changed) vmnos->gatol_vmnos = tao->gatol;
  if (tao->catol_changed) vmnos->catol_vmnos = tao->catol;

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_VMNOS(Tao tao)
{
  TAO_VMNOS      *vmnos = (TAO_VMNOS*)tao->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  vmnos->xk = NULL;

  ierr = VecDestroy(&vmnos->zk);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->zk_old);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->xk);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->xk_old);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->uk);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->f1grad);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->f1grad_old);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->temp);CHKERRQ(ierr);
  ierr = VecDestroy(&vmnos->temp2);CHKERRQ(ierr);
  ierr = MatDestroy(&vmnos->vm);CHKERRQ(ierr);
  ierr = TaoDestroy(&vmnos->f1subtao);CHKERRQ(ierr);

  for (i=0; i<vmnos->proxnum; i++) {
    ierr = TaoDestroy(&vmnos->subtaos[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(vmnos->subtaos);CHKERRQ(ierr);

  vmnos->parent = NULL;
  ierr          = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC

  TAOVMNOS - Variable Metric N-Operator Split

  Options Database Keys:
+ -tao_vmnos_regularizer_coefficient        - regularizer constant (default 1.e-6)
. -tao_vmnos_spectral_penalty               - Constant for Augmented Lagrangian term (default 1.)

  Level: beginner

.seealso: TaoVMNOSSetSeparableOperatorCount(), TaoVMNOSGetUpdateType(), TaoVMNOSSetUpdateType(),
          TaoGetVMNOSParentTao(), TaoVMNOSSetF1ObjectiveAndGradientRoutine()

M*/

PETSC_EXTERN PetscErrorCode TaoCreate_VMNOS(Tao tao)
{
  TAO_VMNOS      *vmnos;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&vmnos);CHKERRQ(ierr);

  tao->ops->destroy        = TaoDestroy_VMNOS;
  tao->ops->setup          = TaoSetUp_VMNOS;
  tao->ops->setfromoptions = TaoSetFromOptions_VMNOS;
  tao->ops->view           = TaoView_VMNOS;
  tao->ops->solve          = TaoSolve_VMNOS;

  tao->data          = (void*)vmnos;
  vmnos->mu          = 1.;
  vmnos->parent      = tao;
  vmnos->vm_update   = TAO_VMNOS_ADAPTIVE;
  vmnos->tol         = PETSC_SMALL;
  vmnos->gatol_vmnos = 1e-8;
  vmnos->stepsize    = 1.;
  vmnos->catol_vmnos = 0;
  vmnos->ls_eps      = 2.22E-16;
  vmnos->ls_tol      = 1.E-8;
  vmnos->lniter      = 1000;
  vmnos->bs_factor   = 0.7;
  vmnos->ops->f1obj  = NULL;
  vmnos->ops->f1grad = NULL;
  vmnos->f1objP      = NULL;
  vmnos->f1gradP     = NULL;
  vmnos->linesearch  = PETSC_TRUE;

  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&vmnos->f1subtao);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(vmnos->f1subtao,"f1subtao__");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)vmnos->f1subtao,(PetscObject)tao,1);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoVMNOSSetUpdateType_C",TaoVMNOSSetUpdateType_VMNOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoVMNOSGetUpdateType_C",TaoVMNOSGetUpdateType_VMNOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TaoVMNOSSetF1ObjectiveRoutine - Sets the user-defined F1 objective call-back function

   Collective on tao

   Input Parameters:
   + tao - the Tao context
   . func - function pointer for the F1 function value evaluation
   - ctx - user context for the F1 function

   Level: advanced

.seealso: TAOVMNOS

@*/
PetscErrorCode TaoVMNOSSetF1ObjectiveRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, PetscReal*, void*), void *ctx)
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  vmnos->f1objP     = ctx;
  vmnos->ops->f1obj = func;
  PetscFunctionReturn(0);
}

/*@C
   TaoVMNOSSetF1GradientRoutine - Sets the user-defined F1 gradient call-back function

   Collective on tao

   Input Parameters:
   + tao - the Tao context
   . func - function pointer for the F1 gradient evaluation
   - ctx - user context for the F1 function

   Level: advanced

.seealso: TAOVMNOS

@*/
PetscErrorCode TaoVMNOSSetF1GradientRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, Vec, void*), void *ctx)
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  vmnos->f1gradP     = ctx;
  vmnos->ops->f1grad = func;
  PetscFunctionReturn(0);
}
/*@
   TaoGetVMNOSParentTao - Gets pointer to parent VMNOS tao, used by inner subsolver.

   Collective on tao

   Input Parameter:
   . tao - the Tao context

   Output Parameter:
   . vmnos_tao - the parent Tao context

   Level: advanced

.seealso: TAOVMNOS

@*/
PetscErrorCode TaoGetVMNOSParentTao(Tao tao, Tao *vmnos_tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)tao,"TaoGetVMNOSParentTao_VMNOS", (PetscObject*) vmnos_tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoVMNOSSetUpdateType - Set update routine for VMNOS routine

  Not Collective

  Input Parameter:
+ tao  - the Tao context
- type - spectral parameter update type

  Level: intermediate

.seealso: TaoVMNOSGetUpdateType(), TaoVMNOSUpdateType, TAOVMNOS
@*/
PetscErrorCode TaoVMNOSSetUpdateType(Tao tao, TaoVMNOSUpdateType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,type,2);
  ierr = PetscTryMethod(tao,"TaoVMNOSSetUpdateType_C",(Tao,TaoVMNOSUpdateType),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TaoVMNOSGetUpdateType - Gets the type of spectral penalty update routine for VMNOS

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameter:
.  type - the type of spectral penalty update routine

   Level: intermediate

.seealso: TaoVMNOSSetUpdateType(), TaoVMNOSUpdateType, TAOVMNOS
@*/
PetscErrorCode TaoVMNOSGetUpdateType(Tao tao, TaoVMNOSUpdateType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscUseMethod(tao,"TaoVMNOSGetUpdateType_C",(Tao,TaoVMNOSUpdateType*),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoVMNOSSetSeparableOperatorCount - Set number of subsolvers routine for VMNOS routine

  Not Collective

  Input Parameter:
+ tao  - the Tao context
- num  - Number of subsolver operators

  Level: intermediate

.seealso: TAOVMNOS
@*/
PetscErrorCode TaoVMNOSSetSeparableOperatorCount(Tao tao, PetscInt num)
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  vmnos->proxnum = num;
  PetscFunctionReturn(0);
}

/*@
  TaoVMNOSGetSubsolvers - Get the pointer to the i-th subsolver inside VMNOS

  Collective on Tao

  Input Parameter:
.  tao - the Tao solver context

  Output Parameter:
.  sub - List of Tao subsolver contexts

  Level: advanced

.seealso: TAOVMNOS

@*/
PetscErrorCode TaoVMNOSGetSubsolvers(Tao tao, Tao sub[])
{
  TAO_VMNOS *vmnos = (TAO_VMNOS*)tao->data;
  PetscInt  i;

  PetscFunctionBegin;
  /* subtao index starts from zero */
  if (&vmnos->proxnum == NULL) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"Set number of subsolvers first. See TaoVMNOSSetSeparableOperatorCount.");
  for (i=0; i<vmnos->proxnum; i++) sub[i] = vmnos->subtaos[i];
  PetscFunctionReturn(0);
}

/*@
   TaoVMNOSGetVMMat - Gets the variable metric matrix. This musted be called from subtao context.

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameter:
.  vm  - pointer to variable metric matrix

   Level: intermediate

.seealso: TAOVMNOS
@*/
PetscErrorCode TaoVMNOSGetVMMat(Tao tao, Mat *vm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)tao,"TaoVMNOSGetVMMat_VMNOS", (PetscObject*) vm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
