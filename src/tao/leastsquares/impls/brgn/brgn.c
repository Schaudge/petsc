#include <../src/tao/leastsquares/impls/brgn/brgn.h> /*I "petsctao.h" I*/

#define BRGN_REGULARIZATION_USER    0
#define BRGN_REGULARIZATION_L2PROX  1
#define BRGN_REGULARIZATION_L1DICT  2
#define BRGN_REGULARIZATION_TYPES   3

static const char *BRGN_REGULARIZATION_TABLE[64] = {"user","l2prox","l1dict"};

static PetscErrorCode GNHessianProd(Mat H,Vec in,Vec out)
{
  TAO_BRGN              *gn;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;    
  ierr = MatShellGetContext(H,&gn);CHKERRQ(ierr);
  ierr = MatMult(gn->subsolver->ls_jac,in,gn->r_work);CHKERRQ(ierr);
  ierr = MatMultTranspose(gn->subsolver->ls_jac,gn->r_work,out);CHKERRQ(ierr);
  switch (gn->reg_type) {
  case BRGN_REGULARIZATION_USER:
    ierr = MatMult(gn->Hreg,in,gn->x_work);CHKERRQ(ierr);
    ierr = VecAXPY(out,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L2PROX:
    ierr = VecAXPY(out,gn->lambda,in);CHKERRQ(ierr);
    break;
  case BRGN_REGULARIZATION_L1DICT:
    /* out = out + lambda*D'*(diag.*(D*in)) */
    if (gn->D) {
      ierr = MatMult(gn->D,in,gn->y);CHKERRQ(ierr);/* y = D*in */
    } else {
      ierr = VecCopy(in,gn->y);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(gn->y_work,gn->diag,gn->y);CHKERRQ(ierr);   /* y_work = diag.*(D*in), where diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3 */
    if (gn->D) {
      ierr = MatMultTranspose(gn->D,gn->y_work,gn->x_work);CHKERRQ(ierr); /* x_work = D'*(diag.*(D*in)) */
    } else {
      ierr = VecCopy(gn->y_work,gn->x_work);CHKERRQ(ierr);
    }
    ierr = VecAXPY(out,gn->lambda,gn->x_work);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNObjectiveGradientEval(Tao tao,Vec X,PetscReal *fcn,Vec G,void *ptr)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscInt              K;                    /* dimension of D*X */
  PetscScalar           yESum;
  PetscErrorCode        ierr;
  PetscReal             f_reg,workNorm;
  Vec                   z,u,temp;
  
  PetscFunctionBegin;
  /* compute objective *fcn*/
  /* compute first term 0.5*||ls_res||_2^2 */
  ierr = TaoComputeResidual(tao,X,tao->ls_res);CHKERRQ(ierr);
  ierr = VecDot(tao->ls_res,tao->ls_res,fcn);CHKERRQ(ierr);
  *fcn *= 0.5;
  /* compute gradient G */
  ierr = TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->ls_jac,tao->ls_res,G);CHKERRQ(ierr);
  /* add the regularization contribution */
  if (!gn->use_admm) {
    switch (gn->reg_type) {
    case BRGN_REGULARIZATION_USER:
      ierr = (*gn->regularizerobjandgrad)(tao,X,&f_reg,gn->x_work,gn->reg_obj_ctx);CHKERRQ(ierr);
      *fcn += gn->lambda*f_reg;
      ierr = VecAXPY(G,gn->lambda,gn->x_work);CHKERRQ(ierr);
      break;
    case BRGN_REGULARIZATION_L2PROX:
      /* compute f = f + lambda*0.5*(xk - xkm1)'*(xk - xkm1) */
      ierr = VecAXPBYPCZ(gn->x_work,1.0,-1.0,0.0,X,gn->x_old);CHKERRQ(ierr); 
      ierr = VecDot(gn->x_work,gn->x_work,&f_reg);CHKERRQ(ierr);
      *fcn += gn->lambda*0.5*f_reg;
      /* compute G = G + lambda*(xk - xkm1) */
      ierr = VecAXPBYPCZ(G,gn->lambda,-gn->lambda,1.0,X,gn->x_old);CHKERRQ(ierr);
      break;
    case BRGN_REGULARIZATION_L1DICT:
      /* compute f = f + lambda*sum(sqrt(y.^2+epsilon^2) - epsilon), where y = D*x*/
      if (gn->D) {
        ierr = MatMult(gn->D,X,gn->y);CHKERRQ(ierr);/* y = D*x */
      } else {
        ierr = VecCopy(X,gn->y);CHKERRQ(ierr);
      }
      ierr = VecPointwiseMult(gn->y_work,gn->y,gn->y);CHKERRQ(ierr);
      ierr = VecShift(gn->y_work,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
      ierr = VecSqrtAbs(gn->y_work);CHKERRQ(ierr);  /* gn->y_work = sqrt(y.^2+epsilon^2) */ 
      ierr = VecSum(gn->y_work,&yESum);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = VecGetSize(gn->y,&K);CHKERRQ(ierr);
      *fcn += gn->lambda*(yESum - K*gn->epsilon);
      /* compute G = G + lambda*D'*(y./sqrt(y.^2+epsilon^2)),where y = D*x */  
      ierr = VecPointwiseDivide(gn->y_work,gn->y,gn->y_work);CHKERRQ(ierr); /* reuse y_work = y./sqrt(y.^2+epsilon^2) */
      if (gn->D) {
        ierr = MatMultTranspose(gn->D,gn->y_work,gn->x_work);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(gn->y_work,gn->x_work);CHKERRQ(ierr);
      }
      ierr = VecAXPY(G,gn->lambda,gn->x_work);CHKERRQ(ierr);
      break;
    }
  } else {
	/* Regularization Misfit Aug Lag Term */
	z    = gn->z_old;
    u    = gn->u;
    temp = gn->x_work2; /* TODO maybe another work vector? */
    /* temp = x - z + u */
	ierr = VecCopy(X,temp);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(temp,-1.,1.,1.,z,u);CHKERRQ(ierr);
    /* workNorm = ||x - z + u||^2 */
	ierr = VecDot(temp,temp,&workNorm);CHKERRQ(ierr);
    *fcn += gn->aug_lag*0.5*workNorm;
	/* Gradient Term */
    ierr = VecAXPY(G,gn->aug_lag,temp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNComputeHessian(Tao tao,Vec X,Mat H,Mat Hpre,void *ptr)
{ 
  TAO_BRGN              *gn = (TAO_BRGN *)ptr;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoComputeResidualJacobian(tao,X,tao->ls_jac,tao->ls_jac_pre);CHKERRQ(ierr);
  
  if (!gn->use_admm) {
    switch (gn->reg_type) {
    case BRGN_REGULARIZATION_USER:
      ierr = (*gn->regularizerhessian)(tao,X,gn->Hreg,gn->reg_hess_ctx);CHKERRQ(ierr);
      break;
    case BRGN_REGULARIZATION_L2PROX:
      break;
    case BRGN_REGULARIZATION_L1DICT:
      /* calculate and store diagonal matrix as a vector: diag = epsilon^2 ./ sqrt(x.^2+epsilon^2).^3* --> diag = epsilon^2 ./ sqrt(y.^2+epsilon^2).^3,where y = D*x */  
      if (gn->D) {
        ierr = MatMult(gn->D,X,gn->y);CHKERRQ(ierr);/* y = D*x */
      } else {
        ierr = VecCopy(X,gn->y);CHKERRQ(ierr);
      }
      ierr = VecPointwiseMult(gn->y_work,gn->y,gn->y);CHKERRQ(ierr);
      ierr = VecShift(gn->y_work,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
      ierr = VecCopy(gn->y_work,gn->diag);CHKERRQ(ierr);                  /* gn->diag = y.^2+epsilon^2 */
      ierr = VecSqrtAbs(gn->y_work);CHKERRQ(ierr);                        /* gn->y_work = sqrt(y.^2+epsilon^2) */ 
      ierr = VecPointwiseMult(gn->diag,gn->y_work,gn->diag);CHKERRQ(ierr);/* gn->diag = sqrt(y.^2+epsilon^2).^3 */
      ierr = VecReciprocal(gn->diag);CHKERRQ(ierr);
      ierr = VecScale(gn->diag,gn->epsilon*gn->epsilon);CHKERRQ(ierr);
      break;
    }
  } else {
    ierr = MatShift(H, gn->aug_lag);CHKERRQ(ierr);
    if (Hpre != H) {
      ierr = MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/* NORM_2 Case: 0.5 || x ||_2 + 0.5 * mu * ||x + u - z||^2 */
static PetscErrorCode ObjectiveRegularizationADMM(Tao tao, Vec z, PetscReal *J, void *ptr)
{
  TAO_BRGN       *gn = (TAO_BRGN *)ptr;
  PetscReal      mu, workNorm, reg;
  Vec            x, u, temp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu   = gn->aug_lag;
  x    = gn->xk;
  u    = gn->u;
  temp = gn->x_work2; /* TODO maybe another work vector? */

  ierr = VecNorm(z, NORM_2, &reg);CHKERRQ(ierr);
  reg  = 0.5 * reg * reg;

  ierr = VecCopy(z,temp);CHKERRQ(ierr);
  /* temp = x + u -z */
  ierr = VecAXPBYPCZ(temp,1.,1.,-1.,x,u);CHKERRQ(ierr);
  /* workNorm = ||x + u - z ||^2 */
  ierr = VecDot(temp, temp, &workNorm);CHKERRQ(ierr);
  *J   = reg + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: x - mu*(x + u - z) */
static PetscErrorCode GradientRegularizationADMM(Tao tao, Vec z, Vec V, void *ptr)
{
  TAO_BRGN       *gn = (TAO_BRGN *)ptr;
  PetscReal      mu;
  Vec            x, u, temp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu   = gn->aug_lag;
  x    = gn->xk;
  u    = gn->u;
  temp = gn->x_work; /* TODO maybe another work vector? */
  ierr = VecCopy(x, V);CHKERRQ(ierr);
  ierr = VecCopy(z, temp);CHKERRQ(ierr);
  /* temp = x + u -z */
  ierr = VecAXPBYPCZ(temp,1.,1.,-1.,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(V, -mu, temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NORM_2 Case: returns diag(mu) */
static PetscErrorCode HessianRegularizationADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  TAO_BRGN       *gn = (TAO_BRGN *)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Identity matrix scaled by mu */
  ierr = MatZeroEntries(H);CHKERRQ(ierr);
  ierr = MatShift(H,gn->aug_lag);CHKERRQ(ierr);
  if (Hpre != H) {
    ierr = MatZeroEntries(Hpre);CHKERRQ(ierr);
    ierr = MatShift(Hpre,gn->aug_lag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GNHookFunction(Tao tao,PetscInt iter, void *ctx)
{
  TAO_BRGN              *gn = (TAO_BRGN *)ctx;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;
  /* Update basic tao information from the subsolver */
  gn->parent->nfuncs = tao->nfuncs;
  gn->parent->ngrads = tao->ngrads;
  gn->parent->nfuncgrads = tao->nfuncgrads;
  gn->parent->nhess = tao->nhess;
  gn->parent->niter = tao->niter;
  gn->parent->ksp_its = tao->ksp_its;
  gn->parent->ksp_tot_its = tao->ksp_tot_its;
  ierr = TaoGetConvergedReason(tao,&gn->parent->reason);CHKERRQ(ierr);
  /* Update the solution vectors */
  if (iter == 0) {
    ierr = VecSet(gn->x_old,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(tao->solution,gn->x_old);CHKERRQ(ierr);
    ierr = VecCopy(tao->solution,gn->parent->solution);CHKERRQ(ierr);
  }
  /* Update the gradient */
  ierr = VecCopy(tao->gradient,gn->parent->gradient);CHKERRQ(ierr);
  /* Call general purpose update function */
  if (gn->parent->ops->update) {
    ierr = (*gn->parent->ops->update)(gn->parent,gn->parent->niter,gn->parent->user_update);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;
  PetscInt              i,N;
  PetscReal             u_norm, r_norm, s_norm, x_norm, z_norm, primal, dual, temp, temp2;
  Vec                   u,xk,z,zold,diff,zdiff;

  PetscFunctionBegin;
  if (gn->use_admm){
	u    = gn->u;
	xk   = gn->subsolver->solution;
	zold = gn->ztemp;
	z    = gn->admm_subsolver->solution;
	diff = gn->xzdiff;
	zdiff = gn->zdiff;
    ierr = VecGetSize(u,&N);CHKERRQ(ierr);
	for (i=0; i<gn->admm_iter; i++){
      ierr = VecCopy(z,zold);CHKERRQ(ierr);
      ierr = TaoSolve(gn->subsolver);CHKERRQ(ierr); /* xk */
      ierr = VecNorm(xk,NORM_2,&temp);CHKERRQ(ierr);
      ierr = TaoSolve(gn->admm_subsolver);CHKERRQ(ierr); /* z */
      ierr = VecNorm(z,NORM_2,&temp2);CHKERRQ(ierr);
      /* u = u + xk -z */
      ierr   = VecAXPBYPCZ(u,1.,-1.,1.,xk,z);CHKERRQ(ierr);
      /* r_norm : norm(x-z) */
      ierr   = VecWAXPY(diff,-1.,z,xk);CHKERRQ(ierr);
      ierr   = VecNorm(diff,NORM_2,&r_norm);CHKERRQ(ierr);
      /* s_norm : norm(-mu(z-zold)) */
      ierr   = VecWAXPY(zdiff, -1.,zold,z);CHKERRQ(ierr);
      ierr   = VecNorm(zdiff,NORM_2,&s_norm);CHKERRQ(ierr);
      s_norm = s_norm * gn->aug_lag;
      /* primal : sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z))*/
      ierr   = VecNorm(xk,NORM_2,&x_norm);CHKERRQ(ierr);
      ierr   = VecNorm(z,NORM_2,&z_norm);CHKERRQ(ierr);
      primal = PetscSqrtReal(N)*gn->abstol + gn->reltol*PetscMax(x_norm,z_norm);
      /* Duality : sqrt(n)*ABSTOL + RELTOL*norm(mu*u)*/
      ierr   = VecNorm(u,NORM_2,&u_norm);CHKERRQ(ierr);
      dual   = PetscSqrtReal(N)*gn->abstol + gn->reltol*u_norm*gn->aug_lag;
      ierr   = PetscPrintf(PetscObjectComm((PetscObject)gn->admm_subsolver),"Iter %D : ||x-z||: %g, mu*||z-zold||: %g\n", i, (double) r_norm, (double) s_norm);CHKERRQ(ierr);
      if (r_norm < primal && s_norm < dual) break;
	}
  }
  else {
    ierr = TaoSolve(gn->subsolver);CHKERRQ(ierr);
      ierr = VecNorm(gn->subsolver->solution,NORM_2,&temp);CHKERRQ(ierr);
  }
  /* Update basic tao information from the subsolver */
  tao->nfuncs = gn->subsolver->nfuncs;
  tao->ngrads = gn->subsolver->ngrads;
  tao->nfuncgrads = gn->subsolver->nfuncgrads;
  tao->nhess = gn->subsolver->nhess;
  tao->niter = gn->subsolver->niter;
  tao->ksp_its = gn->subsolver->ksp_its;
  tao->ksp_tot_its = gn->subsolver->ksp_tot_its;
  ierr = TaoGetConvergedReason(gn->subsolver,&tao->reason);CHKERRQ(ierr);
  /* Update vectors */
  ierr = VecCopy(gn->subsolver->solution,tao->solution);CHKERRQ(ierr);
  ierr = VecCopy(gn->subsolver->gradient,tao->gradient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BRGN(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"least-squares problems with regularizer: ||f(x)||^2 + lambda*g(x), g(x) = ||xk-xkm1||^2 or ||Dx||_1 or user defined function.");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_regularizer_weight","regularizer weight (default 1e-4)","",gn->lambda,&gn->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_l1_smooth_epsilon","L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)","",gn->epsilon,&gn->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_brgn_aug_lag","Augmented Lagrangian Multiplier in ADMM","",gn->aug_lag,&gn->aug_lag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-brgn_admm_abstol","ADMM abstol","",gn->abstol,&gn->abstol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-brgn_admm_reltol","ADMM reltol","",gn->reltol,&gn->reltol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_brgn_solve_admm","Trigger to use ADMM for BRGN","",gn->use_admm,&gn->use_admm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_brgn_regularization_type","regularization type", "",BRGN_REGULARIZATION_TABLE,BRGN_REGULARIZATION_TYPES,BRGN_REGULARIZATION_TABLE[gn->reg_type],&gn->reg_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-admm_iter", "ADMM iteration limit", "", gn->admm_iter, &(gn->admm_iter), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(gn->subsolver);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BRGN(Tao tao,PetscViewer viewer)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = TaoView(gn->subsolver,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;
  PetscBool             is_bnls,is_bntr,is_bntl;
  PetscInt              i,n,N,K; /* dict has size K*N*/

  PetscFunctionBegin;
  if (!tao->ls_res) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualRoutine() must be called before setup!");
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNLS,&is_bnls);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTR,&is_bntr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)gn->subsolver,TAOBNTL,&is_bntl);CHKERRQ(ierr);
  if ((is_bnls || is_bntr || is_bntl) && !tao->ls_jac) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ORDER,"TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!gn->x_work) {
    ierr = VecDuplicate(tao->solution,&gn->x_work);CHKERRQ(ierr);
  }
  if (!gn->r_work) {
    ierr = VecDuplicate(tao->ls_res,&gn->r_work);CHKERRQ(ierr);
  }
  if (!gn->x_old) {
    ierr = VecDuplicate(tao->solution,&gn->x_old);CHKERRQ(ierr);
    ierr = VecSet(gn->x_old,0.0);CHKERRQ(ierr);
  }
  if (!gn->u) {
    ierr = VecDuplicate(tao->solution,&gn->u);CHKERRQ(ierr);
    ierr = VecSet(gn->u,0.0);CHKERRQ(ierr);
  }
  if (!gn->x_work2) {
    ierr = VecDuplicate(tao->solution,&gn->x_work2);CHKERRQ(ierr);
    ierr = VecSet(gn->x_work2,0.0);CHKERRQ(ierr);
  }
  if (!gn->z_old) {
    ierr = VecDuplicate(tao->solution,&gn->z_old);CHKERRQ(ierr);
    ierr = VecSet(gn->z_old,0.0);CHKERRQ(ierr);
  }
  if (!gn->xk) {
    ierr = VecDuplicate(tao->solution,&gn->xk);CHKERRQ(ierr);
    ierr = VecSet(gn->xk,0.0);CHKERRQ(ierr);
  }
  if (!gn->xzdiff) {
    ierr = VecDuplicate(tao->solution,&gn->xzdiff);CHKERRQ(ierr);
    ierr = VecSet(gn->xzdiff,0.0);CHKERRQ(ierr);
  }
  if (!gn->zdiff) {
    ierr = VecDuplicate(tao->solution,&gn->zdiff);CHKERRQ(ierr);
    ierr = VecSet(gn->zdiff,0.0);CHKERRQ(ierr);
  }
  if (!gn->ztemp) {
    ierr = VecDuplicate(tao->solution,&gn->ztemp);CHKERRQ(ierr);
    ierr = VecSet(gn->ztemp,0.0);CHKERRQ(ierr);
  }

  if (BRGN_REGULARIZATION_L1DICT == gn->reg_type) {
    if (gn->D) {
      ierr = MatGetSize(gn->D,&K,&N);CHKERRQ(ierr); /* Shell matrices still must have sizes defined. K = N for identity matrix, K=N-1 or N for gradient matrix */
    } else {
      ierr = VecGetSize(tao->solution,&K);CHKERRQ(ierr); /* If user does not setup dict matrix, use identiy matrix, K=N */
    }
    if (!gn->y) {    
      ierr = VecCreate(PETSC_COMM_SELF,&gn->y);CHKERRQ(ierr);
      ierr = VecSetSizes(gn->y,PETSC_DECIDE,K);CHKERRQ(ierr);
      ierr = VecSetFromOptions(gn->y);CHKERRQ(ierr);
      ierr = VecSet(gn->y,0.0);CHKERRQ(ierr);

    }
    if (!gn->y_work) {
      ierr = VecDuplicate(gn->y,&gn->y_work);CHKERRQ(ierr);
    }
    if (!gn->diag) {
      ierr = VecDuplicate(gn->y,&gn->diag);CHKERRQ(ierr);
      ierr = VecSet(gn->diag,0.0);CHKERRQ(ierr);
    }
  }

  if (!tao->setupcalled) {
    /* Hessian setup */
    ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
    ierr = MatSetSizes(gn->H,n,n,N,N);CHKERRQ(ierr);
    ierr = MatSetType(gn->H,MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(gn->H);CHKERRQ(ierr);
    ierr = MatShellSetOperation(gn->H,MATOP_MULT,(void (*)(void))GNHessianProd);CHKERRQ(ierr);
    ierr = MatShellSetContext(gn->H,(void*)gn);CHKERRQ(ierr);
	if (!(gn->use_admm)) {
      /* Subsolver setup,include initial vector and dicttionary D */
      ierr = TaoSetUpdate(gn->subsolver,GNHookFunction,(void*)gn);CHKERRQ(ierr);
      ierr = TaoSetInitialVector(gn->subsolver,tao->solution);CHKERRQ(ierr);
      if (tao->bounded) {
        ierr = TaoSetVariableBounds(gn->subsolver,tao->XL,tao->XU);CHKERRQ(ierr);
      }
      ierr = TaoSetResidualRoutine(gn->subsolver,tao->ls_res,tao->ops->computeresidual,tao->user_lsresP);CHKERRQ(ierr);
      ierr = TaoSetJacobianResidualRoutine(gn->subsolver,tao->ls_jac,tao->ls_jac,tao->ops->computeresidualjacobian,tao->user_lsjacP);CHKERRQ(ierr);
      ierr = TaoSetObjectiveAndGradientRoutine(gn->subsolver,GNObjectiveGradientEval,(void*)gn);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(gn->subsolver,gn->H,gn->H,GNComputeHessian,(void*)gn);CHKERRQ(ierr);
      /* Propagate some options down */
      ierr = TaoSetTolerances(gn->subsolver,tao->gatol,tao->grtol,tao->gttol);CHKERRQ(ierr);
      ierr = TaoSetMaximumIterations(gn->subsolver,tao->max_it);CHKERRQ(ierr);
      ierr = TaoSetMaximumFunctionEvaluations(gn->subsolver,tao->max_funcs);CHKERRQ(ierr);
      for (i=0; i<tao->numbermonitors; ++i) {
        ierr = TaoSetMonitor(gn->subsolver,tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i]);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)(tao->monitorcontext[i]));CHKERRQ(ierr);
      }
      ierr = TaoSetUp(gn->subsolver);CHKERRQ(ierr);
	} else {
      ierr = TaoSetUpdate(gn->subsolver,GNHookFunction,(void*)gn);CHKERRQ(ierr);
      ierr = MatSetSizes(gn->Hr,n,n,N,N);CHKERRQ(ierr);
      ierr = MatSetType(gn->Hr,MATAIJ);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(gn->Hr, 5, NULL, 5, NULL);CHKERRQ(ierr); /*TODO: some number other than 5?*/
      ierr = MatSeqAIJSetPreallocation(gn->Hr, 5, NULL);CHKERRQ(ierr);
      ierr = MatSetUp(gn->Hr);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(gn->Hr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(gn->Hr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      /* Subsolver setup,include initial vector and dicttionary D */
      ierr = TaoSetUpdate(gn->subsolver,GNHookFunction,(void*)gn);CHKERRQ(ierr);
/*      ierr = TaoSetUpdate(gn->admm_subsolver,GNHookFunction,(void*)gn);CHKERRQ(ierr); */
      ierr = TaoSetInitialVector(gn->subsolver,gn->xk);CHKERRQ(ierr);
      ierr = TaoSetInitialVector(gn->admm_subsolver,gn->z_old);CHKERRQ(ierr);
      if (tao->bounded) {
        ierr = TaoSetVariableBounds(gn->subsolver,tao->XL,tao->XU);CHKERRQ(ierr);
        ierr = TaoSetVariableBounds(gn->admm_subsolver,tao->XL,tao->XU);CHKERRQ(ierr);
      }
      ierr = TaoSetResidualRoutine(gn->subsolver,tao->ls_res,tao->ops->computeresidual,tao->user_lsresP);CHKERRQ(ierr);
      ierr = TaoSetResidualRoutine(gn->admm_subsolver,tao->ls_res,tao->ops->computeresidual,tao->user_lsresP);CHKERRQ(ierr);
      ierr = TaoSetJacobianResidualRoutine(gn->subsolver,tao->ls_jac,tao->ls_jac,tao->ops->computeresidualjacobian,tao->user_lsjacP);CHKERRQ(ierr);
      ierr = TaoSetJacobianResidualRoutine(gn->admm_subsolver,tao->ls_jac,tao->ls_jac,tao->ops->computeresidualjacobian,tao->user_lsjacP);CHKERRQ(ierr);
      ierr = TaoSetObjectiveAndGradientRoutine(gn->subsolver,GNObjectiveGradientEval,(void*)gn);CHKERRQ(ierr);
      ierr = TaoSetObjectiveRoutine(gn->admm_subsolver, ObjectiveRegularizationADMM, (void*)gn);CHKERRQ(ierr);
      ierr = TaoSetGradientRoutine(gn->admm_subsolver, GradientRegularizationADMM, (void*)gn);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(gn->admm_subsolver, gn->Hr, gn->Hr, HessianRegularizationADMM, (void*)gn);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(gn->subsolver,gn->H,gn->H,GNComputeHessian,(void*)gn);CHKERRQ(ierr);
      /* Propagate some options down */
      ierr = TaoSetTolerances(gn->subsolver,tao->gatol,tao->grtol,tao->gttol);CHKERRQ(ierr);
      ierr = TaoSetTolerances(gn->admm_subsolver,tao->gatol,tao->grtol,tao->gttol);CHKERRQ(ierr);
      ierr = TaoSetMaximumIterations(gn->subsolver,tao->max_it);CHKERRQ(ierr);
      ierr = TaoSetMaximumIterations(gn->admm_subsolver,tao->max_it);CHKERRQ(ierr);
      ierr = TaoSetMaximumFunctionEvaluations(gn->subsolver,tao->max_funcs);CHKERRQ(ierr);
      ierr = TaoSetMaximumFunctionEvaluations(gn->admm_subsolver,tao->max_funcs);CHKERRQ(ierr);
      for (i=0; i<tao->numbermonitors; ++i) {
        ierr = TaoSetMonitor(gn->subsolver,tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i]);CHKERRQ(ierr);
        ierr = TaoSetMonitor(gn->admm_subsolver,tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i]);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)(tao->monitorcontext[i]));CHKERRQ(ierr);
      }
      ierr = TaoSetUp(gn->subsolver);CHKERRQ(ierr);
      ierr = TaoSetUp(gn->admm_subsolver);CHKERRQ(ierr);
      gn->xk = gn->subsolver->solution;
	  gn->z_old = gn->admm_subsolver->solution;
	}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BRGN(Tao tao)
{
  TAO_BRGN              *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&tao->gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->x_work);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->x_work2);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->r_work);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->x_old);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->diag);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->y);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->y_work);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->z_old);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->u);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->xk);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->xzdiff);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->zdiff);CHKERRQ(ierr);
    ierr = VecDestroy(&gn->ztemp);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&gn->H);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->Hr);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->D);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->Hreg);CHKERRQ(ierr);
  ierr = TaoDestroy(&gn->subsolver);CHKERRQ(ierr);
  ierr = TaoDestroy(&gn->admm_subsolver);CHKERRQ(ierr);
  gn->parent = NULL;
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  TAOBRGN - Bounded Regularized Gauss-Newton method for solving nonlinear least-squares 
            problems with bound constraints. This algorithm is a thin wrapper around TAOBNTL 
            that constructs the Gauss-Newton problem with the user-provided least-squares 
            residual and Jacobian. The algorithm offers both an L2-norm proximal point ("l2prox") 
            regularizer, and a L1-norm dictionary regularizer ("l1dict"), where we approximate the 
            L1-norm ||x||_1 by sum_i(sqrt(x_i^2+epsilon^2)-epsilon) with a small positive number epsilon.
            The user can also provide own regularization function.

  Options Database Keys:
  + -tao_brgn_regularizer_weight  - regularizer weight (default 1e-4)
  . -tao_brgn_l1_smooth_epsilon   - L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)
  - -tao_brgn_regularization_type - regularization type ("user", "l2prox", "l1dict")

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BRGN(Tao tao)
{
  TAO_BRGN       *gn;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&gn);CHKERRQ(ierr);
  
  tao->ops->destroy = TaoDestroy_BRGN;
  tao->ops->setup = TaoSetUp_BRGN;
  tao->ops->setfromoptions = TaoSetFromOptions_BRGN;
  tao->ops->view = TaoView_BRGN;
  tao->ops->solve = TaoSolve_BRGN;
  
  tao->data = (void*)gn;
  gn->lambda = 1e-4;
  gn->epsilon = 1e-6;
  gn->aug_lag = 1.0;
  gn->parent = tao;
  gn->use_admm = PETSC_FALSE;
  gn->admm_iter = 50;
  gn->abstol = 1.e-4;
  gn->reltol = 1.e-2;
  
  ierr = MatCreate(PetscObjectComm((PetscObject)tao),&gn->H);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(gn->H,"tao_brgn_hessian_");CHKERRQ(ierr);
  
  ierr = MatCreate(PetscObjectComm((PetscObject)tao),&gn->Hr);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(gn->Hr,"tao_brgn_admm_hessian_");CHKERRQ(ierr);
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&gn->subsolver);CHKERRQ(ierr);
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&gn->admm_subsolver);CHKERRQ(ierr);
  ierr = TaoSetType(gn->subsolver,TAOBNLS);CHKERRQ(ierr);
  ierr = TaoSetType(gn->admm_subsolver,TAOBNLS);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(gn->subsolver,"tao_brgn_subsolver_");CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(gn->admm_subsolver,"tao_brgn_admm_subsolver_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoBRGNGetSubsolver - Get the pointer to the subsolver inside BRGN

  Collective on Tao

  Level: advanced
  
  Input Parameters:
+  tao - the Tao solver context
-  subsolver - the Tao sub-solver context
@*/
PetscErrorCode TaoBRGNGetSubsolver(Tao tao,Tao *subsolver)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  PetscFunctionBegin;
  *subsolver = gn->subsolver;
  PetscFunctionReturn(0);
}


PetscErrorCode TaoBRGNGetADMMSubsolver(Tao tao,Tao *subsolver)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  PetscFunctionBegin;
  *subsolver = gn->admm_subsolver;
  PetscFunctionReturn(0);
}
/*@
  TaoBRGNSetRegularizerWeight - Set the regularizer weight for the Gauss-Newton least-squares algorithm

  Collective on Tao
  
  Input Parameters:
+  tao - the Tao solver context
-  lambda - L1-norm regularizer weight

  Level: beginner
@*/
PetscErrorCode TaoBRGNSetRegularizerWeight(Tao tao,PetscReal lambda)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  /* Initialize lambda here */

  PetscFunctionBegin;
  gn->lambda = lambda;
  PetscFunctionReturn(0);
}

/*@
  TaoBRGNSetL1SmoothEpsilon - Set the L1-norm smooth approximation parameter for L1-regularized least-squares algorithm

  Collective on Tao
  
  Input Parameters:
+  tao - the Tao solver context
-  epsilon - L1-norm smooth approximation parameter

  Level: advanced
@*/
PetscErrorCode TaoBRGNSetL1SmoothEpsilon(Tao tao,PetscReal epsilon)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  
  /* Initialize epsilon here */

  PetscFunctionBegin;
  gn->epsilon = epsilon;
  PetscFunctionReturn(0);
}

/*@
   TaoBRGNSetDictionaryMatrix - bind the dictionary matrix from user application context to gn->D, for compressed sensing (with least-squares problem)

   Input Parameters:
+  tao  - the Tao context
.  dict - the user specified dictionary matrix.  We allow to set a null dictionary, which means identity matrix by default

    Level: advanced
@*/
PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao tao,Mat dict)  
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (dict) {
    PetscValidHeaderSpecific(dict,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,dict,2);
    ierr = PetscObjectReference((PetscObject)dict);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&gn->D);CHKERRQ(ierr);
  gn->D = dict;
  PetscFunctionReturn(0);
}

/*@C
   TaoBRGNSetRegularizerObjectiveAndGradientRoutine - Sets the user-defined regularizer call-back 
   function into the algorithm.

   Input Parameters:
   + tao - the Tao context
   . func - function pointer for the regularizer value and gradient evaluation
   - ctx - user context for the regularizer

   Level: advanced
@*/
PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(Tao tao,PetscErrorCode (*func)(Tao,Vec,PetscReal *,Vec,void*),void *ctx)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (ctx) {
    gn->reg_obj_ctx = ctx;
  }
  if (func) {
    gn->regularizerobjandgrad = func;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoBRGNSetRegularizerHessianRoutine - Sets the user-defined regularizer call-back 
   function into the algorithm.

   Input Parameters:
   + tao - the Tao context
   . Hreg - user-created matrix for the Hessian of the regularization term
   . func - function pointer for the regularizer Hessian evaluation
   - ctx - user context for the regularizer Hessian

   Level: advanced
@*/
PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(Tao tao,Mat Hreg,PetscErrorCode (*func)(Tao,Vec,Mat,void*),void *ctx)
{
  TAO_BRGN       *gn = (TAO_BRGN *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (Hreg) {
    PetscValidHeaderSpecific(Hreg,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,Hreg,2);
  } else SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONG,"NULL Hessian detected! User must provide valid Hessian for the regularizer.");
  if (ctx) {
    gn->reg_hess_ctx = ctx;
  }
  if (func) {
    gn->regularizerhessian = func;
  }
  if (Hreg) {
    ierr = PetscObjectReference((PetscObject)Hreg);CHKERRQ(ierr);
    ierr = MatDestroy(&gn->Hreg);CHKERRQ(ierr);
    gn->Hreg = Hreg;
  }
  PetscFunctionReturn(0);
}
