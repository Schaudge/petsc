#include <petsctaomerit.h> /*I "petsctaomerit.h" I*/
#include <petsc/private/taomeritimpl.h>
#include <petsc/private/taoimpl.h>

/*@C
  TaoMeritComputeObjective - Compute the objective function.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. fval - objective function value

  Level: developer

@*/
PetscErrorCode TaoMeritComputeObjective(TaoMerit merit, Vec X, PetscReal *fval)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  if (merit->use_tao) {
      ierr = TaoComputeObjective(merit->tao, X, fval);CHKERRQ(ierr);
  } else {
      if (merit->ops->userobjective) {
          ierr = (*(merit->ops->userobjective))(merit, X, fval, merit->user_obj);CHKERRQ(ierr);
      } else if (merit->ops->userobjandgrad) {
          ierr = (*(merit->ops->userobjandgrad))(merit, X, fval, merit->Gtrial, merit->user_obj);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callback for objective function evaluation");
      }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeGradient - Compute the gradient of the objective function.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. G - gradient vector

  Level: developer

@*/
PetscErrorCode TaoMeritComputeGradient(TaoMerit merit, Vec X, Vec G)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(G,VEC_CLASSID,3);
  if (merit->use_tao) {
      ierr = TaoComputeGradient(merit->tao, X, G);CHKERRQ(ierr);
  } else {
      if (merit->ops->usergradient) {
          ierr = (*(merit->ops->usergradient))(merit, X, G, merit->user_grad);CHKERRQ(ierr);
      } else if (merit->ops->userobjandgrad) {
          ierr = (*(merit->ops->userobjandgrad))(merit, X, &merit->last_value, G, merit->user_grad);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callback for gradient evaluation");
      }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeObjectiveAndGradient - Compute the objective function and its gradient at the same time.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
+ fval - objective function value
- G - gradient vector

  Level: developer

@*/
PetscErrorCode TaoMeritComputeObjectiveAndGradient(TaoMerit merit, Vec X, PetscReal *fval, Vec G)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(G,VEC_CLASSID,3);
  if (merit->use_tao) {
      ierr = TaoComputeGradient(merit->tao, X, G);CHKERRQ(ierr);
  } else {
      if (merit->ops->userobjandgrad) {
          ierr = (*(merit->ops->userobjandgrad))(merit, X, fval, G, merit->user_objgrad);CHKERRQ(ierr);
      } else if ((merit->ops->userobjective) && (merit->ops->usergradient)) {
          ierr = (*(merit->ops->userobjective))(merit, X, fval, merit->user_obj);CHKERRQ(ierr);
          ierr = (*(merit->ops->usergradient))(merit, X, G, merit->user_grad);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for objective and gradient evaluation");
      }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeHessian - Compute the Hessian of the objective.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
+ H - Hessian matrix
- Hpre - preconditioner of the Hessian

  Level: developer

@*/
PetscErrorCode TaoMeritComputeHessian(TaoMerit merit, Vec X, Mat H, Mat Hpre)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(H,MAT_CLASSID,3);
  PetscValidHeaderSpecific(Hpre,MAT_CLASSID,4);
  if (merit->use_tao) {
      ierr = TaoComputeHessian(merit->tao, X, H, Hpre);CHKERRQ(ierr);
  } else {
      if (merit->ops->userhessian) {
          ierr = (*(merit->ops->userhessian))(merit, X, H, Hpre, merit->user_hess);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for Hessian evaluation");
      }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeEqualityConstraints - Compute the equality constraints.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. Ceq - equality constraint vector

  Level: developer

@*/
PetscErrorCode TaoMeritComputeEqualityConstraints(TaoMerit merit, Vec X, Vec Ceq)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Ceq,VEC_CLASSID,3);
  if (merit->use_tao) {
    ierr = TaoComputeEqualityConstraints(merit->tao, X, Ceq);CHKERRQ(ierr);
  } else {
    if (merit->ops->usercnstreq) {
      ierr = (*(merit->ops->usercnstreq))(merit, X, Ceq, merit->user_cnstreq);CHKERRQ(ierr);
    } else {
      SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for equality constraints");
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeEIneualityConstraints - Compute the inequality constraints.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. Cineq - inequality constraint vector

  Level: developer

@*/
PetscErrorCode TaoMeritComputeInequalityConstraints(TaoMerit merit, Vec X, Vec Cineq)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Cineq,VEC_CLASSID,3);
  if (merit->use_tao) {
    ierr = TaoComputeInequalityConstraints(merit->tao, X, Cineq);CHKERRQ(ierr);
  } else {
    if (merit->ops->usercnstrineq) {
      ierr = (*(merit->ops->usercnstrineq))(merit, X, Cineq, merit->user_cnstrineq);CHKERRQ(ierr);
    } else {
      SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for inequality constraints");
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeEqualityJacobian - Compute the Jacobian of the equality constraints.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. Jeq - equality constraint Jacobian matrix

  Level: developer

@*/
PetscErrorCode TaoMeritComputeEqualityJacobian(TaoMerit merit, Vec X, Mat Jeq)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Jeq,MAT_CLASSID,3);
  if (merit->use_tao) {
      ierr = TaoComputeJacobianEquality(merit->tao, X, Jeq, merit->Jeq_pre);CHKERRQ(ierr);
  } else {
      if (merit->ops->userjaceq) {
          ierr = (*(merit->ops->userjaceq))(merit, X, Jeq, merit->user_jaceq);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for equality constraint Jacobian");
      }
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritComputeInequalityJacobian - Compute the Jacobian of the inequality constraints.

  Input Parameters:
+ merit - the TaoMerit context
- X - vector for optimization variables

  Output Parameters:
. Jineq - inequality constraint Jacobian matrix

  Level: developer

@*/
PetscErrorCode TaoMeritComputeInequalityJacobian(TaoMerit merit, Vec X, Mat Jineq)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Jineq,MAT_CLASSID,3);
  if (merit->use_tao) {
      ierr = TaoComputeJacobianInequality(merit->tao, X, Jineq, merit->Jineq_pre);CHKERRQ(ierr);
  } else {
      if (merit->ops->userjacineq) {
          ierr = (*(merit->ops->userjacineq))(merit, X, Jineq, merit->user_jacineq);CHKERRQ(ierr);
      } else {
          SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ORDER,"Cannot find available user callbacks for inequality constraint Jacobian");
      }
  }
  PetscFunctionReturn(0);
}