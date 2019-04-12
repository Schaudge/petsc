/*
Context for Bounded Regularized Gauss-Newton algorithm.
Extended with L1-regularizer with a linear transformation matrix D:
0.5*||Ax-b||^2 + lambda*||D*x||_1
When D is an identity matrix, we have the classic lasso, aka basis pursuit denoising in compressive sensing problem.
*/

#if !defined(__TAO_BRGN_H)
#define __TAO_BRGN_H

#include <../src/tao/bound/impls/bnk/bnk.h>  /* BNLS, a sub-type of BNK, is used in brgn solver */

typedef struct {
  PetscErrorCode (*regularizerobjandgrad)(Tao,Vec,PetscReal*,Vec,void*);
  PetscErrorCode (*regularizerhessian)(Tao,Vec,Mat,void*);
  void           *reg_obj_ctx;
  void           *reg_hess_ctx;
  Mat            H,Hreg,D, Hr;  /* Hessian, Hessian for regulization part, and Dictionary matrix have size N*N, and K*N respectively. (Jacobian M*N not used here), Hessian for ADMM sub-subsolver*/
  Vec            x_old,x_work,r_work,diag,y,y_work,xk,u,z_old;  /* x, r=J*x, and y=D*x have size N, M, and K respectively. */
  Tao            subsolver,parent,admm_subsolver;
  PetscReal      lambda,epsilon,aug_lag; /* lambda is regularizer weight for both L2-norm Gaussian-Newton and L1-norm, ||x||_1 is approximated with sum(sqrt(x.^2+epsilon^2)-epsilon). aug_lag is augmented lagrangian multiplier*/
  PetscInt       reg_type;
  PetscInt       admm_iter;
  PetscBool      use_admm;
} TAO_BRGN;

#endif /* if !defined(__TAO_BRGN_H) */
