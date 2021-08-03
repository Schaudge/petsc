#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/

static PetscErrorCode VecSafeguard(Vec X)
{
  Vec Z;
  IS zero;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(X, &Z);CHKERRQ(ierr);
  ierr = VecZeroEntries(Z);CHKERRQ(ierr);
  ierr = VecWhichEqual(X, Z, &zero);CHKERRQ(ierr);
  ierr = VecISSet(X, zero, PETSC_SQRT_MACHINE_EPSILON);CHKERRQ(ierr);
  ierr = VecDestroy(&Z);CHKERRQ(ierr);
  ierr = ISDestroy(&zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeBarrierFunction(Tao tao, FullSpaceVec *Q, PetscReal *barrier)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  FullSpaceVec   *W = mad->W;
  PetscReal      slacksum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *barrier = 0.0;
  if (!mad->use_ipm) PetscFunctionReturn(0);
  if (tao->ineq_constrained) {
    if (mad->isIL) {
      ierr = VecCopy(Q->Scl, W->Scl);CHKERRQ(ierr);
      ierr = VecLog(W->Scl);CHKERRQ(ierr);
      ierr = VecSum(W->Scl, &slacksum);CHKERRQ(ierr);
      *barrier += slacksum;
    }
    if (mad->isIU) {
      ierr = VecCopy(Q->Scu, W->Scu);CHKERRQ(ierr);
      ierr = VecLog(W->Scu);CHKERRQ(ierr);
      ierr = VecSum(W->Scu, &slacksum);CHKERRQ(ierr);
      *barrier += slacksum;
    }
  }
  if (tao->bounded) {
    if (mad->isXL) {
      ierr = VecCopy(Q->Sxl, W->Sxl);CHKERRQ(ierr);
      ierr = VecLog(W->Sxl);CHKERRQ(ierr);
      ierr = VecSum(W->Sxl, &slacksum);CHKERRQ(ierr);
      *barrier += slacksum;
    }
    if (mad->isXU) {
      ierr = VecCopy(Q->Sxu, W->Sxu);CHKERRQ(ierr);
      ierr = VecLog(W->Sxu);CHKERRQ(ierr);
      ierr = VecSum(W->Sxu, &slacksum);CHKERRQ(ierr);
      *barrier += slacksum;
    }
  }
  *barrier *= -1.0;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeLagrangianAndGradient(Tao tao, FullSpaceVec *Q, Lagrangian *L, FullSpaceVec *dLdQ)
{
  TAO_MAD           *mad = (TAO_MAD*)tao->data;
  FullSpaceVec      *W = mad->W;
  Vec               Xb, Cb;
  PetscReal         slacksum, tmp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* compute f(x) and dL/dX = dF/dX */
  L->barrier = 0.0;
  ierr = VecZeroEntries(dLdQ->F);CHKERRQ(ierr);
  ierr = TaoComputeObjectiveAndGradient(tao, Q->X, &L->obj, dLdQ->X);
  ierr = VecScale(dLdQ->X, mad->Gscale);CHKERRQ(ierr);
  ierr = VecCopy(dLdQ->X, tao->gradient);CHKERRQ(ierr);
  if (tao->eq_constrained) {
    /* dL/dYe = Ce */
    ierr = TaoComputeEqualityConstraints(tao, Q->X, mad->Ce);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mad->Ce, mad->CeScale, mad->Ce);CHKERRQ(ierr);  /* dynamic scaling */
    ierr = VecCopy(mad->Ce, dLdQ->Ye);CHKERRQ(ierr);
    /* dL/dX += Ae^T Ye */
    ierr = TaoComputeJacobianEquality(tao, Q->X, mad->Ae, mad->Ae);CHKERRQ(ierr);
    ierr = MatDiagonalScale(mad->Ae, mad->CeScale, NULL);CHKERRQ(ierr);  /* dynamic scaling */
    ierr = MatMultTransposeAdd(mad->Ae, Q->Ye, dLdQ->X, dLdQ->X);CHKERRQ(ierr);
    /* compute contribution to the Lagrangian, L += Ye^T Ce */
    ierr = VecDot(Q->Ye, dLdQ->Ye, &L->yeTce);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    /* dL/dYi = Ci */
    ierr = TaoComputeInequalityConstraints(tao, Q->X, mad->Ci);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mad->Ci, mad->CiScale, mad->Ci);CHKERRQ(ierr);  /* dynamic scaling */
    /* dL/dX += Ai^T Yi */
    ierr = TaoComputeJacobianInequality(tao, Q->X, mad->Ai, mad->Ai);CHKERRQ(ierr);
    ierr = MatDiagonalScale(mad->Ai, mad->CiScale, NULL);CHKERRQ(ierr);  /* dynamic scaling */
    ierr = MatMultTransposeAdd(mad->Ai, Q->Yi, dLdQ->X, dLdQ->X);CHKERRQ(ierr);
    /* compute contribution to the Lagrangian, L += Yi^T Ci */
    ierr = VecDot(Q->Yi, dLdQ->Yi, &L->yiTci);CHKERRQ(ierr);
    if (mad->use_ipm) {
      /* dL/dYi -= Sc */
      ierr = VecWAXPY(dLdQ->Yi, -1.0, Q->Sc, mad->Ci);CHKERRQ(ierr);
      /* dL/dSc = -Yi */
      ierr = VecCopy(Q->Yi, dLdQ->Sc);CHKERRQ(ierr);
      ierr = VecScale(dLdQ->Sc, -1.0);CHKERRQ(ierr);
      /* slack contribution to the Lagrangian */
      ierr = VecDot(Q->Yi, Q->Sc, &tmp);CHKERRQ(ierr);
      L->yiTci -= tmp;
      if (mad->isIL) {
        /* dL/dSc += -Vl */
        ierr = VecISAXPY(dLdQ->Sc, mad->isIL, -1.0, Q->Vl);CHKERRQ(ierr);
        /* dL/dScl = Scl*Vl - mu */
        ierr = VecPointwiseMult(dLdQ->Scl, Q->Scl, Q->Vl);CHKERRQ(ierr);
        ierr = VecShift(dLdQ->Scl, -mad->mu);CHKERRQ(ierr);
        /* dL/dVl = -Sc + Scl + IL */
        ierr = VecGetSubVector(Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
        ierr = VecWAXPY(dLdQ->Vl, -1.0, Cb, mad->IL);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
        ierr = VecAXPY(dLdQ->Vl, 1.0, Q->Scl);CHKERRQ(ierr);
        /* compute contribution to the Lagrangian, L += Vl^T (-Sc + Scl + IL) */
        ierr = VecDot(Q->Vl, dLdQ->Vl, &L->vlTcl);CHKERRQ(ierr);
        /* compute contribution to barrier term, barrier += sum[ln(Scl)] */
        ierr = VecCopy(Q->Scl, W->Scl);CHKERRQ(ierr);
        ierr = VecLog(W->Scl);CHKERRQ(ierr);
        ierr = VecSum(W->Scl, &slacksum);CHKERRQ(ierr);
        L->barrier += slacksum;
      }
      if (mad->isIU) {
        /* dL/dSc += Vu */
        ierr = VecISAXPY(dLdQ->Sc, mad->isIU, 1.0, Q->Vu);CHKERRQ(ierr);
        /* dL/dScu = Scu*Vu - mu */
        ierr = VecPointwiseMult(dLdQ->Scu, Q->Scu, Q->Vu);CHKERRQ(ierr);
        ierr = VecShift(dLdQ->Scu, -mad->mu);CHKERRQ(ierr);
        /* dL/dVu = Sc + Scu - IU */
        ierr = VecGetSubVector(Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
        ierr = VecWAXPY(dLdQ->Vu, -1.0, mad->IU, Cb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
        ierr = VecAXPY(dLdQ->Vu, 1.0, Q->Scu);CHKERRQ(ierr);
        /* compute contribution to the Lagrangian, L += Vu^T (Sc + Scu - IU) */
        ierr = VecDot(Q->Vu, dLdQ->Vu, &L->vuTcu);CHKERRQ(ierr);
        /* compute contribution to barrier term, barrier += sum[ln(Scu)] */
        ierr = VecCopy(Q->Scu, W->Scu);CHKERRQ(ierr);
        ierr = VecLog(W->Scu);CHKERRQ(ierr);
        ierr = VecSum(W->Scu, &slacksum);CHKERRQ(ierr);
        L->barrier += slacksum;
      }
    }
  }
  if (tao->bounded && mad->use_ipm) {
    if (mad->isXL) {
      /* dL/dX += -Zl */
      ierr = VecISAXPY(dLdQ->X, mad->isXL, -1.0, Q->Zl);CHKERRQ(ierr);
      /* dL/dSxl = Sxl*Zl - mu */
      ierr = VecPointwiseMult(dLdQ->Sxl, Q->Sxl, Q->Zl);CHKERRQ(ierr);
      ierr = VecShift(dLdQ->Sxl, -mad->mu);CHKERRQ(ierr);
      /* dL/dZl = -X + Sxl + XL */
      ierr = VecGetSubVector(Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(dLdQ->Zl, -1.0, Xb, mad->XL);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecAXPY(dLdQ->Zl, 1.0, Q->Sxl);CHKERRQ(ierr);
      /* compute contribution to the Lagrangian, L += Zl^T (-X + Sxl + XL) */
      ierr = VecDot(Q->Zl, dLdQ->Zl, &L->zlTxl);CHKERRQ(ierr);
      /* compute contribution to barrier term, barrier += sum[ln(Sxl)] */
      ierr = VecCopy(Q->Sxl, W->Sxl);CHKERRQ(ierr);
      ierr = VecLog(W->Sxl);CHKERRQ(ierr);
      ierr = VecSum(W->Sxl, &slacksum);CHKERRQ(ierr);
      L->barrier += slacksum;
    }
    if (mad->isXU) {
      /* dL/dX += Zu */
      ierr = VecISAXPY(dLdQ->X, mad->isXU, 1.0, Q->Zu);CHKERRQ(ierr);
      /* dL/dSxu = Sxu*Zu - mu */
      ierr = VecPointwiseMult(dLdQ->Sxu, Q->Sxu, Q->Zu);CHKERRQ(ierr);
      ierr = VecShift(dLdQ->Sxu, -mad->mu);CHKERRQ(ierr);
      /* dL/dZu = X + Sxu - XU */
      ierr = VecGetSubVector(Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(dLdQ->Zu, -1.0, mad->XU, Xb);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecAXPY(dLdQ->Zu, 1.0, Q->Sxu);CHKERRQ(ierr);
      /* compute contribution to the Lagrangian, L += Zu^T (X + Sxu - XU) */
      ierr = VecDot(Q->Zu, dLdQ->Zu, &L->zuTxu);CHKERRQ(ierr);
      /* compute contribution to the barrier term, barrier += sum[ln(Sxu)] */
      ierr = VecCopy(Q->Sxu, W->Sxu);CHKERRQ(ierr);
      ierr = VecLog(W->Sxu);CHKERRQ(ierr);
      ierr = VecSum(W->Sxu, &slacksum);CHKERRQ(ierr);
      L->barrier += slacksum;
    }
  }
  /* assemble the Lagrangian */
  L->val = L->obj + L->yeTce + L->yiTci + L->vlTcl + L->vuTcu + L->zlTxl + L->zuTxu - mad->mu*L->barrier;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeReducedKKT(Tao tao, FullSpaceVec *Q, FullSpaceVec *dLdQ, ReducedSpaceVec *G)
{
  TAO_MAD           *mad = (TAO_MAD*)tao->data;
  FullSpaceVec      *W = mad->W;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!mad->use_ipm) {
    ierr = VecCopy(dLdQ->R, G->R);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* start with Gx = dLdX */
  ierr = VecCopy(dLdQ->X, G->X);CHKERRQ(ierr);
  if (tao->bounded) {
    if (mad->isXL) {
      /* compute and add Wzl = (Zl*dLdZl - dLdSxl)/Sxl */
      ierr = VecPointwiseMult(W->Zl, Q->Zl, dLdQ->Zl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Zl, -1.0, dLdQ->Sxl);CHKERRQ(ierr);
      ierr = VecCopy(Q->Sxl, W->Sxl);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Sxl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Zl, W->Zl, W->Sxl);CHKERRQ(ierr);
      /* add the lower bound contribution */
      ierr = VecISAXPY(G->X, mad->isXL, -1.0, W->Zl);CHKERRQ(ierr);
    }
    if (mad->isXU) {
      /* compute and stash Wzu = (Zu*dLdZu - dLdSxu)/Sxu */
      ierr = VecPointwiseMult(W->Zu, Q->Zu, dLdQ->Zu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Zu, -1.0, dLdQ->Sxu);CHKERRQ(ierr);
      ierr = VecCopy(Q->Sxu, W->Sxu);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Sxu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Zu, W->Zu, W->Sxu);CHKERRQ(ierr);
      /* add the upper bound contribution */
      ierr = VecISAXPY(G->X, mad->isXU, 1.0, W->Zu);CHKERRQ(ierr);
    }
  }
  if (tao->ineq_constrained) {
    /* zero out scaling vector and start accumulating with Gi = dLdSc */
    ierr = VecZeroEntries(mad->B);CHKERRQ(ierr);
    ierr = VecCopy(dLdQ->Sc, G->Yi);CHKERRQ(ierr);
    if (mad->isIL) {
      /* compute B += Vl/Scl */
      ierr = VecCopy(Q->Scl, W->Scl);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vl, Q->Vl, W->Scl);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIL, 1.0, W->Vl);CHKERRQ(ierr);
      /* compute and stash Wvl = (Vl*dLdVl - dLdScl)/Scl */
      ierr = VecPointwiseMult(W->Vl, Q->Vl, dLdQ->Vl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vl, -1.0, dLdQ->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vl, W->Vl, W->Scl);CHKERRQ(ierr);
      /* add the lower bound contribution Gi += Wvl */
      ierr = VecISAXPY(G->Yi, mad->isIL, -1.0, W->Vl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* compute B += Vu/Scu */
      ierr = VecCopy(Q->Scu, W->Scu);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vu, Q->Vu, W->Scu);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIU, 1.0, W->Vu);CHKERRQ(ierr);
      /* compute and stash Wvu = (Vu*dLdVu - dLdScu)/Scu */
      ierr = VecPointwiseMult(W->Vu, Q->Vu, dLdQ->Vu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vu, -1.0, dLdQ->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vu, W->Vu, W->Scu);CHKERRQ(ierr);
      /* add the upper bound contribution Gi += Wvu */
      ierr = VecISAXPY(G->Yi, mad->isIU, 1.0, W->Vu);CHKERRQ(ierr);
    }
    /* scale the bound terms and add to central Gi = (dLdSc - Wvl + Wvu)/B */
    ierr = VecSafeguard(mad->B);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(G->Yi, G->Yi, mad->B);CHKERRQ(ierr);
    ierr = VecAXPY(G->Yi, 1.0, dLdQ->Yi);CHKERRQ(ierr);
  }
  if (tao->eq_constrained) {
    /* set Rye = dLdYe */
    ierr = VecCopy(dLdQ->Ye, G->Ye);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADEvaluateClosedFormUpdates(Tao tao, FullSpaceVec *Q, FullSpaceVec *dLdQ, FullSpaceVec *D)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  FullSpaceVec       *W = mad->W;
  Vec                Xb, S;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!mad->use_ipm) PetscFunctionReturn(0);
  if (tao->ineq_constrained) {
    /* zero work vector for (Vl/Scl + Vu/Scu) coefficient */
    ierr = VecSet(mad->B, 0.0);CHKERRQ(ierr);
    /* compute dSc = dYi - dLdSc */
    ierr = VecWAXPY(D->Sc, -1.0, dLdQ->Sc, D->Yi);CHKERRQ(ierr);
    if (mad->isIL) {
      /* add lower bound contribution dSc += (Vl*dLdVl - dLdScl)/Scl */
      ierr = VecPointwiseMult(W->Vl, Q->Vl, dLdQ->Vl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vl, -1.0, dLdQ->Scl);CHKERRQ(ierr);
      ierr = VecCopy(Q->Scl, W->Scl);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vl, W->Vl, W->Scl);CHKERRQ(ierr);
      ierr = VecAXPY(D->Sc, 1.0, W->Vl);CHKERRQ(ierr);
      /* compute B += Vl/Scl */
      ierr = VecPointwiseDivide(W->Vl, Q->Vl, W->Scl);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIL, 1.0, W->Vl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* add upper bound contribution dSc += -(Vu*dLdVu - dLdScu)/Scu */
      ierr = VecPointwiseMult(W->Vu, Q->Vu, dLdQ->Vu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vu, -1.0, dLdQ->Scu);CHKERRQ(ierr);
      ierr = VecCopy(Q->Scu, W->Scu);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vu, W->Vu, W->Scu);CHKERRQ(ierr);
      ierr = VecAXPY(D->Sc, -1.0, W->Vu);CHKERRQ(ierr);
      /* compute B += Vu/Scu */
      ierr = VecPointwiseDivide(W->Vu, Q->Vu, W->Scu);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIU, 1.0, W->Vu);CHKERRQ(ierr);
    }
    /* apply final coefficient to dSc /= B */
    ierr = VecSafeguard(mad->B);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(D->Sc, D->Sc, mad->B);CHKERRQ(ierr);
    /* go back and compute dScl, dVl, dScu and dVu based on dSc */
    if (mad->isIL) {
      /* dScl = - dLdVl + dSc */
      ierr = VecGetSubVector(D->Sc, mad->isIL, &S);CHKERRQ(ierr);
      ierr = VecWAXPY(D->Scl, -1.0, dLdQ->Vl, S);CHKERRQ(ierr);
      /* dVl = (Vl*dLdVl - dLdScl + Vl*dSc)/Scl*/
      ierr = VecWAXPY(D->Vl, 1.0, dLdQ->Vl, S);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(D->Sc, mad->isIL, &S);CHKERRQ(ierr);
      ierr = VecPointwiseMult(D->Vl, Q->Vl, D->Vl);CHKERRQ(ierr);
      ierr = VecAXPY(D->Vl, -1.0, dLdQ->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Vl, D->Vl, W->Scl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* dScu = - dLdVu - dSc */
      ierr = VecGetSubVector(D->Sc, mad->isIU, &S);CHKERRQ(ierr);
      ierr = VecWAXPY(D->Scu, 1.0, dLdQ->Vu, S);CHKERRQ(ierr);
      ierr = VecScale(D->Scu, -1.0);CHKERRQ(ierr);
      /* dVu = (Vu*dLdVu - dLdScu - Vu*dSc)/Scu */
      ierr = VecWAXPY(D->Vu, -1.0, S, dLdQ->Vu);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(D->Sc, mad->isIL, &S);CHKERRQ(ierr);
      ierr = VecPointwiseMult(D->Vu, Q->Vu, D->Vu);CHKERRQ(ierr);
      ierr = VecAXPY(D->Vu, -1.0, dLdQ->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Vu, D->Vu, W->Scu);CHKERRQ(ierr);
    }
  }
  if (tao->bounded) {
    if (mad->isXL) {
      /* dSxl = - dLdZl + dX */
      ierr = VecGetSubVector(D->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(D->Sxl, -1.0, dLdQ->Zl, Xb);CHKERRQ(ierr);
      /* dZl = (Zl*dLdZl - dLdSxl + Zl*dX)/Sxl */
      ierr = VecWAXPY(D->Zl, 1.0, dLdQ->Zl, Xb);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(D->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecPointwiseMult(D->Zl, Q->Zl, D->Zl);CHKERRQ(ierr);
      ierr = VecAXPY(D->Zl, -1.0, dLdQ->Sxl);CHKERRQ(ierr);
      ierr = VecCopy(Q->Sxl, W->Sxl);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Sxl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Zl, D->Zl, W->Sxl);CHKERRQ(ierr);
    }
    if (mad->isXU) {
      /* dSxu = - dLdZu - dX */
      ierr = VecGetSubVector(D->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(D->Sxu, 1.0, dLdQ->Zu, Xb);CHKERRQ(ierr);
      ierr = VecScale(D->Sxu, -1.0);CHKERRQ(ierr);
      /* dZu = (Zu*dLdZu - dLdSxu - Zu*dX)/Sxu */
      ierr = VecWAXPY(D->Zu, 1.0, dLdQ->Zu, Xb);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(D->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecPointwiseMult(D->Zu, Q->Zu, D->Zu);CHKERRQ(ierr);
      ierr = VecAXPY(D->Zu, -1.0, dLdQ->Sxu);CHKERRQ(ierr);
      ierr = VecCopy(Q->Sxu, W->Sxu);CHKERRQ(ierr);
      ierr = VecSafeguard(W->Sxu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Zu, D->Zu, W->Sxu);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADEstimateActiveSet(Tao tao, Vec X, Vec dLdX, PetscReal alpha, Vec D, PetscBool *changed)
{
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  PetscInt n;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  *changed = PETSC_FALSE;
  if (mad->use_ipm) PetscFunctionReturn(0);
  if (tao->bounded) {
    /* Bertsekas active set estimation with a gradient descent step */
    ierr = VecCopy(mad->dLdQ->F, mad->dLdQwork->F);CHKERRQ(ierr);
    ierr = VecScale(mad->dLdQwork->F, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(X, tao->XL, tao->XU, dLdX, mad->dLdQwork->X, mad->W->X, alpha, &mad->bound_tol, &mad->isXL, &mad->isXU, &mad->fixedXB, &mad->activeXB, &mad->inactiveXB);CHKERRQ(ierr);
    ierr = ISGetSize(mad->activeXB, &n);CHKERRQ(ierr);
    if (n > 0) *changed = PETSC_TRUE;
  }
  if (tao->ineq_constrained) {
    /* inequality constraints are estimated with a linearized "step" */
    ierr = TaoComputeInequalityConstraints(tao, X, mad->Ci);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao, X, mad->Ai, mad->Ai);CHKERRQ(ierr);
    ierr = MatMultTranspose(mad->Ai, dLdX, mad->Qwork->Yi);CHKERRQ(ierr);
    ierr = VecCopy(mad->Qwork->Yi, mad->dLdQwork->Yi);CHKERRQ(ierr);
    ierr = VecScale(mad->dLdQwork->Yi, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(mad->Ci, tao->IL, tao->IU, mad->Qwork->Yi, mad->dLdQwork->Yi, mad->W->Yi, alpha, &mad->cons_tol, NULL, NULL, NULL, NULL, &mad->inactiveCI);CHKERRQ(ierr);
    ierr = ISGetSize(mad->inactiveCI, &n);CHKERRQ(ierr);
    if (n > 0) *changed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADEstimateMaxAlphas(Tao tao, FullSpaceVec *Q, FullSpaceVec *D,
                                      PetscReal *alpha_p, PetscReal *alpha_y)
{
  PetscReal          alpha_trial;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  *alpha_p = 1.0;
  if (Q->Scl) {
    ierr = VecStepMax(Q->Scl, D->Scl, &alpha_trial);CHKERRQ(ierr);
    *alpha_p = PetscMin(alpha_trial, *alpha_p);CHKERRQ(ierr);
  }
  if (Q->Scu) {
    ierr = VecStepMax(Q->Scu, D->Scu, &alpha_trial);CHKERRQ(ierr);
    *alpha_p = PetscMin(alpha_trial, *alpha_p);CHKERRQ(ierr);
  }
  if (Q->Scl) {
    ierr = VecStepMax(Q->Sxl, D->Sxl, &alpha_trial);CHKERRQ(ierr);
    *alpha_p = PetscMin(alpha_trial, *alpha_p);CHKERRQ(ierr);
  }
  if (Q->Scl) {
    ierr = VecStepMax(Q->Sxu, D->Sxu, &alpha_trial);CHKERRQ(ierr);
    *alpha_p = PetscMin(alpha_trial, *alpha_p);CHKERRQ(ierr);
  }
  if (Q->Ys) {
    ierr = VecStepMax(Q->Ys, D->Ys, &alpha_trial);CHKERRQ(ierr);
    *alpha_y = PetscMin(alpha_trial, 1.0);CHKERRQ(ierr);
  } else {
    *alpha_y = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADApplyFilterStep(Tao tao, FullSpaceVec *Q, FullSpaceVec *D, Lagrangian *L,
                                     FullSpaceVec *dLdQ, PetscReal *alpha)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  SimpleFilter       *filter = mad->filter;
  PetscReal          cnorm2;
  PetscReal          alpha_p, alpha_y, alpha_t;
  PetscReal          merit, merit_filter;
  PetscInt           i;
  PetscBool          dominated;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TaoMADEstimateMaxAlphas(tao, Q, D, &alpha_p, &alpha_y);CHKERRQ(ierr);
  dominated = PETSC_FALSE;
  alpha_t = 1.0;
  if (mad->filter_type == TAO_MAD_FILTER_NONE) {
    /* just accept the max step length, ignore filter */
    ierr = VecAXPY(Q->P, alpha_t*alpha_p, D->P);CHKERRQ(ierr);
    if (Q->Y) {
      ierr = VecAXPY(Q->Y, alpha_t*alpha_y, D->Y);CHKERRQ(ierr);
    }
    ierr = TaoMADComputeLagrangianAndGradient(tao, Q, L, dLdQ);CHKERRQ(ierr);
  } else {
    /* enter the filter loop here */
    while (alpha_t >= mad->alpha_min) {
      /* apply the trial step and compute objective and feasibility */
      ierr = VecWAXPY(mad->Qtrial->P, alpha_t*alpha_p, D->P, Q->P);CHKERRQ(ierr);
      if (Q->Y) {
        ierr = VecWAXPY(mad->Qtrial->Y, alpha_t*alpha_y, D->Y, Q->Y);CHKERRQ(ierr);
      }
      ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Qtrial, mad->Ltrial, mad->dLdQtrial);CHKERRQ(ierr);
      if (dLdQ->Y) {
        ierr = VecDot(mad->dLdQtrial->Y, mad->dLdQtrial->Y, &cnorm2);CHKERRQ(ierr);
      } else {
        cnorm2 = 0.0;
      }
      /* compute the barrier function if necessary */
      merit = mad->Ltrial->obj;
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
        merit -= mad->mu*mad->Ltrial->barrier;
      }
      /* iterate through the filter and compare the trial point */
      dominated = PETSC_FALSE;
      for (i=0; i<filter->size; i++){
        if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
          /* re-evaluate the barrier function for the filter point using new barrier factor */
          merit_filter = filter->f[i] - mad->mu*filter->b[i];
        } else {
          merit_filter = filter->f[i];
        }
        if ((merit_filter < merit) && (filter->h[i] < cnorm2)) {
          dominated = PETSC_TRUE;
          break;
        }
      }
      if (dominated) {
        /* if the filter dominates, shrink step length and try again */
        alpha_t *= mad->alpha_fac;
      } else {
        /* we found a step so let's accept it */
        ierr = VecCopy(mad->Qtrial->F, Q->F);CHKERRQ(ierr);
        ierr = VecCopy(mad->dLdQtrial->F, dLdQ->F);CHKERRQ(ierr);
        ierr = LagrangianCopy(mad->Ltrial, L);CHKERRQ(ierr);
        /* we found a step length, add to filter and exit */
        ierr = TaoMADUpdateFilter(tao, L->obj, L->barrier, cnorm2);CHKERRQ(ierr);
        break;
      }
    }
  }
  if (dominated) alpha_t = 0.0;
  *alpha = alpha_t;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADUpdateFilter(Tao tao, PetscReal fval, PetscReal barrier, PetscReal cnorm2)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  SimpleFilter       *filter = mad->filter;
  PetscReal          *new_f, *new_h, *new_b;
  PetscInt           i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (filter->size == 0 && mad->filter_type != TAO_MAD_FILTER_NONE) {
    /* this is the first update ever so we have to create the arrays from scratch */
    filter->size += 1;
    ierr = PetscMalloc2(filter->size, &filter->f, filter->size, &filter->h);CHKERRQ(ierr);
    if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
      ierr = PetscMalloc1(filter->size, &filter->b);CHKERRQ(ierr);
    }
  }

  if (mad->filter_type == TAO_MAD_FILTER_MARKOV) {
    /* Markov "filter" only tracks previous iteration so the "filter" never grows */
    filter->f[0] = fval;  filter->h[0] = cnorm2;
  } else {
    filter->size += 1;
    if (filter->size > filter->max_size) {
      /* we need to shift filter values and discard oldest to make room */
      filter->size = filter->max_size;
      for (i=0; i<filter->max_size-1; i++) {
        filter->f[i] = filter->f[i+1];
        filter->h[i] = filter->h[i+1];
        if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
          filter->b[i] = filter->b[i+1];
        }
      }
    } else {
      /* resize filter arrays and add new point */
      ierr = PetscMalloc2(filter->size, &new_f, filter->size, &new_h);CHKERRQ(ierr);
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
        ierr = PetscMalloc1(filter->size, &new_b);CHKERRQ(ierr);
      }
      for (i=0; i<filter->size-1; i++) {
        new_f[i] = filter->f[i];
        new_h[i] = filter->h[i];
        if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
          new_b[i] = filter->b[i];
        } 
      }
      ierr = PetscFree2(filter->f, filter->h);CHKERRQ(ierr);
      filter->f = new_f;  filter->h = new_h;
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER && mad->use_ipm && (tao->bounded || tao->ineq_constrained)) {
        ierr = PetscFree(filter->b);CHKERRQ(ierr);
        filter->b = new_b;
      }
    }
    filter->f[filter->size-1] = fval;
    filter->h[filter->size-1] = cnorm2;
    if (mad->filter_type == TAO_MAD_FILTER_BARRIER) filter->b[filter->size-1] = barrier;
  }
  PetscFunctionReturn(0);
}

/* barrier parameter computation is based on (El-Bakry et. a. 1996), also used in LOQO */
PetscErrorCode TaoMADUpdateBarrier(Tao tao, FullSpaceVec *Q, PetscReal *mu)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  FullSpaceVec   *W = mad->W;
  PetscReal      yTs, min_ys, xi, mu_aff, mu_tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mad->use_ipm) PetscFunctionReturn(0);  /* barrier doesn't exist for active-set */
  if (!tao->ineq_constrained && !tao->bounded) PetscFunctionReturn(0);  /* no inequality constraints */
  /* need dot product of slacks and multipliers, and min value of slacks times multipliers */
  ierr = VecDot(Q->Ys, Q->S, &yTs);CHKERRQ(ierr);
  ierr = VecPointwiseMult(W->Ys, Q->Ys, Q->S);CHKERRQ(ierr);
  ierr = VecMin(W->Ys, NULL, &min_ys);CHKERRQ(ierr);
  /* compute distance from uniformity between slacks and multipliers, xi = min_ys / (yTs/Ns) */
  xi = min_ys/(yTs/mad->Ns);
  /* compute affine scaling/centering parameter, mu_aff = mu_g * min((1-mu_r)*(1-xi)/xi, 2)^3 */
  mu_aff = mad->mu_g*PetscPowReal(PetscMin((1.0 - mad->mu_r)*(1.0 - xi)/xi,2.0), 3.0);
  /* finally compute the new barrier parameter, mu = mu_aff * yTs / Ns */
  mu_tmp = mu_aff*yTs/mad->Ns;
  /* safeguard to prevent uncontrolled increases */
  *mu = PetscMax(mad->mu_min, PetscMin(mad->mu_max, mu_tmp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADCheckConvergence(Tao tao, FullSpaceVec *Q, Lagrangian *L, FullSpaceVec *dLdQ, PetscReal alpha)
{
  TAO_MAD*       mad = (TAO_MAD*)tao->data;
  PetscReal      pnorm, ynorm;
  IS             active, inactive;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* project the gradient to bounds and constraints */
  if (mad->use_ipm) {
    ierr = VecNorm(mad->dLdQ->P, NORM_2, &pnorm);CHKERRQ(ierr);
    if (mad->dLdQ->Y) {
      ierr = VecNorm(mad->dLdQ->Y, NORM_2, &ynorm);CHKERRQ(ierr);
    } else {
      ynorm = 0.0;
    }
  } else {
    if (tao->bounded) {
      ierr = VecFischer(mad->Q->X, mad->dLdQ->X, tao->XL, tao->XU, mad->dLdQwork->X);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(mad->dLdQ->X, mad->dLdQwork->X);CHKERRQ(ierr);
    }
    ierr = VecNorm(mad->dLdQwork->X, NORM_2, &pnorm);CHKERRQ(ierr);
    if (mad->dLdQ->Y) {
      ierr = VecCopy(mad->dLdQ->Y, mad->dLdQwork->Y);CHKERRQ(ierr);
      if (tao->ineq_constrained) {
        ierr = VecWhichLessThan(mad->Ci, tao->IL, &active);CHKERRQ(ierr);
        ierr = ISComplementVec(active, mad->Ci, &inactive);CHKERRQ(ierr);
        ierr = VecISSet(mad->dLdQwork->Yi, inactive, 0.0);CHKERRQ(ierr);
        ierr = ISDestroy(&active);CHKERRQ(ierr);
        ierr = ISDestroy(&inactive);CHKERRQ(ierr);
      }
      ierr = VecNorm(mad->dLdQwork->Y, NORM_2, &ynorm);CHKERRQ(ierr);
    } else {
      ynorm = 0.0;
    }
  }
  /* compute scaling factors for each optimality term */
  ierr = TaoLogConvergenceHistory(tao, L->obj, pnorm, ynorm, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, L->obj, pnorm, ynorm, alpha);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADCheckLagrangianAndGradient(Tao tao, FullSpaceVec *Q, FullSpaceVec *dLdQ)
{
  PetscReal      eps, fd_dirderiv, gradTdir;
  Lagrangian     *L, *Ltrial;
  FullSpaceVec   *Qtrial, *dLdQdummy, *ones;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&Qtrial);CHKERRQ(ierr);
  ierr = PetscNew(&dLdQdummy);CHKERRQ(ierr);
  ierr = PetscNew(&ones);CHKERRQ(ierr);
  ierr = PetscNew(&L);CHKERRQ(ierr);
  ierr = PetscNew(&Ltrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(Q, Qtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(dLdQ, dLdQdummy);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(Q, ones);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Evaluation point:\n");CHKERRQ(ierr);
  ierr = VecView(Q->F, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  eps = PETSC_SQRT_MACHINE_EPSILON;
  ierr = VecSet(ones->F, 1.0);CHKERRQ(ierr);
  ierr = VecWAXPY(Qtrial->F, eps, ones->F, Q->F);CHKERRQ(ierr);
  ierr = TaoMADComputeLagrangianAndGradient(tao, Q, L, dLdQ);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Exact derivative:\n");CHKERRQ(ierr);
  ierr = VecView(dLdQ->F, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TaoMADComputeLagrangianAndGradient(tao, Qtrial, Ltrial, dLdQdummy);CHKERRQ(ierr);
  fd_dirderiv = (Ltrial->val - L->val)/eps;
  ierr = VecDot(dLdQ->F, ones->F, &gradTdir);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)tao), "F-D directional derivative   = %e\n", fd_dirderiv);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)tao), "Exact directional derivative = %e\n", gradTdir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADMonitor(Tao tao, void *ctx)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscViewer    viewer = (PetscViewer)ctx;
  PetscInt       tabs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscViewerASCIIGetTab(viewer, &tabs);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"          Barrier parameter: %g,",(double)mad->mu);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Fraction-to-boundary tolerance: %g\n",(double)mad->tau);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetTab(viewer, tabs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void TaoMADConvertReasonToSNES(TaoConvergedReason taoReason, SNESConvergedReason* snesReason)
{
  switch (taoReason) {
    case TAO_CONTINUE_ITERATING:
      *snesReason = SNES_CONVERGED_ITERATING;
      break;

    case TAO_CONVERGED_GATOL:
      *snesReason = SNES_CONVERGED_FNORM_ABS;
      break;

    case TAO_CONVERGED_GRTOL:
      *snesReason = SNES_CONVERGED_FNORM_RELATIVE;
      break;

    case TAO_CONVERGED_MINF:
    case TAO_CONVERGED_STEPTOL:
      *snesReason = SNES_CONVERGED_SNORM_RELATIVE;
      break;

    case TAO_DIVERGED_MAXFCN:
      *snesReason = SNES_DIVERGED_FUNCTION_COUNT;
      break;

    case TAO_DIVERGED_MAXITS:
      *snesReason = SNES_DIVERGED_MAX_IT;
      break;

    case TAO_DIVERGED_LS_FAILURE:
      *snesReason = SNES_DIVERGED_LINE_SEARCH;
      break;

    case TAO_DIVERGED_NAN:
      *snesReason = SNES_DIVERGED_FNORM_NAN;
      break;

    case TAO_DIVERGED_TR_REDUCTION:
      *snesReason = SNES_DIVERGED_TR_DELTA;
      break;

    default:
      *snesReason = SNES_DIVERGED_LOCAL_MIN;
      break;
  }
}

void TaoMADConvertReasonFromSNES(SNESConvergedReason snesReason, TaoConvergedReason* taoReason)
{
  switch (snesReason) {
    case SNES_CONVERGED_ITERATING:
      *taoReason = TAO_CONTINUE_ITERATING;
      break;

    case SNES_CONVERGED_FNORM_ABS:
      *taoReason = TAO_CONVERGED_GATOL;
      break;

    case SNES_CONVERGED_FNORM_RELATIVE:
      *taoReason = TAO_CONVERGED_GRTOL;
      break;

    case SNES_CONVERGED_SNORM_RELATIVE:
      *taoReason = TAO_CONVERGED_STEPTOL;
      break;

    case SNES_DIVERGED_FUNCTION_COUNT:
      *taoReason = TAO_DIVERGED_MAXFCN;
      break;

    case SNES_DIVERGED_MAX_IT:
      *taoReason = TAO_DIVERGED_MAXITS;
      break;

    case SNES_DIVERGED_LINE_SEARCH:
      *taoReason = TAO_DIVERGED_LS_FAILURE;
      break;

    case SNES_DIVERGED_FNORM_NAN:
      *taoReason = TAO_DIVERGED_NAN;
      break;

    case SNES_DIVERGED_TR_DELTA:
      *taoReason = TAO_DIVERGED_TR_REDUCTION;
      break;

    default:
      *taoReason = TAO_DIVERGED_USER;
      break;
  }
}