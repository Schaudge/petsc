#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/

PetscErrorCode TaoMADComputeBarrierFunction(Tao tao, FullSpaceVec*Q, PetscReal *barrier)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  FullSpaceVec   *W = mad->W;
  PetscReal      slacksum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *barrier = 0.0;
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
  PetscReal         slacksum;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* compute f(x) and dL/dX = dF/dX */
  L->barrier = 0.0;
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
    /* dL/dYi = Ci - Sc */
    ierr = TaoComputeInequalityConstraints(tao, Q->X, mad->Ci);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mad->Ci, mad->CiScale, mad->Ci);CHKERRQ(ierr);  /* dynamic scaling */
    ierr = VecWAXPY(dLdQ->Yi, -1.0, Q->Sc, mad->Ci);CHKERRQ(ierr);
    /* dL/dX += Ai^T (Ci - Sc) */
    ierr = TaoComputeJacobianInequality(tao, Q->X, mad->Ai, mad->Ai);CHKERRQ(ierr);
    ierr = MatDiagonalScale(mad->Ai, mad->CiScale, NULL);CHKERRQ(ierr);  /* dynamic scaling */
    ierr = MatMultTransposeAdd(mad->Ai, dLdQ->Yi, dLdQ->X, dLdQ->X);CHKERRQ(ierr);
    /* dL/dSc = -Yi */
    ierr = VecCopy(Q->Yi, dLdQ->Sc);CHKERRQ(ierr);
    ierr = VecScale(dLdQ->Sc, -1.0);CHKERRQ(ierr);
    /* compute contribution to the Lagrangian, L += Yi^T (Ci - Sc) */
    ierr = VecDot(Q->Yi, dLdQ->Yi, &L->yiTci);CHKERRQ(ierr);
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
    /* dL/dSc = Sc*(-Yi - Vl + Vu) - mu */
    ierr = VecPointwiseMult(dLdQ->Sc, Q->Sc, dLdQ->Sc);CHKERRQ(ierr);
    ierr = VecShift(dLdQ->Sc, -mad->mu);CHKERRQ(ierr);
  }
  if (tao->bounded) {
    if (mad->isXL) {
      /* dL/dX += -Zl */
      ierr = VecAXPY(dLdQ->X, -1.0, Q->Zl);CHKERRQ(ierr);
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
      ierr = VecAXPY(dLdQ->X, -1.0, Q->Zl);CHKERRQ(ierr);
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
  /* start with Gx = dLdX */
  ierr = VecCopy(dLdQ->X, G->X);CHKERRQ(ierr);
  if (mad->unconstrained) PetscFunctionReturn(0);
  if (tao->bounded) {
    if (mad->isXL) {
      /* compute and add Wzl = (Zl*dLdZl - dLdSxl)/Sxl */
      ierr = VecPointwiseMult(W->Zl, Q->Zl, dLdQ->Zl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Zl, -1.0, dLdQ->Sxl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Zl, W->Zl, Q->Sxl);CHKERRQ(ierr);
      /* add the lower bound contribution */
      ierr = VecISAXPY(G->X, mad->isXL, -1.0, W->Zl);CHKERRQ(ierr);
    }
    if (mad->isXU) {
      /* compute and stash Wzu = (Zu*dLdZu - dLdSxu)/Sxu */
      ierr = VecPointwiseMult(W->Zu, Q->Zu, dLdQ->Zu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Zu, -1.0, dLdQ->Sxu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Zu, W->Zu, Q->Sxu);CHKERRQ(ierr);
      /* add the upper bound contribution */
      ierr = VecISAXPY(G->X, mad->isXU, 1.0, W->Zu);CHKERRQ(ierr);
    }
  }
  if (tao->ineq_constrained) {
    /* zero out scaling vector and start accumulating with Gi = dLdSc */
    ierr = VecSet(mad->B, 0.0);CHKERRQ(ierr);
    ierr = VecCopy(dLdQ->Sc, G->Yi);CHKERRQ(ierr);
    if (mad->isIL) {
      /* compute B += Vl/Scl */
      ierr = VecPointwiseDivide(W->Vl, Q->Vl, Q->Scl);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIL, 1.0, W->Vl);CHKERRQ(ierr);
      /* compute and stash Wvl = (Vl*dLdVl - dLdScl)/Scl */
      ierr = VecPointwiseMult(W->Vl, Q->Vl, dLdQ->Vl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vl, -1.0, dLdQ->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vl, W->Vl, Q->Scl);CHKERRQ(ierr);
      /* add the lower bound contribution Gi += Wvl */
      ierr = VecISAXPY(G->Yi, mad->isIL, -1.0, W->Vl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* compute B += Vu/Scu */
      ierr = VecPointwiseDivide(W->Vu, Q->Vu, Q->Scu);CHKERRQ(ierr);
      ierr = VecISAXPY(mad->B, mad->isIU, 1.0, W->Vu);CHKERRQ(ierr);
      /* compute and stash Wvu = (Vu*dLdVu - dLdScu)/Scu */
      ierr = VecPointwiseMult(W->Vu, Q->Vu, dLdQ->Vu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vu, -1.0, dLdQ->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vu, W->Vu, Q->Scu);CHKERRQ(ierr);
      /* add the upper bound contribution Gi += Wvu */
      ierr = VecISAXPY(G->Yi, mad->isIU, 1.0, W->Vu);CHKERRQ(ierr);
    }
    /* scale the bound terms and add to central Gi = dLdYi + (dLdSc - Wvl + Wvu)/B */
    ierr = VecPointwiseDivide(G->Yi, G->Yi, mad->B);CHKERRQ(ierr);
    ierr = VecAXPY(G->Yi, 1.0, dLdQ->Yi);CHKERRQ(ierr);
  }
  if (tao->eq_constrained) {
    /* set Rye = dLdYe */
    ierr = VecCopy(dLdQ->Ye, G->Ye);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADEvaluateClosedFormUpdates(Tao tao, FullSpaceVec *P, FullSpaceVec *dLdQ, FullSpaceVec *D)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  FullSpaceVec       *W = mad->W;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mad->unconstrained) PetscFunctionReturn(0);
  if (tao->ineq_constrained) {
    /* zero work vector for (Vl/Scl + Vu/Scu) coefficient */
    ierr = VecSet(W->Sc, 0.0);CHKERRQ(ierr);
    /* compute dSc = dYi - dLdSc */
    ierr = VecWAXPY(D->Sc, -1.0, dLdQ->Sc, D->Yi);CHKERRQ(ierr);
    if (mad->isIL) {
      /* add lower bound contribution dSc += (Vl*dLdVl - dLdScl)/Scl */
      ierr = VecPointwiseMult(W->Vl, P->Vl, dLdQ->Vl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vl, -1.0, dLdQ->Scl);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vl, W->Vl, P->Scl);CHKERRQ(ierr);
      ierr = VecAXPY(D->Sc, 1.0, W->Vl);CHKERRQ(ierr);
      /* compute Wsc += Vl/Scl */
      ierr = VecPointwiseDivide(W->Scl, P->Vl, P->Scl);CHKERRQ(ierr);
      ierr = VecAXPY(W->Sc, 1.0, W->Scl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* add upper bound contribution dSc += -(Vu*dLdVu - dLdScu)/Scu */
      ierr = VecPointwiseMult(W->Vu, P->Vu, dLdQ->Vu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Vu, -1.0, dLdQ->Scu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(W->Vu, W->Vu, P->Scu);CHKERRQ(ierr);
      ierr = VecAXPY(D->Sc, -1.0, W->Vu);CHKERRQ(ierr);
      /* compute Wsc += Vu/Scu */
      ierr = VecPointwiseDivide(W->Scu, P->Vu, P->Scu);CHKERRQ(ierr);
      ierr = VecAXPY(W->Sc, 1.0, W->Scl);CHKERRQ(ierr);
    }
    /* apply final coefficient to dSc /= Wsc */
    ierr = VecPointwiseDivide(D->Sc, D->Sc, W->Sc);CHKERRQ(ierr);
    /* go back and compute dScl, dVl, dScu and dVu based on dSc */
    if (mad->isIL) {
      /* dScl = -dLdVl + dSc */
      ierr = VecWAXPY(D->Scl, -1.0, dLdQ->Vl, D->Sc);CHKERRQ(ierr);
      /* dVl = Wvl + (Vl*dSc)/Scl */
      ierr = VecPointwiseMult(D->Vl, P->Vl, D->Sc);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Vl, D->Vl, P->Scu);CHKERRQ(ierr);
      ierr = VecAXPBY(D->Vl, 1.0, 1.0, W->Vl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      /* dScu = -dLdVu - dSc */
      ierr = VecWAXPY(D->Scu, 1.0, dLdQ->Vu, D->Sc);CHKERRQ(ierr);
      ierr = VecScale(D->Scu, -1.0);CHKERRQ(ierr);
      /* dVu = Wvu - (Vu*dSc)/Scu */
      ierr = VecPointwiseMult(D->Vu, P->Vu, D->Sc);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Vu, D->Vu, P->Scu);CHKERRQ(ierr);
      ierr = VecAXPBY(D->Vu, 1.0, -1.0, W->Vu);CHKERRQ(ierr);
    }
  }
  if (tao->bounded) {
    if (mad->isXL) {
      /* dSxl = -dLdZl + dX */
      ierr = VecWAXPY(D->Sxl, -1.0, dLdQ->Zl, D->X);CHKERRQ(ierr);
      /* dZl = Wzl + (Zl*dX)/Sxl */
      ierr = VecPointwiseMult(D->Zl, P->Zl, D->X);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Zl, D->Zl, P->Sxl);CHKERRQ(ierr);
      ierr = VecAXPBY(D->Zl, 1.0, 1.0, W->Zl);CHKERRQ(ierr);
    }
    if (mad->isXU) {
      /* dSxu = -dLdZu - dX */
      ierr = VecWAXPY(D->Sxu, 1.0, dLdQ->Zu, D->X);CHKERRQ(ierr);
      ierr = VecScale(D->Sxu, -1.0);CHKERRQ(ierr);
      /* dZu = Wzu - (Zu*dX)/Sxu */
      ierr = VecPointwiseMult(D->Zu, P->Zu, D->X);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(D->Zu, D->Zu, P->Sxl);CHKERRQ(ierr);
      ierr = VecAXPBY(D->Zu, 1.0, -1.0, W->Zu);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADTestFractionToBoundary(Tao tao, Vec Q, PetscReal alpha, Vec D, PetscBool *violates)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  const PetscScalar  *qq, *dd;
  PetscInt           i, start, end;
  PetscBool          violates_local;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Q, &qq);CHKERRQ(ierr);
  ierr = VecGetArrayRead(D, &dd);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(D, &start, &end);CHKERRQ(ierr);
  violates_local = PETSC_FALSE;
  for (i=start; i<end; i++) {
    if (qq[i] + alpha*dd[i] < (1.0 - mad->tau)*qq[i]) {
      violates_local = PETSC_TRUE;
      break;
    }
  }
  ierr = VecRestoreArrayRead(Q, &qq);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(D, &dd);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&violates_local, violates, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)tao));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADEstimateMaxAlphas(Tao tao, FullSpaceVec *Q, FullSpaceVec *D,
                                      PetscReal *alpha_p, PetscReal *alpha_y)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscReal          alpha_trial;
  PetscBool          ftb_violated;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mad->unconstrained) {
    *alpha_p = 1.0;
    *alpha_y = 0.0;
    PetscFunctionReturn(0);
  }

  /* first estimate the primal step */
  alpha_trial = 1.0;
  while (alpha_trial >= mad->alpha_min) {
    if (tao->ineq_constrained) {
      ftb_violated = PETSC_FALSE;
      ierr = TaoMADTestFractionToBoundary(tao, Q->Sc, alpha_trial, D->Sc, &ftb_violated);CHKERRQ(ierr);
      if (ftb_violated) continue;
      if (mad->isIL) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Scl, alpha_trial, D->Scl, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
      if (mad->isIU) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Scu, alpha_trial, D->Scu, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
    }
    if (tao->bounded) {
      if (mad->isXL) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Sxl, alpha_trial, D->Sxl, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
      if (mad->isXU) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Sxu, alpha_trial, D->Sxu, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
    }
    if (ftb_violated) {
      if (alpha_trial > mad->alpha_cut) alpha_trial -= mad->alpha_cut;
      else alpha_trial *= mad->alpha_fac;
    } else break;
  }
  *alpha_p = alpha_trial;

  /* repeat for the dual step */
  alpha_trial = 1.0;
  while (alpha_trial >= mad->alpha_min) {
    if (tao->ineq_constrained) {
      ftb_violated = PETSC_FALSE;
      ierr = TaoMADTestFractionToBoundary(tao, Q->Yi, alpha_trial, D->Yi, &ftb_violated);CHKERRQ(ierr);
      if (ftb_violated) continue;
      if (mad->isIL) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Vl, alpha_trial, D->Vl, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
      if (mad->isIU) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Vu, alpha_trial, D->Vu, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
    }
    if (tao->bounded) {
      if (mad->isXL) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Zl, alpha_trial, D->Zl, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
      if (mad->isXU) {
        ftb_violated = PETSC_FALSE;
        ierr = TaoMADTestFractionToBoundary(tao, Q->Zu, alpha_trial, D->Zu, &ftb_violated);CHKERRQ(ierr);
        if (ftb_violated) continue;
      }
    }
    if (ftb_violated) {
      if (alpha_trial > mad->alpha_cut) alpha_trial -= mad->alpha_cut;
      else alpha_trial *= mad->alpha_fac;
    } else break;
  }
  *alpha_y = alpha_trial;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADApplyFilterStep(Tao tao, FullSpaceVec *Q, FullSpaceVec *D, Lagrangian *L,
                                     FullSpaceVec *dLdQ, PetscBool *dominated)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  SimpleFilter       *filter = mad->filter;
  PetscReal          cnorm, dfTdx, dcTdc, dsTs;
  PetscReal          suff_f, suff_h;
  PetscReal          alpha_p, alpha_y;
  PetscReal          merit, merit_filter;
  PetscInt           i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* use fraction-to-the-boundary rule to estimate maximum step lengths */
  ierr = TaoMADEstimateMaxAlphas(tao, Q, D, &alpha_p, &alpha_y);CHKERRQ(ierr);
  *dominated = PETSC_FALSE;
  if (mad->filter_type == TAO_MAD_FILTER_NONE) {
    /* just accept the max step length, ignore filter */
    ierr = VecAXPY(Q->P, alpha_p, D->P);CHKERRQ(ierr);
    if (Q->Y) {
      ierr = VecAXPY(Q->Y, alpha_y, D->Y);CHKERRQ(ierr);
    }
    ierr = TaoMADComputeLagrangianAndGradient(tao, Q, L, dLdQ);CHKERRQ(ierr);
    if (dLdQ->Y) {
      ierr = VecNorm(dLdQ->Y, NORM_2, &cnorm);CHKERRQ(ierr);
    } else {
      cnorm = 0.0;
    }
  } else {
    /* compute sufficient decrease conditions at the initial iterate */
    ierr = VecDot(dLdQ->X, D->X, &dfTdx);CHKERRQ(ierr);
    if (dLdQ->Y) {
      ierr = VecDot(dLdQ->Y, dLdQ->Y, &dcTdc);CHKERRQ(ierr);
      suff_h = -2.0*mad->suff_decr*dcTdc;
    } else {
      suff_h = 0.0;
    }
    if (mad->filter_type == TAO_MAD_FILTER_BARRIER) {
      ierr = VecDot(dLdQ->S, D->S, &dsTs);CHKERRQ(ierr);
    } else {
      dsTs = 0.0;
    }
    suff_f = mad->suff_decr*(dfTdx + dsTs);
    /* enter the filter loop here */
    while (alpha_p >= mad->alpha_min) {
      /* apply the trial step and compute objective and feasibility */
      ierr = VecAXPY(Q->P, alpha_p, D->P);CHKERRQ(ierr);
      if (Q->Y) {
        ierr = VecAXPY(Q->Y, alpha_y, D->Y);CHKERRQ(ierr);
      }
      ierr = TaoMADComputeLagrangianAndGradient(tao, Q, L, dLdQ);CHKERRQ(ierr);
      if (dLdQ->Y) {
        ierr = VecDot(dLdQ->Y, dLdQ->Y, &cnorm);CHKERRQ(ierr);
      } else {
        cnorm = 0.0;
      }
      /* compute the barrier function if necessary */
      merit = L->obj;
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER) {
        merit += mad->mu*L->barrier;
      }
      /* iterate through the filter and compare the trial point */
      *dominated = PETSC_FALSE;
      for (i=0; i<filter->size; i++){
        if (mad->filter_type == TAO_MAD_FILTER_BARRIER) {
          /* re-evaluate the barrier function for the filter point using new barrier factor */
          merit_filter = filter->f[i] + mad->mu*filter->b[i];
        } else {
          merit_filter = filter->f[i];
        }
        if ((merit > merit_filter + alpha_p*suff_f) && (cnorm > filter->h[i] + alpha_y*suff_h)) {
          *dominated = PETSC_TRUE;
        }
      }
      if (*dominated) {
        /* if the filter dominates, shrink step length and try again */
        if (alpha_p > mad->alpha_cut) {
          alpha_p -= mad->alpha_cut;
        } else {
          alpha_p *= mad->alpha_fac;
        }
      } else {
        /* we found a step length, add to filter and exit */
        ierr = TaoMADUpdateFilter(tao, L->obj, L->barrier, cnorm);CHKERRQ(ierr);
        /* we've been working with dot products so square-root them for the actual L2 norm */
        cnorm = PetscSqrtReal(cnorm);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADUpdateFilter(Tao tao, PetscReal fval, PetscReal barrier, PetscReal cnorm2)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  SimpleFilter       *filter = mad->filter;
  PetscInt           i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mad->filter_type == TAO_MAD_FILTER_MARKOV) {
    /* Markov "filter" only tracks previous iteration so the "filter" never grows */
    filter->f[0] = fval;  filter->h[0] = cnorm2;
  } else {
    filter->size += 1;
    if (filter->size == 1) {
      /* this is the first update ever so we have to create the arrays from scratch */
      ierr = PetscMalloc2(1, &filter->f, 1, &filter->h);CHKERRQ(ierr);
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER) {
        ierr = PetscMalloc1(1, &filter->b);CHKERRQ(ierr);
      }
    } else if (filter->size > filter->max_size) {
      /* we need to shift filter values and discard oldest to make room */
      for (i=0; i<filter->max_size-1; i++) {
        filter->f[i] = filter->f[i+1];
        filter->h[i] = filter->h[i+1];
        if (mad->filter_type == TAO_MAD_FILTER_BARRIER) filter->b[i] = filter->b[i+1];
      }
    } else {
      /* resize filter arrays and add new point */
      ierr = PetscRealloc(filter->size, &filter->f);CHKERRQ(ierr);
      ierr = PetscRealloc(filter->size, &filter->h);CHKERRQ(ierr);
      if (mad->filter_type == TAO_MAD_FILTER_BARRIER) {
        ierr = PetscRealloc(filter->size, &filter->b);CHKERRQ(ierr);
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
  PetscReal      tmp, yTs, min_tmp, min_ys, xi, mu_aff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mad->unconstrained) PetscFunctionReturn(0);  /* unconstrained problem no-op */
  /* need dot product of slacks and multipliers, and min value of slacks times multipliers */
  yTs = 0.0;
  min_ys = PETSC_INFINITY;
  if (tao->ineq_constrained) {
    ierr = VecDot(Q->Yi, Q->Sc, &tmp);CHKERRQ(ierr);
    yTs += tmp;
    ierr = VecPointwiseMult(W->Yi, Q->Yi, Q->Sc);CHKERRQ(ierr);
    ierr = VecMin(W->Yi, NULL, &min_tmp);CHKERRQ(ierr);
    min_ys = PetscMin(min_ys, min_tmp);CHKERRQ(ierr);
    if (mad->isIL) {
      ierr = VecDot(Q->Vl, Q->Scl, &tmp);CHKERRQ(ierr);
      yTs += tmp;
      ierr = VecPointwiseMult(W->Vl, Q->Vl, Q->Scl);CHKERRQ(ierr);
      ierr = VecMin(W->Vl, NULL, &min_tmp);CHKERRQ(ierr);
      min_ys = PetscMin(min_ys, min_tmp);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      ierr = VecDot(Q->Vu, Q->Scu, &tmp);CHKERRQ(ierr);
      yTs += tmp;
      ierr = VecPointwiseMult(W->Vu, Q->Vl, Q->Scu);CHKERRQ(ierr);
      ierr = VecMin(W->Vu, NULL, &min_tmp);CHKERRQ(ierr);
      min_ys = PetscMin(min_ys, min_tmp);CHKERRQ(ierr);
    }
  }
  if (tao->bounded) {
    if (mad->isXL) {
      ierr = VecDot(Q->Zl, Q->Sxl, &tmp);CHKERRQ(ierr);
      yTs += tmp;
      ierr = VecPointwiseMult(W->Zl, Q->Zl, Q->Sxl);CHKERRQ(ierr);
      ierr = VecMin(W->Zl, NULL, &min_tmp);CHKERRQ(ierr);
      min_ys = PetscMin(min_ys, min_tmp);CHKERRQ(ierr);
    }
    if (mad->isXU) {
      ierr = VecDot(Q->Zu, Q->Sxu, &tmp);CHKERRQ(ierr);
      yTs += tmp;
      ierr = VecPointwiseMult(W->Zu, Q->Zu, Q->Sxu);CHKERRQ(ierr);
      ierr = VecMin(W->Zu, NULL, &min_tmp);CHKERRQ(ierr);
      min_ys = PetscMin(min_ys, min_tmp);CHKERRQ(ierr);
    }
  }
  /* compute distance from uniformity between slacks and multipliers, xi = min_ys / (yTs/Ns) */
  xi = min_ys/(yTs/mad->Ns);
  /* compute affine scaling/centering parameter, mu_aff = mu_g * min((1-mu_r)*(1-xi)/xi, 2)^3 */
  mu_aff = mad->mu_g*PetscPowReal(PetscMin((1.0 - mad->mu_r)*(1.0 - xi)/xi,2.0), 3.0);
  /* finally compute the new barrier parameter, mu = mu_aff * yTs / Ns */
  *mu = mu_aff*yTs/mad->Ns;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADCheckConvergence(Tao tao, Lagrangian *L, FullSpaceVec *dLdQ)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscReal      gnorm, cnorm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNorm(mad->dLdQ->P, NORM_2, &gnorm);CHKERRQ(ierr);
  if (mad->dLdQ->Y) {
    ierr = VecNorm(mad->dLdQ->Y, NORM_2, &cnorm);CHKERRQ(ierr);
  } else {
    cnorm = 0.0;
  }
  ierr = TaoLogConvergenceHistory(tao, L->obj, gnorm, cnorm, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, L->obj, gnorm, cnorm, 0.0);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}