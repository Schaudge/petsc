#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/
#include <petsctao.h>
#include <petscblaslapack.h>

static PetscErrorCode TaoSolve_MAD(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  Mat                AeT, AiT;
  Vec                Xb, Cb, Qtmp, Gtmp;
  PetscReal          ginf, cnorm2, alpha;
  PetscScalar        *Ae_norms, *Ai_norms, *riTr, *ga, *gg, *rhs;
  PetscInt           i, j, *idx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* set initial multipliers to 1.0 */
  ierr = VecSet(mad->Q->Y, 1.0);CHKERRQ(ierr);
  /* initial slacks are a safeguarded clone of the corresponding constraints (Vanderbei and Shanno, 1998) */
  /* however, in their formulation their slacks have a negative sign, ours is positive, so we flip */
  if (tao->ineq_constrained) {
    ierr = TaoComputeInequalityConstraints(tao, mad->Q->X, mad->Ci);CHKERRQ(ierr);
    ierr = VecSet(mad->W->Sc, mad->slack_init);CHKERRQ(ierr);
    ierr = VecPointwiseMax(mad->Q->Sc, mad->Ci, mad->W->Sc);CHKERRQ(ierr);
    if (mad->isIL) {
      ierr = VecGetSubVector(mad->Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
      ierr = VecWAXPY(mad->Q->Scl, -1.0, Cb, mad->IL);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
      ierr = VecSet(mad->W->Scl, mad->slack_init);CHKERRQ(ierr);
      ierr = VecPointwiseMax(mad->Q->Scl, mad->Q->Scl, mad->W->Scl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      ierr = VecGetSubVector(mad->Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
      ierr = VecWAXPY(mad->Q->Scu, -1.0, mad->IU, Cb);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
      ierr = VecSet(mad->W->Scu, mad->slack_init);CHKERRQ(ierr);
      ierr = VecPointwiseMax(mad->Q->Scu, mad->Q->Scu, mad->W->Scu);CHKERRQ(ierr);
    }
  }
  if (tao->ineq_constrained) {
    if (mad->isIL) {
      ierr = VecGetSubVector(mad->Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(mad->Q->Sxl, -1.0, Xb, mad->XL);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
      ierr = VecSet(mad->W->Sxl, mad->slack_init);CHKERRQ(ierr);
      ierr = VecPointwiseMax(mad->Q->Sxl, mad->Q->Sxl, mad->W->Sxl);CHKERRQ(ierr);
    }
    if (mad->isIU) {
      ierr = VecGetSubVector(mad->Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecWAXPY(mad->Q->Sxu, -1.0, mad->XU, Xb);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
      ierr = VecSet(mad->W->Sxu, mad->slack_init);CHKERRQ(ierr);
      ierr = VecPointwiseMax(mad->Q->Sxu, mad->Q->Sxu, mad->W->Sxu);CHKERRQ(ierr);
    }
  }

  /* since we have initial multipliers and slacks, we can now estimate the barrier parameter */
  ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);

  /* compute primal and dual scaling factors (from Wachter and Biegler, 2004 [IPOPT]) */
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Q, mad->L, mad->dLdQ);CHKERRQ(ierr);
  ierr = VecNorm(mad->dLdQ->P, NORM_INFINITY, &ginf);CHKERRQ(ierr);
  mad->Gscale = PetscMin(1.0, mad->scale_max/ginf);
  if (tao->eq_constrained) {
    ierr = MatTranspose(mad->Ae, MAT_INITIAL_MATRIX, &AeT);CHKERRQ(ierr);
    ierr = VecGetArray(mad->W->Ye, &Ae_norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(AeT, NORM_INFINITY, Ae_norms);CHKERRQ(ierr);
    ierr = VecRestoreArray(mad->W->Ye, &Ae_norms);CHKERRQ(ierr);
    ierr = MatDestroy(&AeT);CHKERRQ(ierr);
    ierr = VecSet(mad->CeScale, mad->scale_max);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mad->CeScale, mad->CeScale, mad->W->Ye);CHKERRQ(ierr);
    ierr = VecSet(mad->W->Ye, 1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMin(mad->CeScale, mad->W->Ye, mad->CeScale);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    ierr = MatTranspose(mad->Ai, MAT_INITIAL_MATRIX, &AiT);CHKERRQ(ierr);
    ierr = VecGetArray(mad->W->Yi, &Ai_norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(AiT, NORM_INFINITY, Ai_norms);CHKERRQ(ierr);
    ierr = VecRestoreArray(mad->W->Yi, &Ai_norms);CHKERRQ(ierr);
    ierr = MatDestroy(&AiT);CHKERRQ(ierr);
    ierr = VecSet(mad->CiScale, mad->scale_max);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mad->CiScale, mad->CiScale, mad->W->Yi);CHKERRQ(ierr);
    ierr = VecSet(mad->W->Yi, 1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMin(mad->CiScale, mad->W->Yi, mad->CiScale);CHKERRQ(ierr);
  }

  /* unfortunately we have to recompute the whole Lagrangian and gradient with the new scaling factors */
  ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Q, mad->L, mad->dLdQ);CHKERRQ(ierr);
  /* check convergence at the scaled initial point */
  ierr = TaoMADCheckConvergence(tao, mad->L, mad->dLdQ, 0.0);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* store first iterate as the previous solution for future update */
  ierr = VecCopy(mad->Q->F, mad->Qprev->F);CHKERRQ(ierr);
  ierr = VecCopy(mad->dLdQ->F, mad->dLdQprev->F);CHKERRQ(ierr);
  ierr = VecCopy(mad->G->R, mad->Gprev->R);CHKERRQ(ierr);
  if (mad->dLdQ->Y) {
    ierr = VecDot(mad->dLdQ->Y, mad->dLdQ->Y, &cnorm2);CHKERRQ(ierr);
  } else {
    cnorm2 = 0.0;
  }
  ierr = TaoMADUpdateFilter(tao, mad->L->obj, mad->L->barrier, cnorm2);CHKERRQ(ierr);

  /* first iteration is a simple gradient descent step */
  ++tao->niter;
  ierr = TaoMADComputeReducedKKT(tao, mad->Q, mad->dLdQ, mad->G);CHKERRQ(ierr);
  ierr = VecCopy(mad->G->R, mad->D->R);CHKERRQ(ierr);
  ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
  ierr = TaoMADEvaluateClosedFormUpdates(tao, mad->Q, mad->dLdQ, mad->D);CHKERRQ(ierr);
  /* check the new point against the filter (only contains initial iterate here) */
  ierr = TaoMADApplyFilterStep(tao, mad->Q, mad->D, mad->L, mad->dLdQ, &alpha);CHKERRQ(ierr);
  ierr = TaoMADCheckConvergence(tao, mad->L, mad->dLdQ, alpha);
  if (alpha == 0.0) tao->reason = TAO_DIVERGED_LS_FAILURE;
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    /* Lagrangian and gradient were updated during filter search but we need to update barrier and get reduced KKT */
    ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);
    ierr = TaoMADComputeReducedKKT(tao, mad->Q, mad->dLdQ, mad->G);CHKERRQ(ierr);

    /* update the iterate history with the reduced KKT information */
    if (mad->k == mad->kmax) {
      /* the history is full so we need to shift */
      Qtmp = mad->QR[0];
      Gtmp = mad->GR[0];
      for (i=0; i<mad->kmax-1; i++) {
        mad->QR[i] = mad->QR[i+1];
        mad->GR[i] = mad->GR[i+1];
      }
      mad->QR[mad->kmax-1] = Qtmp;
      mad->GR[mad->kmax-1] = Gtmp;
      mad->k -= 1;
    }
    ierr = VecWAXPY(mad->QR[mad->k], -1.0, mad->Qprev->R, mad->Q->R);CHKERRQ(ierr);
    ierr = VecWAXPY(mad->GR[mad->k], -1.0, mad->Gprev->R, mad->G->R);CHKERRQ(ierr);
    ierr = VecAXPY(mad->GR[mad->k], mad->beta, mad->QR[mad->k]);CHKERRQ(ierr);  /* Tikhonov regularization */
    mad->k += 1;
    mad->nupdates += 1;

    /* store current point as previous for future updates */
    ierr = VecCopy(mad->Q->F, mad->Qprev->F);CHKERRQ(ierr);
    ierr = VecCopy(mad->dLdQ->F, mad->dLdQprev->F);CHKERRQ(ierr);
    ierr = VecCopy(mad->G->R, mad->Gprev->R);CHKERRQ(ierr);
    ierr = LagrangianCopy(mad->L, mad->Lprev);CHKERRQ(ierr);

    /* create LS solution vector */
    ierr = VecDestroy(&mad->gamma);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject)tao), &mad->gamma);CHKERRQ(ierr);
    ierr = VecSetType(mad->gamma, VECSEQ);CHKERRQ(ierr);
    ierr = VecSetSizes(mad->gamma, PETSC_DECIDE, mad->k);CHKERRQ(ierr);
    ierr = VecSetUp(mad->gamma);CHKERRQ(ierr);

    /* solve the least squares problem either with LAPACK truncated SVD or with KSPCG on Normal equations */
    switch (mad->subsolver_type) {
      case TAO_MAD_SUBSOLVER_SVD:
        /* first we need to put the R basis and the RHS vector into 1D arrays copied on all processes */
        for (i=0; i<mad->k; i++) {
          ierr = VecNestConcatenate(mad->GR[i], &Gtmp, NULL);CHKERRQ(ierr);
          ierr = VecScatterBegin(mad->allgather, Gtmp, mad->Gseq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecScatterEnd(mad->allgather, Gtmp, mad->Gseq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecDestroy(&Gtmp);CHKERRQ(ierr);
          ierr = VecGetArray(mad->Gseq, &gg);CHKERRQ(ierr);
          for (j=0; j<mad->Nr; j++) mad->GRarr[i*mad->Nr + j] = gg[j];
          ierr = VecRestoreArray(mad->Gseq, &gg);CHKERRQ(ierr);
        }
        ierr = VecNestConcatenate(mad->G->R, &Gtmp, NULL);CHKERRQ(ierr);
        ierr = VecScatterBegin(mad->allgather, Gtmp, mad->Gseq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(mad->allgather, Gtmp, mad->Gseq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecDestroy(&Gtmp);CHKERRQ(ierr);
        ierr = VecGetArray(mad->Gseq, &rhs);CHKERRQ(ierr);
        for (j=0; j<mad->Nr; j++) mad->rhs[j] = rhs[j];
        ierr = VecRestoreArray(mad->Gseq, &rhs);CHKERRQ(ierr);
        /* trigger LAPACK least squares solver with truncated SVD */
        ierr = PetscBLASIntCast(mad->k, &mad->nsize);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(PetscMax(mad->Nr, mad->k), &mad->ldb);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&mad->msize,&mad->nsize,&mad->nrhs,mad->GRarr,&mad->lda,mad->rhs,&mad->ldb,mad->sigma,&mad->rcond,&mad->rank,mad->work,&mad->lwork,&mad->info));
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        if (mad->info < 0) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_LIB,"Bad argument to GELSS");
        if (mad->info > 0) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_LIB,"SVD failed to converge");
        /* extract the solution out of the RHS vector */
        ierr = VecGetArray(mad->gamma, &ga);CHKERRQ(ierr);
        for (i=0; i<mad->k; i++) ga[i] = mad->rhs[i];
        ierr = VecRestoreArray(mad->gamma, &ga);CHKERRQ(ierr);
        break;

      case TAO_MAD_SUBSOLVER_LSQR:
        /* TODO: implement this!! */
        break;

      case TAO_MAD_SUBSOLVER_NORMAL:
      default:
        /* construct the matrix for the least squares problem */
        ierr = MatDestroy(&mad->GRmat);CHKERRQ(ierr);
        ierr = MatCreate(PetscObjectComm((PetscObject)tao), &mad->GRmat);CHKERRQ(ierr);
        ierr = MatSetType(mad->GRmat, MATSEQDENSE);
        ierr = MatSetSizes(mad->GRmat, PETSC_DECIDE, PETSC_DECIDE, mad->k, mad->k);CHKERRQ(ierr);
        ierr = MatSetUp(mad->GRmat);CHKERRQ(ierr);
        ierr = PetscCalloc2(mad->k, &idx, mad->k, &riTr);CHKERRQ(ierr);
        for (i=0; i<mad->k; i++) idx[i] = i;
        for (i=0; i<mad->k; i++) {
          ierr = VecMTDot(mad->GR[i], mad->k, mad->GR, riTr);CHKERRQ(ierr);
          ierr = MatSetValues(mad->GRmat, 1, &i, mad->k, idx, riTr, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(mad->GRmat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mad->GRmat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = PetscFree2(idx, riTr);CHKERRQ(ierr);
        /* construct the RHS vector */
        ierr = VecCreate(PetscObjectComm((PetscObject)tao), &mad->RHS);CHKERRQ(ierr);
        ierr = VecSetType(mad->RHS, VECSEQ);CHKERRQ(ierr);
        ierr = VecSetSizes(mad->RHS, PETSC_DECIDE, mad->k);CHKERRQ(ierr);
        ierr = VecSetUp(mad->RHS);CHKERRQ(ierr);
        ierr = VecGetArray(mad->RHS, &rhs);CHKERRQ(ierr);
        ierr = VecMDot(mad->G->R, mad->k, mad->GR, rhs);CHKERRQ(ierr);
        ierr = VecRestoreArray(mad->RHS, &rhs);CHKERRQ(ierr);
        /* solve for gamma using KSPCG */
        ierr = KSPReset(tao->ksp);CHKERRQ(ierr);
        ierr = KSPSetOperators(tao->ksp, mad->GRmat, mad->GRmat);CHKERRQ(ierr);
        ierr = KSPSolve(tao->ksp, mad->RHS, mad->gamma);CHKERRQ(ierr);
        break;
    }

    /* construct the search direction D = -eta*r - (S - eta*R)*gamma */
    ierr = VecCopy(mad->G->R, mad->D->R);CHKERRQ(ierr);
    if (mad->pre) {
      ierr = MatMult(mad->pre, mad->D->R, mad->D->R);
    }
    ierr = VecScale(mad->D->R, -mad->eta);CHKERRQ(ierr);
    ierr = VecGetArray(mad->gamma, &ga);CHKERRQ(ierr);
    for (i=0; i<mad->k; i++) {
      if (mad->pre) {
        ierr = MatMult(mad->pre, mad->GR[i], mad->W->R);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(mad->GR[i], mad->W->R);CHKERRQ(ierr);
      }
      ierr = VecAXPBYPCZ(mad->D->R, 1.0, -ga[i], mad->eta*ga[i], mad->QR[i], mad->W->R);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(mad->gamma, &ga);CHKERRQ(ierr);
    ierr = TaoMADEvaluateClosedFormUpdates(tao, mad->Q, mad->dLdQ, mad->D);CHKERRQ(ierr);

    /* check the new point against the filter*/
    ierr = TaoMADApplyFilterStep(tao, mad->Q, mad->D, mad->L, mad->dLdQ, &alpha);CHKERRQ(ierr);
    if (alpha == 0.0) {
      /* couldn't find a valid step length, reset back to previous solution */
      ierr = VecCopy(mad->Qprev->F, mad->Q->F);CHKERRQ(ierr);
      ierr = VecCopy(mad->dLdQprev->F, mad->dLdQ->F);CHKERRQ(ierr);
      ierr = VecCopy(mad->Gprev->R, mad->G->R);CHKERRQ(ierr);
      ierr = LagrangianCopy(mad->Lprev, mad->L);CHKERRQ(ierr);
      /* now we fallback to a simple gradient descent step */
      ierr = VecCopy(mad->G->R, mad->D->R);CHKERRQ(ierr);
      ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
      ierr = TaoMADEvaluateClosedFormUpdates(tao, mad->Q, mad->dLdQ, mad->D);CHKERRQ(ierr);
      ierr = TaoMADApplyFilterStep(tao, mad->Q, mad->D, mad->L, mad->dLdQ, &alpha);CHKERRQ(ierr);
      if (alpha != 0.0) {
        /* gradient descent worked so let's just reset MAD and start fresh */
        mad->k = 0;
        mad->nresets += 1;
      }
    }
    ierr = TaoMADCheckConvergence(tao, mad->L, mad->dLdQ, alpha);
    if (alpha == 0.0) tao->reason = TAO_DIVERGED_LS_FAILURE;
    if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_MAD(Tao tao,PetscViewer viewer)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "MAD history size: %i\n", mad->kmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "MAD updates: %i\n", mad->nupdates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "MAD resets: %i\n", mad->nresets);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_MAD(Tao tao)
{
  TAO_MAD              *mad = (TAO_MAD*)tao->data;
  Vec                  Wtmp;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) PetscFunctionReturn(0);
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);

  /* start with the basics...aliasing */
  mad->Q->X = tao->solution;
  ierr = VecGetSize(mad->Q->X, &mad->Nx);CHKERRQ(ierr);
  ierr = VecDuplicate(mad->Q->X, &mad->G->X);CHKERRQ(ierr);
  mad->G->nR=1;  mad->Q->nR=1;  mad->Q->nF=1;  mad->Q->nP=1;  mad->Q->nS=0;  mad->Q->nY=0;
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  mad->dFdX = tao->gradient;
  mad->Gscale = 1.0;

  /* check all the constraints types and create necessary vectors */
  if (tao->eq_constrained) {
    mad->unconstrained = PETSC_FALSE;
    mad->Ce = tao->constraints_equality;
    mad->Ae = tao->jacobian_equality;
    ierr = VecGetSize(mad->Ce, &mad->Ne);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->Q->Ye);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->G->Ye);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->CeScale);CHKERRQ(ierr);
    ierr = VecSet(mad->CeScale, 1.0);CHKERRQ(ierr);
    mad->G->nR += 1; mad->Q->nR += 1;  mad->Q->nF += 1;  mad->Q->nY += 1;
  }
  if (tao->ineq_constrained) {
    mad->unconstrained = PETSC_FALSE;
    mad->Ci = tao->constraints_inequality;
    mad->Ai = tao->jacobian_inequality;
    ierr = VecGetSize(mad->Ci, &mad->Ni);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->Q->Yi);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->Q->Sc);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->B);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->G->Yi);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->CiScale);CHKERRQ(ierr);
    ierr = VecSet(mad->CiScale, 1.0);CHKERRQ(ierr);
    mad->G->nR += 1; mad->Q->nR += 1;  mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1; mad->Q->nY += 1;
    if (!tao->ineq_doublesided) {
      /* user did not define lower/upper bounds so we assume c_i(x) <= 0 */
      ierr = VecDuplicate(mad->Ci, &tao->IU);CHKERRQ(ierr);
      ierr = VecSet(tao->IU, 0.0);CHKERRQ(ierr);
    }
    if (tao->IL) {
      ierr = VecSet(mad->Q->Yi, PETSC_NINFINITY);CHKERRQ(ierr);
      ierr = VecWhichGreaterThan(tao->IL, mad->Q->Yi, &mad->isIL);CHKERRQ(ierr);
      ierr = ISGetSize(mad->isIL, &mad->Ncl);CHKERRQ(ierr);
      ierr = VecGetSubVector(tao->IL, mad->isIL, &Wtmp);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Scl);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Vl);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->IL);CHKERRQ(ierr);
      ierr = VecCopy(Wtmp, mad->IL);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Ci, mad->isIL, &Wtmp);CHKERRQ(ierr);
      mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
    }
    if (tao->IU) {
      ierr = VecSet(mad->Q->Yi, PETSC_INFINITY);CHKERRQ(ierr);
      ierr = VecWhichGreaterThan(mad->Q->Yi, tao->IU, &mad->isIU);CHKERRQ(ierr);
      ierr = ISGetSize(mad->isIU, &mad->Ncu);CHKERRQ(ierr);
      ierr = VecGetSubVector(tao->IU, mad->isIU, &Wtmp);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Scu);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Vu);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->IU);CHKERRQ(ierr);
      ierr = VecCopy(Wtmp, mad->IU);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(mad->Ci, mad->isIU, &Wtmp);CHKERRQ(ierr);
      mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
    }
  }
  if (tao->bounded) {
    mad->unconstrained = PETSC_FALSE;
    /* we need to determine index sets for bound constraints that are infinity */
    if (tao->XL) {
      ierr = VecSet(mad->dFdX, PETSC_NINFINITY);CHKERRQ(ierr);
      ierr = VecWhichGreaterThan(tao->XL, mad->dFdX, &mad->isXL);CHKERRQ(ierr);
      ierr = ISGetSize(mad->isXL, &mad->Nxl);CHKERRQ(ierr);
      ierr = VecGetSubVector(tao->XL, mad->isXL, &Wtmp);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Sxl);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Zl);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->XL);CHKERRQ(ierr);
      ierr = VecCopy(Wtmp, mad->XL);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(tao->solution, mad->isXL, &Wtmp);CHKERRQ(ierr);
      mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
    }
    if (tao->XU) {
      ierr = VecSet(mad->dFdX, PETSC_INFINITY);CHKERRQ(ierr);
      ierr = VecWhichGreaterThan(mad->dFdX, tao->XU, &mad->isXU);CHKERRQ(ierr);
      ierr = ISGetSize(mad->isXU, &mad->Nxu);CHKERRQ(ierr);
      ierr = VecGetSubVector(tao->XU, mad->isXU, &Wtmp);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Sxu);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->Q->Zu);CHKERRQ(ierr);
      ierr = VecDuplicate(Wtmp, &mad->XU);CHKERRQ(ierr);
      ierr = VecCopy(Wtmp, mad->XU);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(tao->solution, mad->isXU, &Wtmp);CHKERRQ(ierr);
      mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
    }
  }
  /* compute sizing of combined vectors */
  mad->Ns = mad->Ni + mad->Ncl + mad->Ncu + mad->Nxl + mad->Nxu;
  mad->Np = mad->Nx + mad->Ns;
  mad->Ny = mad->Ne + mad->Ni + mad->Ncl + mad->Ncu + mad->Nxl + mad->Nxu;
  mad->Nf = mad->Np + mad->Ny;
  mad->Nr = mad->Nx + mad->Ne + mad->Ni;

  /* at this point we should have created all the base vectors for Q */
  /* now we need to construct the VECNEST combinations */
  ierr = FullSpaceVecCreate(mad->Q);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->Qprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->Qtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->D);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQ);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->W);CHKERRQ(ierr);

  /* create reduced space counterpart */
  ierr = ReducedSpaceVecCreate(mad->G);CHKERRQ(ierr);
  ierr = ReducedSpaceVecDuplicate(mad->G, mad->Gprev);CHKERRQ(ierr);

  /* finally we have to create the MAD history */
  ierr = VecDuplicateVecs(mad->Q->R, mad->kmax, &mad->QR);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->G->R, mad->kmax, &mad->GR);CHKERRQ(ierr);

  /* determine workspace size needed for LAPACK */
  if (mad->subsolver_type == TAO_MAD_SUBSOLVER_SVD) {
    ierr = PetscCalloc3(mad->Nr*mad->kmax, &mad->GRarr, mad->kmax, &mad->rhs, PetscMin(mad->Nr, mad->kmax), &mad->sigma);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(mad->Nr, &mad->msize);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(mad->kmax, &mad->nsize);CHKERRQ(ierr);
    mad->lda = mad->msize;
    mad->ldb = mad->msize;
    mad->nrhs = 1;
    mad->info = 0;
    mad->lwork = 12*mad->msize;
    ierr = PetscCalloc1(mad->lwork, &mad->work);CHKERRQ(ierr);
    ierr = VecNestConcatenate(mad->G->R, &Wtmp, NULL);CHKERRQ(ierr);
    ierr = VecScatterCreateToAll(Wtmp, &mad->allgather, &mad->Gseq);CHKERRQ(ierr);
    ierr = VecDestroy(&Wtmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  SimpleFilter   *filter = mad->filter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy vector structures */
  ierr = FullSpaceVecDestroy(mad->Q);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->Qprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->Qtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->D);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQ);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->W);CHKERRQ(ierr);
  ierr = ReducedSpaceVecDestroy(mad->G);CHKERRQ(ierr);
  ierr = ReducedSpaceVecDestroy(mad->Gprev);CHKERRQ(ierr);
  /* destroy MAD history arrays */
  ierr = VecDestroyVecs(mad->kmax, &mad->QR);CHKERRQ(ierr);
  ierr = VecDestroyVecs(mad->kmax, &mad->GR);CHKERRQ(ierr);
  /* destroy index sets and intermediate vectors */
  ierr = ISDestroy(&mad->isXL);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->isXU);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->isIL);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->isIU);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->XL);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->XU);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->IL);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->IU);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->B);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->CeScale);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->CiScale);CHKERRQ(ierr);
  /* destroy filter */
  ierr = PetscFree2(filter->f, filter->h);CHKERRQ(ierr);
  ierr = PetscFree(filter);CHKERRQ(ierr);
  /* destroy Lagrangians */
  ierr = PetscFree(mad->L);CHKERRQ(ierr);
  ierr = PetscFree(mad->Lprev);CHKERRQ(ierr);
  ierr = PetscFree(mad->Ltrial);CHKERRQ(ierr);
  /* destroy subsolver data */
  if (mad->subsolver_type == TAO_MAD_SUBSOLVER_SVD) {
    ierr = PetscFree3(mad->GRarr, mad->rhs, mad->sigma);CHKERRQ(ierr);
    ierr = PetscFree(mad->work);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&mad->allgather);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->Gseq);CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_MAD(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Multisecant Accelerated Descent for solving noisy nonlinear optimization problems with general constraints.");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_mad_hist_size","number of iterates stored for the MAD approximation","",mad->kmax,&mad->kmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_eta","step length safeguard","",mad->eta,&mad->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_beta","Tikhonov regularization factor","",mad->beta,&mad->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_max_scale","maximum scaling factor for primal gradient and dual variables","",mad->scale_max,&mad->scale_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_mad_filter_type","globalize the solution with a simple filter","TaoMADFilterType",TaoMADFilters,(PetscEnum)mad->filter_type,(PetscEnum*)&mad->filter_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_step_min","minimum step length for the filter globalization","",mad->alpha_min,&mad->alpha_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_step_factor","backtracking factor for the step length","",mad->alpha_fac,&mad->alpha_fac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_mad_subsolver","solution method for the Least-Squares subproblem","TaoMADSubsolver",TaoMADSubsolvers,(PetscEnum)mad->subsolver_type,(PetscEnum*)&mad->subsolver_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_ls_rcond","conditioning tolerance for truncated SVD","",mad->rcond,&mad->rcond,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*MC
  TaoMAD - Multisecant Accelerated Descent method for solving nonlinear optimization problems with general constraints.

  Options Database Keys:
+ -tao_mad_hist_size <int>                    - number of iterates stored for the MAD approximation (default: 30)
. -tao_mad_eta <real>                         - preconditioning factor (default: 0.1)
. -tao_mad_beta <real>                        - Tikhonov regularization factor (default: 0.5)
. -tao_mad_max_scale <real>                   - maximum scaling factor for primal gradient and dual variables (default: 100.0)
. -tao_mad_barrier_scale <real>               - affine scaling/centering factor for log-barrier parameter updates (default: 0.1)
. -tao_mad_barrier_steplength <real>          - steplength for the log-barrier parameter updates (default: 0.95)
. -tao_mad_filter_type <none,barrier,markov>  - filter type for globalization (default: none)
. -tao_mad_max_filter_size <int>              - maximum number of iterates stored in the filter (default: 300)
. -tao_mad_step_min <real>                    - minimum step length for filter globalization (default: 1e-8)
. -tao_mad_step_shrink <real>                 - backtracking interval for the step length (default: 0.1)
. -tao_mad_step_factor <real>                 - backtracking factor for the step length (default: 0.5)
. -tao_mad_armijo_epsilon <real>              - Armijo condition factor for backtracking (default: 1e-6)
. -tao_mad_subsolver <normal, svd>            - solution method for the Least-Squares subproblem (default: normal)
- -tao_mad_ls_rcond <real>                    - conditioning tolerance for truncated SVD (default: 1e-8)

  Level: advanced

  Notes:
  MAD is an experimental optimization method that uses a "multisecant" approximation of the KKT matrix for
  solving generally constrained optimization problems that exhibit noise/errors in the gradient.

  This implementation supports general inequality constraints with an interior-point formulation where the slack
  variables are reduced out of the problem with closed-form updates. The multisecant approximation is constructed
  only for the optimization variables and Lagrange multipliers. It also supports two-sided inequality constraints
  defined with TaoSetInequalityBounds(). When no inequality bounds are set, the algorithm assumes that the constraint
  is c_i(x) <= 0.

  .vb
  while unconverged
    update Q with q_k - q_{k-1}
    update G with g_k - g_{k-1} + beta(q_k - q_{k-1})
    solve gamma = argmin || g_k - G gamma||
    construct search direction d = -eta g - (Q - eta G)gamma
    globalize primal variables with a simple filter
    update multipliers with fraction-to-the-boundary rule
  endwhile
  .ve

.seealso:
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_MAD(Tao tao)
{
  TAO_MAD          *mad;
  SimpleFilter     *filter;
  Lagrangian       *L, *Lprev, *Ltrial;
  ReducedSpaceVec  *G, *Gprev;
  FullSpaceVec     *Q, *Qprev, *Qtrial, *D;
  FullSpaceVec     *dLdQ, *dLdQprev, *dLdQtrial, *W;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao, &mad);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &G);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Gprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Q);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Qprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Qtrial);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &D);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQ);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQtrial);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &W);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &filter);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &L);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Lprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Ltrial);CHKERRQ(ierr);

  tao->ops->destroy        = TaoDestroy_MAD;
  tao->ops->setup          = TaoSetUp_MAD;
  tao->ops->setfromoptions = TaoSetFromOptions_MAD;
  tao->ops->view           = TaoView_MAD;
  tao->ops->solve          = TaoSolve_MAD;

  tao->gatol = 1.e-5;
  tao->grtol = 0.0;
  tao->gttol = 0.0;
  tao->catol = 1.e-5;
  tao->crtol = 0.0;

  tao->data           = (void*)mad;
  mad->mu             = 100.0;
  mad->mu_r           = 0.95;
  mad->mu_g           = 0.1;
  mad->mu_min         = PETSC_MACHINE_EPSILON;
  mad->mu_max         = 1e5;
  mad->k              = 0;
  mad->kmax           = 30;
  mad->eta            = 0.5;
  mad->beta           = 0.2;
  mad->rcond          = 1e-8;
  mad->scale_max      = 100.0;
  mad->slack_init     = 1.0;
  mad->alpha_min      = 1e-5;
  mad->alpha_fac      = 0.5;
  mad->tau_min        = 0.99;
  mad->suff_decr      = 1e-6;
  mad->unconstrained  = PETSC_TRUE;
  mad->subsolver_type = TAO_MAD_SUBSOLVER_SVD;
  mad->pre            = NULL;

  mad->Q              = Q;
  mad->Qprev          = Qprev;
  mad->Qtrial         = Qtrial;
  mad->D              = D;
  mad->dLdQ           = dLdQ;
  mad->dLdQprev       = dLdQprev;
  mad->dLdQtrial      = dLdQtrial;
  mad->W              = W;
  mad->G              = G;
  mad->Gprev          = Gprev;
  mad->L              = L;
  mad->Lprev          = Lprev;
  mad->Ltrial         = Ltrial;

  mad->filter_type    = TAO_MAD_FILTER_BARRIER;
  mad->filter         = filter;
  filter->max_size    = 10*mad->kmax;
  filter->size        = 0;

  /*  set linear solver to default for symmetric matrices */
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp,"tao_mad_");CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp, KSPCG);

  PetscFunctionReturn(0);
}