
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

/*@C
     KSPGMRESClassicalGramSchmidtOrthogonalization -  This is the basic orthogonalization routine
                using classical Gram-Schmidt with possible iterative refinement to improve the stability

     Collective on ksp

  Input Parameters:
+   ksp - KSP object, must be associated with GMRES, FGMRES, or LGMRES Krylov method
-   its - one less then the current GMRES restart iteration, i.e. the size of the Krylov space

   Options Database Keys:
+   -ksp_gmres_classicalgramschmidt - Activates KSPGMRESClassicalGramSchmidtOrthogonalization()
-   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is
                                   used to increase the stability of the classical Gram-Schmidt  orthogonalization.

    Notes:
    Use KSPGMRESSetCGSRefinementType() to determine if iterative refinement is to be used

   Level: intermediate

.seelaso:  KSPGMRESSetOrthogonalization(), KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetCGSRefinementType(),
           KSPGMRESGetCGSRefinementType(), KSPGMRESGetOrthogonalization()

@*/
PetscErrorCode  KSPGMRESClassicalGramSchmidtOrthogonalization(KSP ksp,PetscInt it)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    *hh,*hes,*lhh;
  PetscReal      hnrm, wnrm;
  PetscBool      refine = (PetscBool)(gmres->cgstype == KSP_GMRES_CGS_REFINE_ALWAYS),use_densemats = gmres->preallocate_densemats;
  Vec            lhhvec = NULL;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  if (!gmres->orthogwork) {
    ierr = PetscMalloc1(gmres->max_k + 2,&gmres->orthogwork);CHKERRQ(ierr);
  }
  lhh = gmres->orthogwork;

  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  if (use_densemats) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,it+1,lhh,&lhhvec);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)VEC_VV(0),&comm);CHKERRQ(ierr);
  }

  /* This is really a matrix-vector product */
  if (use_densemats) {
    ierr = VecGetLocalVectorRead(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(gmres->densemats[it],gmres->lvec,lhhvec);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,lhh,it+1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
    ierr = VecRestoreLocalVector(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
  } else {ierr = VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh);CHKERRQ(ierr);} /* <v,vnew> */

  for (j=0; j<=it; j++) {
    KSPCheckDot(ksp,lhh[j]);
    lhh[j] = -lhh[j];
  }

  /*
         This is really a matrix vector product:
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */
  if (use_densemats) {
    ierr = VecGetLocalVector(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
    ierr = MatMultAdd(gmres->densemats[it],lhhvec,gmres->lvec,gmres->lvec);CHKERRQ(ierr);
    ierr = VecRestoreLocalVector(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
  } else {ierr = VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));CHKERRQ(ierr);}

  /* note lhh[j] is -<v,vnew> , hence the subtraction */
  for (j=0; j<=it; j++) {
    hh[j]  -= lhh[j];     /* hh += <v,vnew> */
    hes[j] -= lhh[j];     /* hes += <v,vnew> */
  }

  /*
   *  the second step classical Gram-Schmidt is only necessary
   *  when a simple test criteria is not passed
   */
  if (gmres->cgstype == KSP_GMRES_CGS_REFINE_IFNEEDED) {
    hnrm = 0.0;
    for (j=0; j<=it; j++) hnrm +=  PetscRealPart(lhh[j] * PetscConj(lhh[j]));

    hnrm = PetscSqrtReal(hnrm);
    ierr = VecNorm(VEC_VV(it+1),NORM_2, &wnrm);CHKERRQ(ierr);
    if (wnrm < hnrm) {
      refine = PETSC_TRUE;
      ierr   = PetscInfo2(ksp,"Performing iterative refinement wnorm %g hnorm %g\n",(double)wnrm,(double)hnrm);CHKERRQ(ierr);
    }
  }

  if (refine) {
    if (use_densemats) {
      ierr = VecGetLocalVector(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
      ierr = MatMultHermitianTranspose(gmres->densemats[it],gmres->lvec,lhhvec);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(MPI_IN_PLACE,lhh,it+1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
      for (j=0; j<=it; j++) lhh[j] = -lhh[j];
      ierr = MatMultAdd(gmres->densemats[it],lhhvec,gmres->lvec,gmres->lvec);CHKERRQ(ierr);
      ierr = VecRestoreLocalVector(VEC_VV(it+1),gmres->lvec);CHKERRQ(ierr);
    } else {
      ierr = VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh);CHKERRQ(ierr); /* <v,vnew> */
      for (j=0; j<=it; j++) lhh[j] = -lhh[j];
      ierr = VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));CHKERRQ(ierr);
    }

    /* note lhh[j] is -<v,vnew> , hence the subtraction */
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] -= lhh[j];     /* hes += <v,vnew> */
    }
  }

  if (use_densemats) {ierr = VecDestroy(&lhhvec);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








