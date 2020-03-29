#include <petsctaolinesearch.h>
#include <../src/tao/constrained/impls/as/pdas.h>
#include <petscsnes.h>

/*
   TaoPDASEvaluateFunctionsAndJacobians - Evaluate the objective function f, gradient fx, constraints, and all the Jacobians at current vector

   Collective on tao

   Input Parameter:
+  tao - solver context
-  x - vector at which all objects to be evaluated

   Level: beginner

.seealso: TaoPDASUpdateConstraints(), TaoPDASSetUpBounds()
*/
PetscErrorCode TaoPDASEvaluateFunctionsAndJacobians(Tao tao,Vec x)
{
  PetscErrorCode ierr;
  TAO_PDAS      *pdas=(TAO_PDAS*)tao->data;

  PetscFunctionBegin;
  /* Compute user objective function and gradient */
  ierr = TaoComputeObjectiveAndGradient(tao,x,&pdas->obj,tao->gradient);CHKERRQ(ierr);

  /* Equality constraints and Jacobian */
  if (pdas->Ng) {
    ierr = TaoComputeEqualityConstraints(tao,x,tao->constraints_equality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao,x,tao->jacobian_equality,tao->jacobian_equality_pre);CHKERRQ(ierr);
  }

  /* Inequality constraints and Jacobian */
  if (pdas->Nh) {
    ierr = TaoComputeInequalityConstraints(tao,x,tao->constraints_inequality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao,x,tao->jacobian_inequality,tao->jacobian_inequality_pre);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  TaoPDASUpdateConstraints - Update the vectors ce and ci at x

  Collective

  Input Parameter:
+ tao - Tao context
- x - vector at which constraints to be evaluted

   Level: beginner

.seealso: TaoPDASEvaluateFunctionsAndJacobians()
*/
PetscErrorCode TaoPDASUpdateConstraints(Tao tao,Vec x)
{
  PetscErrorCode    ierr;
  TAO_PDAS         *pdas=(TAO_PDAS*)tao->data;
  PetscInt          i,offset,offset1,k,xstart;
  PetscScalar       *carr;
  const PetscInt    *ubptr,*lbptr,*bxptr,*fxptr;
  const PetscScalar *xarr,*xuarr,*xlarr,*garr,*harr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(x,&xstart,NULL);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x,&xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XU,&xuarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XL,&xlarr);CHKERRQ(ierr);

  /* (1) Update ce vector */
  ierr = VecGetArray(pdas->ce,&carr);CHKERRQ(ierr);

  if(pdas->Ng) {
    /* (1.a) Inserting updated g(x) */
    ierr = VecGetArrayRead(tao->constraints_equality,&garr);CHKERRQ(ierr);
    ierr = PetscMemcpy(carr,garr,pdas->ng*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(tao->constraints_equality,&garr);CHKERRQ(ierr);
  }

  /* (1.b) Update xfixed */
  if (pdas->Nxfixed) {
    offset = pdas->ng;
    ierr = ISGetIndices(pdas->isxfixed,&fxptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0;k < pdas->nxfixed;k++){
      i = fxptr[k]-xstart;
      carr[offset + k] = xarr[i] - xuarr[i];
    }
  }
  ierr = VecRestoreArray(pdas->ce,&carr);CHKERRQ(ierr);

  /* (2) Update ci vector */
  ierr = VecGetArray(pdas->ci,&carr);CHKERRQ(ierr);

  if(pdas->Nh) {
    /* (2.a) Inserting updated h(x) */
    ierr = VecGetArrayRead(tao->constraints_inequality,&harr);CHKERRQ(ierr);
    ierr = PetscMemcpy(carr,harr,pdas->nh*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(tao->constraints_inequality,&harr);CHKERRQ(ierr);
  }

  /* (2.b) Update xub */
  offset = pdas->nh;
  if (pdas->Nxub) {
    ierr = ISGetIndices(pdas->isxub,&ubptr);CHKERRQ(ierr);
    for (k=0; k<pdas->nxub; k++){
      i = ubptr[k]-xstart;
      carr[offset + k] = xuarr[i] - xarr[i];
    }
  }

  if (pdas->Nxlb) {
    /* (2.c) Update xlb */
    offset += pdas->nxub;
    ierr = ISGetIndices(pdas->isxlb,&lbptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0; k<pdas->nxlb; k++){
      i = lbptr[k]-xstart;
      carr[offset + k] = xarr[i] - xlarr[i];
    }
  }

  if (pdas->Nxbox) {
    /* (2.d) Update xbox */
    offset += pdas->nxlb;
    offset1 = offset + pdas->nxbox;
    ierr = ISGetIndices(pdas->isxbox,&bxptr);CHKERRQ(ierr); /* global indices in x */
    for (k=0; k<pdas->nxbox; k++){
      i = bxptr[k]-xstart; /* local indices in x */
      carr[offset+k]  = xuarr[i] - xarr[i];
      carr[offset1+k] = xarr[i]  - xlarr[i];
    }
  }
  ierr = VecRestoreArray(pdas->ci,&carr);CHKERRQ(ierr);

  /* Restoring Vectors */
  ierr = VecRestoreArrayRead(x,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XU,&xuarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XL,&xlarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TaoPDASSetUpBounds - Create upper and lower bound vectors of x

   Collective

   Input Parameter:
.  tao - holds pdas and XL & XU

   Level: beginner

.seealso: TaoPDASUpdateConstraints
*/
PetscErrorCode TaoPDASSetUpBounds(Tao tao)
{
  PetscErrorCode    ierr;
  TAO_PDAS         *pdas=(TAO_PDAS*)tao->data;
  const PetscScalar *xl,*xu;
  PetscInt          n,*ixlb,*ixub,*ixfixed,*ixfree,*ixbox,i,low,high,idx;
  MPI_Comm          comm;
  PetscInt          sendbuf[5],recvbuf[5];

  PetscFunctionBegin;
  /* Creates upper and lower bounds vectors on x, if not created already */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);

  ierr = VecGetLocalSize(tao->XL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc5(n,&ixlb,n,&ixub,n,&ixfree,n,&ixfixed,n,&ixbox);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->XL,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XL,&xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XU,&xu);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    idx = low + i;
    if((PetscRealPart(xl[i]) > PETSC_NINFINITY) && (PetscRealPart(xu[i]) < PETSC_INFINITY)) {
      if (PetscRealPart(xl[i]) == PetscRealPart(xu[i])) {
        ixfixed[pdas->nxfixed++]  = idx;
      } else ixbox[pdas->nxbox++] = idx;
    } else {
      if ((PetscRealPart(xl[i]) > PETSC_NINFINITY) && (PetscRealPart(xu[i]) >= PETSC_INFINITY)) {
        ixlb[pdas->nxlb++] = idx;
      } else if ((PetscRealPart(xl[i]) <= PETSC_NINFINITY) && (PetscRealPart(xu[i]) < PETSC_INFINITY)) {
        ixub[pdas->nxlb++] = idx;
      } else  ixfree[pdas->nxfree++] = idx;
    }
  }
  ierr = VecRestoreArrayRead(tao->XL,&xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XU,&xu);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  sendbuf[0] = pdas->nxlb;
  sendbuf[1] = pdas->nxub;
  sendbuf[2] = pdas->nxfixed;
  sendbuf[3] = pdas->nxbox;
  sendbuf[4] = pdas->nxfree;

  ierr = MPI_Allreduce(sendbuf,recvbuf,5,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  pdas->Nxlb    = recvbuf[0];
  pdas->Nxub    = recvbuf[1];
  pdas->Nxfixed = recvbuf[2];
  pdas->Nxbox   = recvbuf[3];
  pdas->Nxfree  = recvbuf[4];

  if (pdas->Nxlb) {
    ierr = ISCreateGeneral(comm,pdas->nxlb,ixlb,PETSC_COPY_VALUES,&pdas->isxlb);CHKERRQ(ierr);
  }
  if (pdas->Nxub) {
    ierr = ISCreateGeneral(comm,pdas->nxub,ixub,PETSC_COPY_VALUES,&pdas->isxub);CHKERRQ(ierr);
  }
  if (pdas->Nxfixed) {
    ierr = ISCreateGeneral(comm,pdas->nxfixed,ixfixed,PETSC_COPY_VALUES,&pdas->isxfixed);CHKERRQ(ierr);
  }
  if (pdas->Nxbox) {
    ierr = ISCreateGeneral(comm,pdas->nxbox,ixbox,PETSC_COPY_VALUES,&pdas->isxbox);CHKERRQ(ierr);
  }
  if (pdas->Nxfree) {
    ierr = ISCreateGeneral(comm,pdas->nxfree,ixfree,PETSC_COPY_VALUES,&pdas->isxfree);CHKERRQ(ierr);
  }
  ierr = PetscFree5(ixlb,ixub,ixfixed,ixbox,ixfree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TaoPDASInitializeSolution - Initialize PDAS solution X = [x; lambdae; lambdai; z].
   X consists of four subvectors in the order [x; lambdae; lambdai; z]. These
     four subvectors need to be initialized and its values copied over to X. Instead
     of copying, we use VecPlace/ResetArray functions to share the memory locations for
     X and the subvectors

   Collective

   Input Parameter:
.  tao - Tao context

   Level: beginner
*/
PetscErrorCode TaoPDASInitializeSolution(Tao tao)
{
  PetscErrorCode ierr;
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  PetscScalar    *Xarr,*z,*lambdai;
  PetscInt       i;
  const PetscScalar *xarr,*ci;

  PetscFunctionBegin;
  ierr = VecGetArray(pdas->X,&Xarr);CHKERRQ(ierr);

  /* Set Initialize X.x = tao->solution */
  ierr = VecGetArrayRead(tao->solution,&xarr);CHKERRQ(ierr);
  ierr = PetscMemcpy(Xarr,xarr,pdas->nx*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->solution,&xarr);CHKERRQ(ierr);

  /* Initialize X.lambdae = 0.0 */
  ierr = VecSet(pdas->lambdae,0.0);CHKERRQ(ierr);

  /* Initialize X.lambdai = push_init_lambdai */
  ierr = VecSet(pdas->lambdai,pdas->push_init_lambdai);CHKERRQ(ierr);

  /* Compute constraints */
  ierr = TaoPDASUpdateConstraints(tao,tao->solution);CHKERRQ(ierr);

  ierr = VecSet(pdas->z,1.0);CHKERRQ(ierr);

  /* Additional modification for X.lambdai and X.z */
  ierr = VecGetArray(pdas->lambdai,&lambdai);CHKERRQ(ierr);
  ierr = VecGetArray(pdas->z,&z);CHKERRQ(ierr);
  if(pdas->Nci) {
    ierr = VecGetArrayRead(pdas->ci,&ci);CHKERRQ(ierr);
    for (i=0; i < pdas->nci; i++) {
      if (ci[i] < PETSC_SMALL) {
	/* Active constraint */
	z[i] = 0.0;
	pdas->idxineq_act[i] = 1;
      }
      //      else z[i] = ci[i];
      /*      if (pdas->mu/z[i] > pdas->push_init_lambdai) lambdai[i] = pdas->mu/z[i]; */
    }
    ierr = VecRestoreArrayRead(pdas->ci,&ci);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pdas->lambdai,&lambdai);CHKERRQ(ierr);
  ierr = VecRestoreArray(pdas->z,&z);CHKERRQ(ierr);

  ierr = VecRestoreArray(pdas->X,&Xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TaoSNESJacobian_PDAS - Evaluate the Hessian matrix at X

   Input Parameter:
   snes - SNES context
   X - KKT Vector
   *ctx - pdas context

   Output Parameter:
   J - Hessian matrix
   Jpre - Preconditioner
*/
PetscErrorCode TaoSNESJacobian_PDAS(SNES snes,Vec X, Mat J, Mat Jpre, void *ctx)
{
  PetscErrorCode    ierr;
  Tao               tao=(Tao)ctx;
  TAO_PDAS         *pdas = (TAO_PDAS*)tao->data;
  PetscInt          i,row,cols[2],Jrstart,rjstart,nc,j;
  const PetscInt    *aj,*ranges,*Jranges,*rranges,*cranges;
  const PetscScalar *Xarr,*aa;
  PetscScalar       vals[2];
  PetscInt          proc,nx_all,*nce_all=pdas->nce_all;
  MPI_Comm          comm;
  PetscMPIInt       rank,size;
  Mat               jac_equality_trans=pdas->jac_equality_trans,jac_inequality_trans=pdas->jac_inequality_trans;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&size);CHKERRQ(ierr);

  ierr = MatGetOwnershipRanges(Jpre,&Jranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Jpre,&Jrstart,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&rranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&cranges);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X,&Xarr);CHKERRQ(ierr);

  /* (2) insert Z and Ci to Jpre -- overwrite existing values */
  for (i=0; i < pdas->nci; i++) {
    row     = Jrstart + pdas->off_z + i;
    cols[0] = Jrstart + pdas->off_lambdai + i;
    cols[1] = row;
    if(!pdas->idxineq_act[i]) {
      vals[0] = Xarr[pdas->off_z + i];
      vals[1] = Xarr[pdas->off_lambdai + i];
    } else {
      vals[0] = 0.0;
      vals[1] = 1.0;
    }
    ierr = MatSetValues(Jpre,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* (3) insert 2nd row block of Jpre: [ grad g, 0, 0, 0] */
  if(pdas->Ng) {
    ierr = MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i<pdas->ng; i++){
      row = Jrstart + pdas->off_lambdae + i;
      
      ierr = MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }
  }

  if(pdas->Nh) {
    /* (4) insert 3nd row block of Jpre: [ grad h, 0, 0, 0] */
    ierr = MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdas->nh; i++){
      row = Jrstart + pdas->off_lambdai + i;
      
      ierr = MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        cols[0] = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
    ierr = MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }
  }

  /* (5) insert Wxx, grad g' and -grad h' to Jpre */
  if(pdas->Ng) {
    ierr = MatTranspose(tao->jacobian_equality,MAT_REUSE_MATRIX,&jac_equality_trans);CHKERRQ(ierr);
  }
  if(pdas->Nh) {
    ierr = MatTranspose(tao->jacobian_inequality,MAT_REUSE_MATRIX,&jac_inequality_trans);CHKERRQ(ierr);
  }

  ierr = VecPlaceArray(pdas->x,Xarr);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,pdas->x,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  ierr = VecResetArray(pdas->x);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(tao->hessian,&rjstart,NULL);CHKERRQ(ierr);
  for (i=0; i<pdas->nx; i++){
    row = Jrstart + i;

    /* insert Wxx */
    ierr = MatGetRow(tao->hessian,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      cols[0] = aj[j] - cranges[proc] + Jranges[proc];
      ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);

    if(pdas->ng) {
      /* insert grad g' */
      ierr = MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_equality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        ierr = MatSetValue(Jpre,row,cols[0],aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }

    if(pdas->nh) {
      /* insert -grad h' */
      ierr = MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_inequality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        cols[0] = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        ierr = MatSetValue(Jpre,row,cols[0],-aa[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,&aa);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(X,&Xarr);CHKERRQ(ierr);

  /* (6) assemble Jpre and J */
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   TaoSnesFunction_PDAS - Evaluate KKT function at X

   Input Parameter:
   snes - SNES context
   X - KKT Vector
   *ctx - pdas

   Output Parameter:
   F - Updated Lagrangian vector
*/
PetscErrorCode TaoSNESFunction_PDAS(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Tao            tao=(Tao)ctx;
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  PetscScalar    *Farr;
  Vec            x,L1;
  PetscInt       i;
  PetscReal      res[2],cnorm[2];
  const PetscScalar *Xarr,*carr,*zarr,*larr;

  PetscFunctionBegin;
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = VecGetArrayRead(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArray(F,&Farr);CHKERRQ(ierr);

  /* (0) Evaluate f, fx, Gx, Hx at X.x Note: pdas->x is not changed below */
  x = pdas->x;
  ierr = VecPlaceArray(x,Xarr);CHKERRQ(ierr);
  ierr = TaoPDASEvaluateFunctionsAndJacobians(tao,x);CHKERRQ(ierr);

  /* Update ce, ci, and Jci at X.x */
  ierr = TaoPDASUpdateConstraints(tao,x);CHKERRQ(ierr);
  ierr = VecResetArray(x);CHKERRQ(ierr);

  /* (1) L1 = fx + (gradG'*DE + Jce_xfixed'*lambdae_xfixed) - (gradH'*DI + Jci_xb'*lambdai_xb) */
  L1 = pdas->x;
  ierr = VecPlaceArray(L1,Farr);CHKERRQ(ierr);
  if (pdas->Nci) {
    if(pdas->Nh) {
      /* L1 += gradH'*DI. Note: tao->DI is not changed below */
      ierr = VecPlaceArray(tao->DI,Xarr+pdas->off_lambdai);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(tao->jacobian_inequality,tao->DI,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(tao->DI);CHKERRQ(ierr);
    }

    /* L1 += Jci_xb'*lambdai_xb */
    ierr = VecPlaceArray(pdas->lambdai_xb,Xarr+pdas->off_lambdai+pdas->nh);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(pdas->Jci_xb,pdas->lambdai_xb,L1,L1);CHKERRQ(ierr);
    ierr = VecResetArray(pdas->lambdai_xb);CHKERRQ(ierr);

    /* (1.4) L1 = - (gradH'*DI + Jci_xb'*lambdai_xb) */
    ierr = VecScale(L1,-1.0);CHKERRQ(ierr);
  }

  /* L1 += fx */
  ierr = VecAXPY(L1,1.0,tao->gradient);CHKERRQ(ierr);

  if (pdas->Nce) {
    if(pdas->Ng) {
      /* L1 += gradG'*DE. Note: tao->DE is not changed below */
      ierr = VecPlaceArray(tao->DE,Xarr+pdas->off_lambdae);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(tao->jacobian_equality,tao->DE,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(tao->DE);CHKERRQ(ierr);
    }
    if (pdas->Nxfixed) {
      /* L1 += Jce_xfixed'*lambdae_xfixed */
      ierr = VecPlaceArray(pdas->lambdae_xfixed,Xarr+pdas->off_lambdae+pdas->ng);CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(pdas->Jce_xfixed,pdas->lambdae_xfixed,L1,L1);CHKERRQ(ierr);
      ierr = VecResetArray(pdas->lambdae_xfixed);CHKERRQ(ierr);
    }
  }
  ierr = VecNorm(L1,NORM_2,&res[0]);CHKERRQ(ierr);
  ierr = VecResetArray(L1);CHKERRQ(ierr);

  /* (2) L2 = ce(x) */
  if (pdas->Nce) {
    ierr = VecGetArrayRead(pdas->ce,&carr);CHKERRQ(ierr);
    for (i=0; i<pdas->nce; i++) Farr[pdas->off_lambdae + i] = carr[i];
    ierr = VecRestoreArrayRead(pdas->ce,&carr);CHKERRQ(ierr);
  }
  ierr = VecNorm(pdas->ce,NORM_2,&cnorm[0]);CHKERRQ(ierr);

  if (pdas->Nci) {
    /* (3) L3 = ci(x) - z;
       (4) L4 = Z * Lambdai * e if constraint inactive, else L4 = z (forcing z = 0)
    */
    ierr = VecGetArrayRead(pdas->ci,&carr);CHKERRQ(ierr);
    larr = Xarr+pdas->off_lambdai;
    zarr = Xarr+pdas->off_z;
    for (i=0; i<pdas->nci; i++) {
      Farr[pdas->off_lambdai + i] = carr[i] - zarr[i];
      if(pdas->idxineq_act[i]) {
	Farr[pdas->off_z + i] = zarr[i];
      } else {
	Farr[pdas->off_z       + i] = zarr[i]*larr[i];
      }
    }
    ierr = VecRestoreArrayRead(pdas->ci,&carr);CHKERRQ(ierr);
  }

  ierr = VecPlaceArray(pdas->ci,Farr+pdas->off_lambdai);CHKERRQ(ierr);
  ierr = VecNorm(pdas->ci,NORM_2,&cnorm[1]);CHKERRQ(ierr);
  ierr = VecResetArray(pdas->ci);CHKERRQ(ierr);

  /* note: pdas->z is not changed below */
  ierr = VecPlaceArray(pdas->z,Farr+pdas->off_z);CHKERRQ(ierr);
  ierr = VecNorm(pdas->z,NORM_2,&res[1]);CHKERRQ(ierr);
  ierr = VecResetArray(pdas->z);CHKERRQ(ierr);

  tao->residual = PetscSqrtReal(res[0]*res[0] + res[1]*res[1]);
  tao->cnorm    = PetscSqrtReal(cnorm[0]*cnorm[0] + cnorm[1]*cnorm[1]);

  ierr = VecRestoreArrayRead(X,&Xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&Farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PDASLineSearch - Custom line search used with PDAS.

   Collective on TAO

   Notes:
*/
PetscErrorCode PDASLineSearch(SNESLineSearch linesearch,void *ctx)
{
  PetscErrorCode ierr;
  Tao            tao=(Tao)ctx;
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  SNES           snes;
  Vec            X,F,Y,W,G;
  PetscInt       i,iter;
  PetscReal      alpha_p=1.0,alpha_d=1.0,alpha[4];
  PetscScalar    *Xarr,*z,*lambdai;
  const PetscScalar *dXarr,*dz,*dlambdai;
  PetscScalar    *taosolarr,*ci;

  PetscFunctionBegin;
  ierr = SNESLineSearchGetSNES(linesearch,&snes);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);

  ierr = SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);
  ierr = SNESLineSearchGetVecs(linesearch,&X,&F,&Y,&W,&G);CHKERRQ(ierr);

  ierr = VecGetArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&dXarr);CHKERRQ(ierr);

  z  = Xarr + pdas->off_z;
  dz = dXarr + pdas->off_z;

  for (i=0; i < pdas->nci; i++) {
    /* Inactive constraint */
    if (!pdas->idxineq_act[i]) {
      if ((z[i] - dz[i]) < PETSC_SMALL) {
	alpha_p = PetscMin(alpha_p,0.9999*z[i]/dz[i]);
      }
    }
  }

  lambdai  = Xarr + pdas->off_lambdai;
  dlambdai = dXarr + pdas->off_lambdai;

  for (i=0; i<pdas->nci; i++) {
    if ((lambdai[i] - dlambdai[i]) < PETSC_SMALL) {
      alpha_d = PetscMin(0.9999*lambdai[i]/dlambdai[i],alpha_d);
    }
  }

  ierr = VecRestoreArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&dXarr);CHKERRQ(ierr);

  alpha[0] = alpha_p;
  alpha[1] = alpha_d;

  /* alpha = min(alpha) over all processes */
  ierr = MPI_Allreduce(alpha,alpha+2,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)tao));CHKERRQ(ierr);

  alpha_p = alpha[2];
  alpha_d = alpha[3];

  ierr = VecGetArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&dXarr);CHKERRQ(ierr);

  for (i=0; i<pdas->nx; i++) {
    Xarr[i] = Xarr[i] - alpha_p * dXarr[i];
  }

  for (i=0; i<pdas->nce; i++) {
    Xarr[i+pdas->off_lambdae] = Xarr[i+pdas->off_lambdae] - alpha_d * dXarr[i+pdas->off_lambdae];
  }

  for (i=0; i<pdas->nci; i++) {
    Xarr[i+pdas->off_lambdai] = Xarr[i+pdas->off_lambdai] - alpha_d * dXarr[i+pdas->off_lambdai];
  }

  for (i=0; i<pdas->nci; i++) {
    Xarr[i+pdas->off_z] = Xarr[i+pdas->off_z] - alpha_p*dXarr[i+pdas->off_z];
  }

  ierr = VecGetArray(tao->solution,&taosolarr);CHKERRQ(ierr);
  ierr = PetscMemcpy(taosolarr,Xarr,pdas->nx*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(tao->solution,&taosolarr);CHKERRQ(ierr);

  /* Compute constraints */
  ierr = TaoPDASUpdateConstraints(tao,tao->solution);CHKERRQ(ierr);
  ierr = VecGetArray(pdas->ci,&ci);CHKERRQ(ierr);

  z = Xarr + pdas->off_z;
  for(i=0; i < pdas->nci; i++) {
    if(!pdas->idxineq_act[i]) {
      if(z[i] < -PETSC_SMALL) {
	pdas->idxineq_act[i] = 1;
	z[i] = 0.0;
      }
    } else {
      if(ci[i] > PETSC_SMALL) {
	pdas->idxineq_act[i] = 0;
	z[i] = ci[i];
      }
    }
  }
    
  ierr = VecRestoreArray(X,&Xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&dXarr);CHKERRQ(ierr);

  VecView(tao->solution,0);
  ierr = VecRestoreArray(pdas->ci,&ci);CHKERRQ(ierr);
  /* Evaluate F at X */
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr); /* must call this func, do not know why */

  /* Update F; get tao->residual and tao->cnorm */
  ierr = TaoSNESFunction_PDAS(snes,X,F,(void*)tao);CHKERRQ(ierr);

  tao->niter++;
  ierr = TaoLogConvergenceHistory(tao,pdas->obj,tao->residual,tao->cnorm,tao->niter);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,pdas->obj,tao->residual,tao->cnorm,0.0);CHKERRQ(ierr);

  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason) {
    ierr = SNESSetConvergedReason(snes,SNES_CONVERGED_FNORM_ABS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   TaoSolve_PDAS

   Input Parameter:
   tao - TAO context

   Output Parameter:
   tao - TAO context
*/
PetscErrorCode TaoSolve_PDAS(Tao tao)
{
  PetscErrorCode     ierr;
  TAO_PDAS          *pdas = (TAO_PDAS*)tao->data;
  SNESLineSearch     linesearch;  /* SNESLineSearch context */
  Vec                dummy;

  PetscFunctionBegin;
  /* Initialize all variables */
  ierr = TaoPDASInitializeSolution(tao);CHKERRQ(ierr);

  /* Set linesearch */
  ierr = SNESGetLineSearch(pdas->snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHSHELL);CHKERRQ(ierr);
  ierr = SNESLineSearchShellSetUserFunc(linesearch,PDASLineSearch,tao);CHKERRQ(ierr);
  ierr = SNESLineSearchSetFromOptions(linesearch);CHKERRQ(ierr);

  tao->reason = TAO_CONTINUE_ITERATING;

  /* -tao_monitor for iteration 0 and check convergence */
  ierr = VecDuplicate(pdas->X,&dummy);CHKERRQ(ierr);
  ierr = TaoSNESFunction_PDAS(pdas->snes,pdas->X,dummy,(void*)tao);CHKERRQ(ierr);

  ierr = TaoLogConvergenceHistory(tao,pdas->obj,tao->residual,tao->cnorm,tao->niter);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,pdas->obj,tao->residual,tao->cnorm,0.0);CHKERRQ(ierr);
  ierr = VecDestroy(&dummy);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason) {
    ierr = SNESSetConvergedReason(pdas->snes,SNES_CONVERGED_FNORM_ABS);CHKERRQ(ierr);
  }

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    SNESConvergedReason reason;
    ierr = SNESSolve(pdas->snes,NULL,pdas->X);CHKERRQ(ierr);

    /* Check SNES convergence */
    ierr = SNESGetConvergedReason(pdas->snes,&reason);CHKERRQ(ierr);
    if (reason < 0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"SNES solve did not converged due to reason %D\n",reason);CHKERRQ(ierr);
    }

    /* Check TAO convergence */
    if (PetscIsInfOrNanReal(pdas->obj)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"User-provided compute function generated Inf or NaN");
  }
  PetscFunctionReturn(0);
}

/*
   TaoSetup_PDAS - Sets up tao and pdas

   Input Parameter:
   tao - TAO object

   Output:   pdas - initialized object
*/
PetscErrorCode TaoSetup_PDAS(Tao tao)
{
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    rank,size;
  PetscInt       row,col,Jcrstart,Jcrend,k,tmp,nc,proc,*nh_all,*ng_all;
  PetscInt       offset,*xa,*xb,i,j,rstart,rend;
  PetscScalar    one=1.0,neg_one=-1.0,*Xarr;
  const PetscInt    *cols,*rranges,*cranges,*aj,*ranges;
  const PetscScalar *aa;
  Mat            J,jac_equality_trans,jac_inequality_trans;
  Mat            Jce_xfixed_trans,Jci_xb_trans;
  PetscInt       *dnz,*onz,rjstart,nx_all,*nce_all,*Jranges,cols1[2];

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* (1) Setup Bounds and create Tao vectors */
  ierr = TaoPDASSetUpBounds(tao);CHKERRQ(ierr);

  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  }

  /* (2) Get sizes */
  /* Size of vector x - This is set by TaoSetInitia√lVector */
  ierr = VecGetSize(tao->solution,&pdas->Nx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tao->solution,&pdas->nx);CHKERRQ(ierr);

  /* Size of equality constraints and vectors */
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&pdas->Ng);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->constraints_equality,&pdas->ng);CHKERRQ(ierr);
  } else {
    pdas->ng = pdas->Ng = 0;
  }

  pdas->nce = pdas->ng + pdas->nxfixed;
  pdas->Nce = pdas->Ng + pdas->Nxfixed;

  /* Size of inequality constraints and vectors */
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&pdas->Nh);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->constraints_inequality,&pdas->nh);CHKERRQ(ierr);
  } else {
    pdas->nh = pdas->Nh = 0;
  }

  pdas->nci = pdas->nh + pdas->nxlb + pdas->nxub + 2*pdas->nxbox;
  pdas->Nci = pdas->Nh + pdas->Nxlb + pdas->Nxub + 2*pdas->Nxbox;

  /* Full size of the KKT system to be solved */
  pdas->n = pdas->nx + pdas->nce + 2*pdas->nci;
  pdas->N = pdas->Nx + pdas->Nce + 2*pdas->Nci;

  /* list below to TaoView_PDAS()? */
  /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] nce %d = ng %d + nxfixed %d\n",rank,pdas->nce,pdas->ng,pdas->nxfixed); */
  /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] nci %d = nh %d + nxlb %d + nxub %d + 2*nxbox %d\n",rank,pdas->nci,pdas->nh,pdas->nxlb,pdas->nxub,pdas->nxbox); */
  /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] n %d = nx %d + nce %d + 2*nci %d\n",rank,pdas->n,pdas->nx,pdas->nce,pdas->nci); */

  /* (3) Offsets for subvectors */
  pdas->off_lambdae = pdas->nx;
  pdas->off_lambdai = pdas->off_lambdae + pdas->nce;
  pdas->off_z       = pdas->off_lambdai + pdas->nci;

  /* (4) Create vectors and subvectors */
  /* Ce and Ci vectors */
  ierr = VecCreate(comm,&pdas->ce);CHKERRQ(ierr);
  ierr = VecSetSizes(pdas->ce,pdas->nce,pdas->Nce);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdas->ce);CHKERRQ(ierr);

  ierr = VecCreate(comm,&pdas->ci);CHKERRQ(ierr);
  ierr = VecSetSizes(pdas->ci,pdas->nci,pdas->Nci);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdas->ci);CHKERRQ(ierr);

  /* X=[x; lambdae; lambdai; z] for the big KKT system */
  ierr = VecCreate(comm,&pdas->X);CHKERRQ(ierr);
  ierr = VecSetSizes(pdas->X,pdas->n,pdas->N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdas->X);CHKERRQ(ierr);

  /* Subvectors; they share local arrays with X */
  ierr = VecGetArray(pdas->X,&Xarr);CHKERRQ(ierr);
  /* x shares local array with X.x */
  if (pdas->Nx) {
    ierr = VecCreateMPIWithArray(comm,1,pdas->nx,pdas->Nx,Xarr,&pdas->x);CHKERRQ(ierr);
  }

  /* lambdae shares local array with X.lambdae */
  if (pdas->Nce) {
    ierr = VecCreateMPIWithArray(comm,1,pdas->nce,pdas->Nce,Xarr+pdas->off_lambdae,&pdas->lambdae);CHKERRQ(ierr);
  }

  /* tao->DE shares local array with X.lambdae_g */
  if (pdas->Ng) {
    ierr = VecCreateMPIWithArray(comm,1,pdas->ng,pdas->Ng,Xarr+pdas->off_lambdae,&tao->DE);CHKERRQ(ierr);

    ierr = VecCreate(comm,&pdas->lambdae_xfixed);CHKERRQ(ierr);
    ierr = VecSetSizes(pdas->lambdae_xfixed,pdas->nxfixed,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(pdas->lambdae_xfixed);CHKERRQ(ierr);
  }

  if (pdas->Nci) {
    /* lambdai shares local array with X.lambdai */
    ierr = VecCreateMPIWithArray(comm,1,pdas->nci,pdas->Nci,Xarr+pdas->off_lambdai,&pdas->lambdai);CHKERRQ(ierr);

    /* z for slack variables; it shares local array with X.z */
    ierr = VecCreateMPIWithArray(comm,1,pdas->nci,pdas->Nci,Xarr+pdas->off_z,&pdas->z);CHKERRQ(ierr);

    /* Create index set for active set */
    ierr = PetscCalloc1(pdas->nci,&pdas->idxineq_act);CHKERRQ(ierr);
  }

  /* tao->DI which shares local array with X.lambdai_h */
  if (pdas->Nh) {
    ierr = VecCreateMPIWithArray(comm,1,pdas->nh,pdas->Nh,Xarr+pdas->off_lambdai,&tao->DI);CHKERRQ(ierr);
  }

  ierr = VecCreate(comm,&pdas->lambdai_xb);CHKERRQ(ierr);
  ierr = VecSetSizes(pdas->lambdai_xb,(pdas->nci - pdas->nh),PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(pdas->lambdai_xb);CHKERRQ(ierr);

  ierr = VecRestoreArray(pdas->X,&Xarr);CHKERRQ(ierr);

  /* (5) Create Jacobians Jce_xfixed and Jci */
  /* (5.1) PDAS Jacobian of equality bounds cebound(x) = J_nxfixed */
  if (pdas->Nxfixed) {
    /* Create Jce_xfixed */
    ierr = MatCreate(comm,&pdas->Jce_xfixed);CHKERRQ(ierr);
    ierr = MatSetSizes(pdas->Jce_xfixed,pdas->nxfixed,pdas->nx,PETSC_DECIDE,pdas->Nx);CHKERRQ(ierr);
    ierr = MatSetFromOptions(pdas->Jce_xfixed);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(pdas->Jce_xfixed,1,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(pdas->Jce_xfixed,1,NULL,1,NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(pdas->Jce_xfixed,&Jcrstart,&Jcrend);CHKERRQ(ierr);
    ierr = ISGetIndices(pdas->isxfixed,&cols);CHKERRQ(ierr);
    k = 0;
    for (row = Jcrstart; row < Jcrend; row++) {
      ierr = MatSetValues(pdas->Jce_xfixed,1,&row,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdas->isxfixed, &cols);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pdas->Jce_xfixed,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pdas->Jce_xfixed,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* (5.2) PDAS inequality Jacobian Jci = [tao->jacobian_inequality; ...] */
  ierr = MatCreate(comm,&pdas->Jci_xb);CHKERRQ(ierr);
  ierr = MatSetSizes(pdas->Jci_xb,pdas->nci-pdas->nh,pdas->nx,PETSC_DECIDE,pdas->Nx);CHKERRQ(ierr);
  ierr = MatSetFromOptions(pdas->Jci_xb);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(pdas->Jci_xb,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(pdas->Jci_xb,1,NULL,1,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(pdas->Jci_xb,&Jcrstart,&Jcrend);CHKERRQ(ierr);
  offset = Jcrstart;
  if (pdas->Nxub) {
    /* Add xub to Jci_xb */
    ierr = ISGetIndices(pdas->isxub,&cols);CHKERRQ(ierr);
    k = 0;
    for (row = offset; row < offset + pdas->nxub; row++) {
      ierr = MatSetValues(pdas->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdas->isxub, &cols);CHKERRQ(ierr);
  }

  if (pdas->Nxlb) {
    /* Add xlb to Jci_xb */
    ierr = ISGetIndices(pdas->isxlb,&cols);CHKERRQ(ierr);
    k = 0;
    offset += pdas->nxub;
    for (row = offset; row < offset + pdas->nxlb; row++) {
      ierr = MatSetValues(pdas->Jci_xb,1,&row,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdas->isxlb, &cols);CHKERRQ(ierr);
  }

  /* Add xbox to Jci_xb */
  if (pdas->Nxbox) {
    ierr = ISGetIndices(pdas->isxbox,&cols);CHKERRQ(ierr);
    k = 0;
    offset += pdas->nxlb;
    for (row = offset; row < offset + pdas->nxbox; row++) {
      ierr = MatSetValues(pdas->Jci_xb,1,&row,1,cols+k,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
      tmp = row + pdas->nxbox;
      ierr = MatSetValues(pdas->Jci_xb,1,&tmp,1,cols+k,&one,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
    ierr = ISRestoreIndices(pdas->isxbox, &cols);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(pdas->Jci_xb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pdas->Jci_xb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* ierr = MatView(pdas->Jci_xb,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* (6) Set up ISs for PC Fieldsplit */
  if (pdas->solve_reduced_kkt) {
    ierr = PetscMalloc2(pdas->nx+pdas->nce,&xa,2*pdas->nci,&xb);CHKERRQ(ierr);
    for(i=0; i < pdas->nx + pdas->nce; i++) xa[i] = i;
    for(i=0; i < 2*pdas->nci; i++) xb[i] = pdas->off_lambdai + i;

    ierr = ISCreateGeneral(comm,pdas->nx+pdas->nce,xa,PETSC_OWN_POINTER,&pdas->is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,2*pdas->nci,xb,PETSC_OWN_POINTER,&pdas->is2);CHKERRQ(ierr);
  }

  /* (7) Gather offsets from all processes */
  ierr = PetscMalloc1(size,&pdas->nce_all);CHKERRQ(ierr);

  /* Get rstart of KKT matrix */
  ierr = MPI_Scan(&pdas->n,&rstart,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  rstart -= pdas->n;

  ierr = MPI_Allgather(&pdas->nce,1,MPIU_INT,pdas->nce_all,1,MPIU_INT,comm);CHKERRQ(ierr);

  ierr = PetscMalloc3(size,&ng_all,size,&nh_all,size,&Jranges);CHKERRQ(ierr);
  ierr = MPI_Allgather(&rstart,1,MPIU_INT,Jranges,1,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&pdas->nh,1,MPIU_INT,nh_all,1,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&pdas->ng,1,MPIU_INT,ng_all,1,MPIU_INT,comm);CHKERRQ(ierr);

  ierr = MatGetOwnershipRanges(tao->hessian,&rranges);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(tao->hessian,&cranges);CHKERRQ(ierr);

  if (pdas->Ng) {
    ierr = TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre);CHKERRQ(ierr);
    ierr = MatTranspose(tao->jacobian_equality,MAT_INITIAL_MATRIX,&pdas->jac_equality_trans);CHKERRQ(ierr);
  }
  if (pdas->Nh) {
    ierr = TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre);CHKERRQ(ierr);
    ierr = MatTranspose(tao->jacobian_inequality,MAT_INITIAL_MATRIX,&pdas->jac_inequality_trans);CHKERRQ(ierr);
  }

  /* Count dnz,onz for preallocation of KKT matrix */
  jac_equality_trans   = pdas->jac_equality_trans;
  jac_inequality_trans = pdas->jac_inequality_trans;
  nce_all = pdas->nce_all;

  if (pdas->Nxfixed) {
    ierr = MatTranspose(pdas->Jce_xfixed,MAT_INITIAL_MATRIX,&Jce_xfixed_trans);CHKERRQ(ierr);
  }
  ierr = MatTranspose(pdas->Jci_xb,MAT_INITIAL_MATRIX,&Jci_xb_trans);CHKERRQ(ierr);

  ierr = MatPreallocateInitialize(comm,pdas->n,pdas->n,dnz,onz);CHKERRQ(ierr);

  /* 1st row block of KKT matrix: [Wxx; gradCe'; -gradCi'; 0] */
  ierr = TaoPDASEvaluateFunctionsAndJacobians(tao,pdas->x);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);

  /* Insert tao->hessian */
  ierr = MatGetOwnershipRange(tao->hessian,&rjstart,NULL);CHKERRQ(ierr);
  for (i=0; i<pdas->nx; i++){
    row = rstart + i;

    ierr = MatGetRow(tao->hessian,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (aj[j] >= cranges[proc+1]) proc++;
      col = aj[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->hessian,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);

    if(pdas->ng) {
      /* Insert grad g' */
      ierr = MatGetRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_equality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all;
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_equality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    /* Insert Jce_xfixed^T' */
    if (pdas->nxfixed) {
      ierr = MatGetRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(pdas->Jce_xfixed,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + ng_all[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(Jce_xfixed_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    if(pdas->nh) {
      /* Insert -grad h' */
      ierr = MatGetRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(tao->jacobian_inequality,&ranges);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        /* find row ownership of */
        while (aj[j] >= ranges[proc+1]) proc++;
        nx_all = rranges[proc+1] - rranges[proc];
        col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(jac_inequality_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }

    /* Insert Jci_xb^T' */
    ierr = MatGetRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(pdas->Jci_xb,&ranges);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      /* find row ownership of */
      while (aj[j] >= ranges[proc+1]) proc++;
      nx_all = rranges[proc+1] - rranges[proc];
      col = aj[j] - ranges[proc] + Jranges[proc] + nx_all + nce_all[proc] + nh_all[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(Jci_xb_trans,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
  }

  /* 2nd Row block of KKT matrix: [grad Ce, 0, 0, 0] */
  if(pdas->Ng) {
    ierr = MatGetOwnershipRange(tao->jacobian_equality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdas->ng; i++){
      row = rstart + pdas->off_lambdae + i;

      ierr = MatGetRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr); /* grad g */
      }
      ierr = MatRestoreRow(tao->jacobian_equality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }
  }
  /* Jce_xfixed */
  if (pdas->Nxfixed) {
    ierr = MatGetOwnershipRange(pdas->Jce_xfixed,&Jcrstart,NULL);CHKERRQ(ierr);
    for (i=0; i < (pdas->nce - pdas->ng); i++ ){
      row = rstart + pdas->off_lambdae + pdas->ng + i;

      ierr = MatGetRow(pdas->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
      if (nc != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");

      proc = 0;
      j    = 0;
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(pdas->Jce_xfixed,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    }
  }

  /* 3rd Row block of KKT matrix: [ gradCi, 0, 0, -I] */
  if(pdas->Nh) {
    ierr = MatGetOwnershipRange(tao->jacobian_inequality,&rjstart,NULL);CHKERRQ(ierr);
    for (i=0; i < pdas->nh; i++){
      row = rstart + pdas->off_lambdai + i;

      ierr = MatGetRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (aj[j] >= cranges[proc+1]) proc++;
        col = aj[j] - cranges[proc] + Jranges[proc];
        ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr); /* grad h */
      }
      ierr = MatRestoreRow(tao->jacobian_inequality,i+rjstart,&nc,&aj,NULL);CHKERRQ(ierr);
    }
    /* -I */
    for (i=0; i < pdas->nh; i++){
      row = rstart + pdas->off_lambdai + i;
      col = rstart + pdas->off_z + i;
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
  }

  /* Jci_xb */
  ierr = MatGetOwnershipRange(pdas->Jci_xb,&Jcrstart,NULL);CHKERRQ(ierr);
  for (i=0; i < (pdas->nci - pdas->nh); i++ ){
    row = rstart + pdas->off_lambdai + pdas->nh + i;

    ierr = MatGetRow(pdas->Jci_xb,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    if (nc != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"nc != 1");
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(pdas->Jci_xb,i+Jcrstart,&nc,&cols,NULL);CHKERRQ(ierr);
    /* -I */
    col = rstart + pdas->off_z + pdas->nh + i;
    ierr = MatPreallocateSet(row,1,&col,dnz,onz);CHKERRQ(ierr);
  }

  /* 4-th Row block of KKT matrix: Z and Ci */
  for (i=0; i < pdas->nci; i++) {
    row     = rstart + pdas->off_z + i;
    cols1[0] = rstart + pdas->off_lambdai + i;
    cols1[1] = row;
    ierr = MatPreallocateSet(row,2,cols1,dnz,onz);CHKERRQ(ierr);
  }

  /* diagonal entry */
  for (i=0; i<pdas->n; i++) dnz[i]++; /* diagonal entry */

  /* Create KKT matrix */
  ierr = MatCreate(comm,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,pdas->n,pdas->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,0,dnz,0,onz);CHKERRQ(ierr);
  /* ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr); */
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  pdas->K = J;

  /* (8) Set up nonlinear solver SNES */
  ierr = SNESSetFunction(pdas->snes,NULL,TaoSNESFunction_PDAS,(void*)tao);CHKERRQ(ierr);
  ierr = SNESSetJacobian(pdas->snes,J,J,TaoSNESJacobian_PDAS,(void*)tao);CHKERRQ(ierr);

  if (pdas->solve_reduced_kkt) {
    PC pc;
    ierr = KSPGetPC(tao->ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"2",pdas->is2);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"1",pdas->is1);CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(pdas->snes);CHKERRQ(ierr);

  /* (9) Insert constant entries to  K */
  /* Set 0.0 to diagonal of K, so that the solver does not complain *about missing diagonal value */
  ierr = MatGetOwnershipRange(J,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++){
    ierr = MatSetValue(J,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Row block of K: [ grad Ce, 0, 0, 0] */
  if (pdas->Nxfixed) {
    ierr = MatGetOwnershipRange(pdas->Jce_xfixed,&Jcrstart,NULL);CHKERRQ(ierr);
    for (i=0; i < (pdas->nce - pdas->ng); i++ ){
      row = rstart + pdas->off_lambdae + pdas->ng + i;

      ierr = MatGetRow(pdas->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
      proc = 0;
      for (j=0; j < nc; j++) {
        while (cols[j] >= cranges[proc+1]) proc++;
        col = cols[j] - cranges[proc] + Jranges[proc];
        ierr = MatSetValue(J,row,col,aa[j],INSERT_VALUES);CHKERRQ(ierr); /* grad Ce */
        ierr = MatSetValue(J,col,row,aa[j],INSERT_VALUES);CHKERRQ(ierr); /* grad Ce' */
      }
      ierr = MatRestoreRow(pdas->Jce_xfixed,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
    }
  }

  /* Row block of K: [ grad Ci, 0, 0, -I] */
  ierr = MatGetOwnershipRange(pdas->Jci_xb,&Jcrstart,NULL);CHKERRQ(ierr);
  for (i=0; i < (pdas->nci - pdas->nh); i++ ){
    row = rstart + pdas->off_lambdai + pdas->nh + i;

    ierr = MatGetRow(pdas->Jci_xb,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);
    proc = 0;
    for (j=0; j < nc; j++) {
      while (cols[j] >= cranges[proc+1]) proc++;
      col = cols[j] - cranges[proc] + Jranges[proc];
      ierr = MatSetValue(J,col,row,-aa[j],INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(J,row,col,aa[j],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(pdas->Jci_xb,i+Jcrstart,&nc,&cols,&aa);CHKERRQ(ierr);

    col = rstart + pdas->off_z + pdas->nh + i;
    ierr = MatSetValue(J,row,col,-1,INSERT_VALUES);CHKERRQ(ierr);
  }

  for (i=0; i < pdas->nh; i++){
    row = rstart + pdas->off_lambdai + i;
    col = rstart + pdas->off_z + i;
    ierr = MatSetValue(J,row,col,-1,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (pdas->Nxfixed) {
    ierr = MatDestroy(&Jce_xfixed_trans);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Jci_xb_trans);CHKERRQ(ierr);
  ierr = PetscFree3(ng_all,nh_all,Jranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TaoDestroy_PDAS - Destroys the pdas object

   Input:
   full pdas

   Output:
   Destroyed pdas
*/
PetscErrorCode TaoDestroy_PDAS(Tao tao)
{
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Freeing Vectors assocaiated with KKT (X) */
  ierr = VecDestroy(&pdas->x);CHKERRQ(ierr); /* Solution x */
  ierr = VecDestroy(&pdas->lambdae);CHKERRQ(ierr); /* Equality constraints lagrangian multiplier*/
  ierr = VecDestroy(&pdas->lambdai);CHKERRQ(ierr); /* Inequality constraints lagrangian multiplier*/
  ierr = VecDestroy(&pdas->z);CHKERRQ(ierr);       /* Slack variables */
  ierr = VecDestroy(&pdas->X);CHKERRQ(ierr);       /* Big KKT system vector [x; lambdae; lambdai; z] */

  /* work vectors */
  ierr = VecDestroy(&pdas->lambdae_xfixed);CHKERRQ(ierr);
  ierr = VecDestroy(&pdas->lambdai_xb);CHKERRQ(ierr);

  /* Legrangian equality and inequality Vec */
  ierr = VecDestroy(&pdas->ce);CHKERRQ(ierr); /* Vec of equality constraints */
  ierr = VecDestroy(&pdas->ci);CHKERRQ(ierr); /* Vec of inequality constraints */

  /* Matrices */
  ierr = MatDestroy(&pdas->Jce_xfixed);CHKERRQ(ierr);
  ierr = MatDestroy(&pdas->Jci_xb);CHKERRQ(ierr); /* Jacobian of inequality constraints Jci = [tao->jacobian_inequality ; J(nxub); J(nxlb); J(nxbx)] */
  ierr = MatDestroy(&pdas->K);CHKERRQ(ierr);

  /* Index Sets */
  if (pdas->Nxub) {
    ierr = ISDestroy(&pdas->isxub);CHKERRQ(ierr);    /* Finite upper bound only -inf < x < ub */
  }

  if (pdas->Nxlb) {
    ierr = ISDestroy(&pdas->isxlb);CHKERRQ(ierr);    /* Finite lower bound only  lb <= x < inf */
  }

  if (pdas->Nxfixed) {
    ierr = ISDestroy(&pdas->isxfixed);CHKERRQ(ierr); /* Fixed variables         lb =  x = ub */
  }

  if (pdas->Nxbox) {
    ierr = ISDestroy(&pdas->isxbox);CHKERRQ(ierr);   /* Boxed variables         lb <= x <= ub */
  }

  if (pdas->Nxfree) {
    ierr = ISDestroy(&pdas->isxfree);CHKERRQ(ierr);  /* Free variables        -inf <= x <= inf */
  }

  if (pdas->solve_reduced_kkt) {
    ierr = ISDestroy(&pdas->is1);CHKERRQ(ierr);
    ierr = ISDestroy(&pdas->is2);CHKERRQ(ierr);
  }

  if(pdas->Nci) {
    ierr = PetscFree(pdas->idxineq_act);CHKERRQ(ierr);
  }

  /* SNES */
  ierr = SNESDestroy(&pdas->snes);CHKERRQ(ierr); /* Nonlinear solver */
  ierr = PetscFree(pdas->nce_all);CHKERRQ(ierr);
  ierr = MatDestroy(&pdas->jac_equality_trans);CHKERRQ(ierr);
  ierr = MatDestroy(&pdas->jac_inequality_trans);CHKERRQ(ierr);

  /* Destroy pdas */
  ierr = PetscFree(tao->data);CHKERRQ(ierr); /* Holding locations of pdas */

  /* Destroy Dual */
  ierr = VecDestroy(&tao->DE);CHKERRQ(ierr); /* equality dual */
  ierr = VecDestroy(&tao->DI);CHKERRQ(ierr); /* dinequality dual */
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetFromOptions_PDAS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_PDAS      *pdas = (TAO_PDAS*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PDAS method for constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_pdas_push_init_lambdai","parameter to push initial (inequality) dual variables away from bounds",NULL,pdas->push_init_lambdai,&pdas->push_init_lambdai,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_pdas_solve_reduced_kkt","Solve reduced KKT system using Schur-complement",NULL,pdas->solve_reduced_kkt,&pdas->solve_reduced_kkt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  TAOPDAS - Primal-dual active set algorithm for generally constrained optimization.

  Option Database Keys:
+   -tao_pdas_push_init_lambdai - parameter to push initial dual variables away from bounds (> 0)
-   -tao_pdas_mu_update_factor - update scalar for barrier parameter (mu) update (> 0)

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_PDAS(Tao tao)
{
  TAO_PDAS      *pdas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_PDAS;
  tao->ops->solve          = TaoSolve_PDAS;
  tao->ops->setfromoptions = TaoSetFromOptions_PDAS;
  tao->ops->destroy        = TaoDestroy_PDAS;

  ierr = PetscNewLog(tao,&pdas);CHKERRQ(ierr);
  tao->data = (void*)pdas;

  pdas->nx      = pdas->Nx      = 0;
  pdas->nxfixed = pdas->Nxfixed = 0;
  pdas->nxlb    = pdas->Nxlb    = 0;
  pdas->nxub    = pdas->Nxub    = 0;
  pdas->nxbox   = pdas->Nxbox   = 0;
  pdas->nxfree  = pdas->Nxfree  = 0;

  pdas->ng = pdas->Ng = pdas->nce = pdas->Nce = 0;
  pdas->nh = pdas->Nh = pdas->nci = pdas->Nci = 0;
  pdas->n  = pdas->N  = 0;

  pdas->push_init_lambdai = 1.0;
  pdas->solve_reduced_kkt = PETSC_FALSE;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;

  ierr = SNESCreate(((PetscObject)tao)->comm,&pdas->snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(pdas->snes,tao->hdr.prefix);CHKERRQ(ierr);
  ierr = SNESGetKSP(pdas->snes,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
