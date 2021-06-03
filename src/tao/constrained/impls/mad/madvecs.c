#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/

PetscErrorCode LagrangianCopy(Lagrangian *source, Lagrangian *target)
{
  PetscFunctionBegin;
  target->val = source->val;
  target->obj = source->obj;
  target->barrier = source->barrier;
  target->yeTce = source->yeTce;
  target->yiTci = source->yiTci;
  target->vlTcl = source->vlTcl;
  target->vuTcu = source->vuTcu;
  target->zlTxl = source->zlTxl;
  target->zuTxu = source->zuTxu;
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecCreate(FullSpaceVec *Q)
{
  Vec            *Farr, *Rarr, *Parr, *Sarr, *Yarr;
  PetscInt       fi, ri, pi, si, yi;
  MPI_Comm       comm = PetscObjectComm((PetscObject)Q->X);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* first count the sizing of arrays and name the vectors */
  ierr = VecSetOptionsPrefix(Q->X, "mad_x_");CHKERRQ(ierr);
  Q->nF = 1; Q->nR = 1; Q->nP = 1; Q->nS = 0; Q->nY = 0;
  if (Q->Sc) {
    ierr = VecSetOptionsPrefix(Q->Sc, "mad_sc_");CHKERRQ(ierr);
    Q->nF += 1; Q->nP += 1; Q->nS += 1;
  }
  if (Q->Scl) {
    ierr = VecSetOptionsPrefix(Q->Scl, "mad_scl_");CHKERRQ(ierr);
    Q->nF += 1; Q->nP += 1; Q->nS += 1;
  }
  if (Q->Scu) {
    ierr = VecSetOptionsPrefix(Q->Scu, "mad_scu_");CHKERRQ(ierr);
    Q->nF += 1; Q->nP += 1; Q->nS += 1;
  }
  if (Q->Sxl) {
    ierr = VecSetOptionsPrefix(Q->Sxl, "mad_sxl_");CHKERRQ(ierr);
    Q->nF += 1; Q->nP += 1; Q->nS += 1;
  }
  if (Q->Sxu) {
    ierr = VecSetOptionsPrefix(Q->Sxu, "mad_sxu_");CHKERRQ(ierr);
    Q->nF += 1; Q->nP += 1; Q->nS += 1;
    }
  if (Q->Yi) {
    ierr = VecSetOptionsPrefix(Q->Yi, "mad_yi_");CHKERRQ(ierr);
    Q->nF += 1; Q->nR += 1; Q->nY += 1;
  }
  if (Q->Ye)  {
    ierr = VecSetOptionsPrefix(Q->Ye, "mad_ye_");CHKERRQ(ierr);
    Q->nF += 1; Q->nR += 1; Q->nY += 1;
  }
  if (Q->Vl) {
    ierr = VecSetOptionsPrefix(Q->Vl, "mad_vl_");CHKERRQ(ierr);
    Q->nF += 1; Q->nY += 1;
  }
  if (Q->Vu) {
    ierr = VecSetOptionsPrefix(Q->Vu, "mad_vu_");CHKERRQ(ierr);
    Q->nF += 1; Q->nY += 1;
  }
  if (Q->Zl) {
    ierr = VecSetOptionsPrefix(Q->Zl, "mad_zl_");CHKERRQ(ierr);
    Q->nF += 1; Q->nY += 1;
  }
  if (Q->Zu) {
    ierr = VecSetOptionsPrefix(Q->Zu, "mad_zu_");CHKERRQ(ierr);
    Q->nF += 1; Q->nY += 1;
  }
  ierr = PetscMalloc5(Q->nF, &Farr, Q->nR, &Rarr, Q->nP, &Parr, Q->nS, &Sarr, Q->nY, &Yarr);CHKERRQ(ierr);
  fi = 0; ri = 0; pi = 0;  si = 0; yi = 0;
  Rarr[ri++] = Q->X; Farr[fi++] = Q->X; Parr[pi++] = Q->X;
  if (Q->Sc)  { Farr[fi++] = Q->Sc;   Parr[pi++] = Q->Sc;   Sarr[si++] = Q->Sc;  }
  if (Q->Scl) { Farr[fi++] = Q->Scl;  Parr[pi++] = Q->Scl;  Sarr[si++] = Q->Scl; }
  if (Q->Scu) { Farr[fi++] = Q->Scu;  Parr[pi++] = Q->Scu;  Sarr[si++] = Q->Scu; }
  if (Q->Sxl) { Farr[fi++] = Q->Sxl;  Parr[pi++] = Q->Sxl;  Sarr[si++] = Q->Sxl; }
  if (Q->Sxu) { Farr[fi++] = Q->Sxu;  Parr[pi++] = Q->Sxu;  Sarr[si++] = Q->Sxu; }
  if (Q->Yi)  { Farr[fi++] = Q->Yi;   Rarr[ri++] = Q->Yi;   Yarr[yi++] = Q->Yi;  }
  if (Q->Ye)  { Farr[fi++] = Q->Ye;   Rarr[ri++] = Q->Ye;   Yarr[yi++] = Q->Ye;  }
  if (Q->Vl)  { Farr[fi++] = Q->Vl;   Yarr[yi++] = Q->Vl; }
  if (Q->Vu)  { Farr[fi++] = Q->Vu;   Yarr[yi++] = Q->Vu; }
  if (Q->Zl)  { Farr[fi++] = Q->Zl;   Yarr[yi++] = Q->Zl; }
  if (Q->Zu)  { Farr[fi++] = Q->Zu;   Yarr[yi++] = Q->Zu; }
  ierr = VecCreateNest(comm, Q->nF, NULL, Farr, &Q->F);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(Q->F, "mad_q_");CHKERRQ(ierr);
  ierr = VecCreateNest(comm, Q->nR, NULL, Rarr, &Q->R);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(Q->R, "mad_qr_");CHKERRQ(ierr);
  ierr = VecCreateNest(comm, Q->nP, NULL, Parr, &Q->P);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(Q->P, "mad_qp_");CHKERRQ(ierr);
  if (Q->nS > 0) {
    ierr = VecCreateNest(comm, Q->nS, NULL, Sarr, &Q->S);CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(Q->S, "mad_qs_");CHKERRQ(ierr);
  }
  if (Q->nY > 0) {
    ierr = VecCreateNest(comm, Q->nY, NULL, Yarr, &Q->Y);CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(Q->Y, "mad_qy_");CHKERRQ(ierr);
  }
  for (fi=0; fi<Q->nF; fi++) {
    ierr = VecDestroy(&Farr[fi]);CHKERRQ(ierr);
  }
  ierr = PetscFree5(Farr, Rarr, Parr, Sarr, Yarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecDuplicate(FullSpaceVec *source, FullSpaceVec *target)
{
  Vec            *vb, *Rarr, *Parr, *Sarr, *Yarr;
  PetscInt       i=0, ri=0, pi=0, si=0, yi=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(source->nR, &Rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nP, &Parr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nS, &Sarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nY, &Yarr);CHKERRQ(ierr);
  target->nF = source->nF;
  ierr = VecDuplicate(source->F, &target->F);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(target->F, "mad_f_");CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(target->F, NULL, &vb);CHKERRQ(ierr);
  target->X = vb[i++];
  ierr = VecSetOptionsPrefix(target->X, "mad_x_");CHKERRQ(ierr);
  Rarr[ri++] = target->X;
  Parr[pi++] = target->X;
  if (source->Sc) {
    target->Sc = vb[i++];   Parr[pi++] = target->Sc;   Sarr[si++] = target->Sc;
    ierr = VecSetOptionsPrefix(target->Sc, "mad_sc_");CHKERRQ(ierr);
  }
  if (source->Scl) {
    target->Scl = vb[i++];  Parr[pi++] = target->Scl;  Sarr[si++] = target->Scl;
    ierr = VecSetOptionsPrefix(target->Scl, "mad_scl_");CHKERRQ(ierr);
  }
  if (source->Scu) {
    target->Scu = vb[i++];  Parr[pi++] = target->Scu;  Sarr[si++] = target->Scu;
    ierr = VecSetOptionsPrefix(target->Scu, "mad_scu_");CHKERRQ(ierr);
  }
  if (source->Sxl) {
    target->Sxl = vb[i++];  Parr[pi++] = target->Sxl;  Sarr[si++] = target->Sxl;
    ierr = VecSetOptionsPrefix(target->Sxl, "mad_sxl_");CHKERRQ(ierr);
  }
  if (source->Sxu) {
    target->Sxu = vb[i++];  Parr[pi++] = target->Sxu;  Sarr[si++] = target->Sxu;
    ierr = VecSetOptionsPrefix(target->Sxu, "mad_sxu_");CHKERRQ(ierr);
  }
  if (source->Yi) {
    target->Yi = vb[i++];   Rarr[ri++] = target->Yi;   Yarr[yi++] = target->Yi;
    ierr = VecSetOptionsPrefix(target->Yi, "mad_yi_");CHKERRQ(ierr);
  }
  if (source->Ye) {
    target->Ye = vb[i++];   Rarr[ri++] = target->Ye;   Yarr[yi++] = target->Ye;
    ierr = VecSetOptionsPrefix(target->Ye, "mad_ye_");CHKERRQ(ierr);
  }
  if (source->Vl) {
    target->Vl = vb[i++];   Yarr[yi++] = target->Vl;
    ierr = VecSetOptionsPrefix(target->Vl, "mad_vl_");CHKERRQ(ierr);
  }
  if (source->Vu) {
    target->Vu = vb[i++];   Yarr[yi++] = target->Vu;
    ierr = VecSetOptionsPrefix(target->Vu, "mad_vu_");CHKERRQ(ierr);
  }
  if (source->Zl) {
    target->Zl = vb[i++];   Yarr[yi++] = target->Zl;
    ierr = VecSetOptionsPrefix(target->Zl, "mad_zl_");CHKERRQ(ierr);
  }
  if (source->Zu) {
    target->Zu = vb[i++];   Yarr[yi++] = target->Zu;
    ierr = VecSetOptionsPrefix(target->Zu, "mad_zu_");CHKERRQ(ierr);
  }
  target->nR = source->nR;
  ierr = VecCreateNest(PetscObjectComm((PetscObject)source->X), target->nR, NULL, Rarr, &target->R);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(target->R, "mad_r_");CHKERRQ(ierr);
  target->nP = source->nP;
  ierr = VecCreateNest(PetscObjectComm((PetscObject)source->X), target->nP, NULL, Parr, &target->P);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(target->P, "mad_p_");CHKERRQ(ierr);
  target->nS = source->nS;
  if (target->nS > 0) {
    ierr = VecCreateNest(PetscObjectComm((PetscObject)source->X), target->nS, NULL, Sarr, &target->S);CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(target->S, "mad_s_");CHKERRQ(ierr);
  }
  target->nY = source->nY;
  if (target->nY > 0) {
    ierr = VecCreateNest(PetscObjectComm((PetscObject)source->X), target->nY, NULL, Yarr, &target->Y);CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(target->Y, "mad_y_");CHKERRQ(ierr);
  }
  for (pi=0; pi<target->nP; pi++) {
    ierr = VecDestroy(&Parr[pi]);CHKERRQ(ierr);
  }
  for (yi=0; yi<target->nY; yi++) {
    ierr = VecDestroy(&Yarr[yi]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Rarr);CHKERRQ(ierr);
  ierr = PetscFree(Parr);CHKERRQ(ierr);
  ierr = PetscFree(Sarr);CHKERRQ(ierr);
  ierr = PetscFree(Yarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecDestroy(FullSpaceVec *Q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&Q->F);CHKERRQ(ierr);
  ierr = VecDestroy(&Q->R);CHKERRQ(ierr);
  ierr = VecDestroy(&Q->P);CHKERRQ(ierr);
  ierr = VecDestroy(&Q->S);CHKERRQ(ierr);
  ierr = VecDestroy(&Q->Y);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReducedSpaceVecCreate(ReducedSpaceVec *G)
{
  Vec            *Rarr;
  PetscInt       i=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetOptionsPrefix(G->X, "mad_x_");CHKERRQ(ierr);
  ierr = PetscMalloc1(G->nR, &Rarr);CHKERRQ(ierr);
  Rarr[i++] = G->X;
  if (G->Yi) {
    ierr = VecSetOptionsPrefix(G->Yi, "mad_yi_");CHKERRQ(ierr);
    Rarr[i++] = G->Yi;
  }
  if (G->Ye) {
    ierr = VecSetOptionsPrefix(G->Ye, "mad_ye_");CHKERRQ(ierr);
    Rarr[i++] = G->Ye;
  }
  ierr = VecCreateNest(PetscObjectComm((PetscObject)G->X), G->nR, NULL, Rarr, &G->R);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(G->R, "mad_r_");CHKERRQ(ierr);
  for (i=0; i<G->nR; i++) {
    ierr = VecDestroy(&Rarr[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Rarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReducedSpaceVecDuplicate(ReducedSpaceVec *source, ReducedSpaceVec *target)
{
  Vec            *vb;
  PetscInt       i=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(source->R, &target->R);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(target->R, "mad_r_");CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(target->R, &target->nR, &vb);CHKERRQ(ierr);
  target->X = vb[i++];
  if (source->Yi) {
    target->Yi = vb[i++];
    ierr = VecSetOptionsPrefix(target->Yi, "mad_yi_");CHKERRQ(ierr);
  }
  if (source->Ye) {
    target->Ye = vb[i++];
    ierr = VecSetOptionsPrefix(target->Ye, "mad_ye_");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ReducedSpaceVecDestroy(ReducedSpaceVec *Q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&Q->R);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecGetNorms(FullSpaceVec *Q, NormType norm_type, PetscInt *n, PetscReal **norms)
{
  Vec*           vb;
  PetscInt       i, vn;
  PetscReal      norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs(Q->F, &vn, &vb);CHKERRQ(ierr);
  ierr = PetscMalloc1(vn, norms);CHKERRQ(ierr);
  for (i=0; i<vn; i++) {
    ierr = VecNorm(vb[i], norm_type, &norm);CHKERRQ(ierr);
    (*norms)[i] = norm;
  }
  if (n) *n = vn;
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecPrintNorms(FullSpaceVec *vec, NormType norm_type)
{
  PetscReal      *norms;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FullSpaceVecGetNorms(vec, norm_type, &n, &norms);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "||vec[%i]||_%s = %e\n", i, NormTypes[norm_type], norms[i]);
  }
  ierr = PetscFree(norms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}