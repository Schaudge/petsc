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
  ierr = PetscMalloc1(Q->nF, &Farr);CHKERRQ(ierr);
  ierr = PetscMalloc1(Q->nR, &Rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(Q->nP, &Parr);CHKERRQ(ierr);
  ierr = PetscMalloc1(Q->nS, &Sarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(Q->nY, &Yarr);CHKERRQ(ierr);
  Rarr[0] = Q->X;
  Farr[0] = Q->X;
  Parr[0] = Q->X;
  fi = 0; ri = 0; pi = 0;  si = 0; yi = 0;
  if (Q->Sc) {
    Farr[fi++] = Q->Sc;
    Parr[pi++] = Q->Sc;
    Sarr[si++] = Q->Sc;
  }
  if (Q->Scl) {
    Farr[fi++] = Q->Scl;
    Parr[pi++] = Q->Scl;
    Sarr[si++] = Q->Scl;
  }
  if (Q->Scu) {
    Farr[fi++] = Q->Scu;
    Parr[pi++] = Q->Scu;
    Sarr[si++] = Q->Scu;
  }
  if (Q->Sxl) {
    Farr[fi++] = Q->Sxl;
    Parr[pi++] = Q->Sxl;
    Sarr[si++] = Q->Sxl;
  }
  if (Q->Sxu) {
    Farr[fi++] = Q->Sxu;
    Parr[pi++] = Q->Sxu;
    Sarr[si++] = Q->Sxu;
  }
  if (Q->Yi) {
    Rarr[ri++] = Q->Yi;
    Farr[fi++] = Q->Yi;
    Yarr[yi++] = Q->Yi;
  }
  if (Q->Ye) {
    Rarr[ri++] = Q->Ye;
    Farr[fi++] = Q->Ye;
    Yarr[yi++] = Q->Ye;
  }
  if (Q->Vl) {
    Farr[fi++] = Q->Vl;
    Yarr[yi++] = Q->Vl;
  }
  if (Q->Vu) {
    Farr[fi++] = Q->Vu;
    Yarr[yi++] = Q->Vu;
  }
  if (Q->Zl) {
    Farr[fi++] = Q->Zl;
    Yarr[yi++] = Q->Zl;
  }
  if (Q->Zu) {
    Farr[fi++] = Q->Zu;
    Yarr[yi++] = Q->Zu;
  }
  ierr = VecCreateNest(comm, Q->nF, NULL, Farr, &Q->F);CHKERRQ(ierr);
  ierr = VecCreateNest(comm, Q->nR, NULL, Rarr, &Q->R);CHKERRQ(ierr);
  ierr = VecCreateNest(comm, Q->nP, NULL, Parr, &Q->P);CHKERRQ(ierr);
  if (Q->nS > 0) {
    ierr = VecCreateNest(comm, Q->nS, NULL, Sarr, &Q->S);CHKERRQ(ierr);
  }
  if (Q->nY > 0) {
    ierr = VecCreateNest(comm, Q->nY, NULL, Yarr, &Q->Y);CHKERRQ(ierr);
  }
  for (fi=0; fi<Q->nF; fi++) {
    ierr = VecDestroy(&Farr[fi]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Farr);CHKERRQ(ierr);
  ierr = PetscFree(Rarr);CHKERRQ(ierr);
  ierr = PetscFree(Parr);CHKERRQ(ierr);
  ierr = PetscFree(Sarr);CHKERRQ(ierr);
  ierr = PetscFree(Yarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FullSpaceVecDuplicate(FullSpaceVec *source, FullSpaceVec *target)
{
  Vec            *vb, *Rarr, *Parr, *Sarr, *Yarr;
  PetscInt       i=0, ri=0, pi=0, si=0, yi=0;
  MPI_Comm       comm = PetscObjectComm((PetscObject)source->X);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(source->nR, &Rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nP, &Parr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nS, &Sarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(source->nY, &Yarr);CHKERRQ(ierr);
  target->nF = source->nF;
  ierr = VecDuplicate(source->F, &target->F);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(target->F, NULL, &vb);CHKERRQ(ierr);
  target->X = vb[i++];
  Rarr[ri++] = target->X;
  Parr[pi++] = target->X;
  if (source->Sc) {
    target->Sc = vb[i++];
    Parr[pi++] = target->Sc;
    Sarr[si++] = target->Sc;
  }
  if (source->Scl) {
    target->Scl = vb[i++];
    Parr[pi++] = target->Scl;
    Sarr[si++] = target->Scl;
  }
  if (source->Scu) {
    target->Scu = vb[i++];
    Parr[pi++] = target->Scu;
    Sarr[si++] = target->Scu;
  }
  if (source->Sxl) {
    target->Sxl = vb[i++];
    Parr[pi++] = target->Sxl;
    Sarr[si++] = target->Sxl;
  }
  if (source->Sxu) {
    target->Sxu = vb[i++];
    Parr[pi++] = target->Sxu;
    Sarr[si++] = target->Sxu;
  }
  if (source->Yi) {
    target->Yi = vb[i++];
    Rarr[ri++] = target->Yi;
    Yarr[yi++] = target->Yi;
  }
  if (source->Ye) {
    target->Ye = vb[i++];
    Rarr[ri++] = target->Ye;
    Yarr[yi++] = target->Ye;
  }
  if (source->Vl) {
    target->Vl = vb[i++];
    Yarr[yi++] = target->Vl;
  }
  if (source->Vu) {
    target->Vu = vb[i++];
    Yarr[yi++] = target->Vu;
  }
  if (source->Zl) {
    target->Zl = vb[i++];
    Yarr[yi++] = target->Zl;
  }
  if (source->Zu) {
    target->Zu = vb[i++];
    Yarr[yi++] = target->Zu;
  }
  target->nR = source->nR;
  ierr = VecCreateNest(comm, target->nR, NULL, Rarr, &target->R);CHKERRQ(ierr);
  target->nP = source->nP;
  ierr = VecCreateNest(comm, target->nP, NULL, Parr, &target->P);CHKERRQ(ierr);
  target->nS = source->nS;
  if (target->nS > 0) {
    ierr = VecCreateNest(comm, target->nS, NULL, Sarr, &target->S);CHKERRQ(ierr);
  }
  target->nY = source->nY;
  if (target->nY > 0) {
    ierr = VecCreateNest(comm, target->nY, NULL, Yarr, &target->Y);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(G->nR, &Rarr);CHKERRQ(ierr);
  Rarr[i++] = G->X;
  if (G->Yi) Rarr[i++] = G->Yi;
  if (G->Ye) Rarr[i++] = G->Ye;
  ierr = VecCreateNest(PetscObjectComm((PetscObject)G->X), G->nR, NULL, Rarr, &G->R);CHKERRQ(ierr);
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
  ierr = VecNestGetSubVecs(target->R, &target->nR, &vb);CHKERRQ(ierr);
  target->X = vb[i++];
  if (source->Yi) target->Yi = vb[i++];
  if (source->Ye) target->Ye = vb[i++];
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