#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iguess.c,v 1.26 1999/05/04 20:34:35 balay Exp balay $";
#endif

#include "src/sles/ksp/kspimpl.h"  /*I "ksp.h" I*/
/* 
  This code inplements Paul Fischer's initial guess code for situations where
  a linear system is solved repeatedly 
 */

typedef struct {
    int      curl,     /* Current number of basis vectors */
             maxl;     /* Maximum number of basis vectors */
    Scalar   *alpha;   /* */
    Vec      *xtilde,  /* Saved x vectors */
             *btilde;  /* Saved b vectors */
} KSPIGUESS;

#undef __FUNC__  
#define __FUNC__ "KSPGuessCreate" 
int KSPGuessCreate(KSP ksp,int  maxl,void **ITG )
{
  KSPIGUESS *itg;
  int       ierr;

  *ITG = 0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  itg  = (KSPIGUESS* ) PetscMalloc(sizeof(KSPIGUESS));CHKPTRQ(itg);
  itg->curl = 0;
  itg->maxl = maxl;
  itg->alpha = (Scalar *)PetscMalloc( maxl * sizeof(Scalar) );CHKPTRQ(itg->alpha);
  PLogObjectMemory(ksp,sizeof(KSPIGUESS) + maxl*sizeof(Scalar));
  ierr = VecDuplicateVecs(ksp->vec_rhs,maxl,&itg->xtilde);CHKERRQ(ierr);
  PLogObjectParents(ksp,maxl,itg->xtilde);
  ierr = VecDuplicateVecs(ksp->vec_rhs,maxl,&itg->btilde);CHKERRQ(ierr);
  PLogObjectParents(ksp,maxl,itg->btilde);
  *ITG = (void *)itg;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessDestroy" 
int KSPGuessDestroy( KSP ksp, KSPIGUESS *itg )
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscFree( itg->alpha );CHKERRQ(ierr);
  ierr = VecDestroyVecs( itg->btilde, itg->maxl );CHKERRQ(ierr);
  ierr = VecDestroyVecs( itg->xtilde, itg->maxl );CHKERRQ(ierr);
  ierr = PetscFree( itg );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessFormB"
int KSPGuessFormB( KSP ksp, KSPIGUESS *itg, Vec b )
{
  int    i,ierr;
  Scalar tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  for (i=1; i<=itg->curl; i++) {
    ierr = VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));CHKERRQ(ierr);
    tmp = -itg->alpha[i-1];
    ierr = VecAXPY(&tmp,itg->btilde[i-1],b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessFormX"
int KSPGuessFormX( KSP ksp, KSPIGUESS *itg, Vec x )
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = VecCopy(x,itg->xtilde[itg->curl]);CHKERRQ(ierr);
  for (i=1; i<=itg->curl; i++) {
    ierr = VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessUpdate"
int  KSPGuessUpdate( KSP ksp, Vec x, KSPIGUESS *itg )
{
  double       normax, norm;
  Scalar       tmp;
  MatStructure pflag;
  int          curl = itg->curl, i,ierr;
  Mat          Amat, Pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  if (curl == itg->maxl) {
    ierr = MatMult(Amat,x,itg->btilde[0] );CHKERRQ(ierr);
    ierr = VecNorm(itg->btilde[0],NORM_2,&normax);CHKERRQ(ierr);
    tmp = 1.0/normax; ierr = VecScale(&tmp,itg->btilde[0]);CHKERRQ(ierr);
    /* VCOPY(ksp->vc,x,itg->xtilde[0]); */
    ierr = VecScale(&tmp,itg->xtilde[0]);CHKERRQ(ierr);
  } else {
    ierr = MatMult( Amat, itg->xtilde[curl], itg->btilde[curl] );CHKERRQ(ierr);
    for (i=1; i<=curl; i++) {
      ierr = VecDot(itg->btilde[curl],itg->btilde[i-1],itg->alpha+i-1);CHKERRQ(ierr);
    }
    for (i=1; i<=curl; i++) {
      tmp  = -itg->alpha[i-1];
      ierr = VecAXPY(&tmp,itg->btilde[i-1],itg->btilde[curl]);CHKERRQ(ierr);
      ierr = VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],itg->xtilde[curl]);CHKERRQ(ierr);
    }
    ierr = VecNorm(itg->btilde[curl],NORM_2,&norm);CHKERRQ(ierr);
    tmp = 1.0/norm; ierr = VecScale(&tmp,itg->btilde[curl]);CHKERRQ(ierr);
    ierr = VecNorm(itg->xtilde[curl],NORM_2,&norm);CHKERRQ(ierr);
    ierr = VecScale(&tmp,itg->xtilde[curl]);CHKERRQ(ierr);
    itg->curl++;
  }
  PetscFunctionReturn(0);
}
