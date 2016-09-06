
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/

#undef __FUNCT__
#define __FUNCT__ "KSPCGSetType"
/*@
    KSPCGSetType - Sets the variant of the conjugate gradient method to
    use for solving a linear system with a complex coefficient matrix.
    This option is irrelevant when solving a real system.

    Logically Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   type - the variant of CG to use, one of
.vb
      KSP_CG_HERMITIAN - complex, Hermitian matrix (default)
      KSP_CG_SYMMETRIC - complex, symmetric matrix
.ve

    Level: intermediate

    Options Database Keys:
+   -ksp_cg_Hermitian - Indicates Hermitian matrix
-   -ksp_cg_symmetric - Indicates symmetric matrix

    Note:
    By default, the matrix is assumed to be complex, Hermitian.

.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type
@*/
PetscErrorCode  KSPCGSetType(KSP ksp,KSPCGType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscTryMethod(ksp,"KSPCGSetType_C",(KSP,KSPCGType),(ksp,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGUseSingleReduction"
/*@
    KSPCGUseSingleReduction - Merge the two inner products needed in CG into a single MPI_Allreduce() call.

    Logically Collective on KSP

    Input Parameters:
+   ksp - the iterative context
-   flg - turn on or off the single reduction

    Options Database:
.   -ksp_cg_single_reduction

    Level: intermediate

     The algorithm used in this case is described as Method 1 in Lapack Working Note 56, "Conjugate Gradient Algorithms with Reduced Synchronization Overhead
     Distributed Memory Multiprocessors", by E. F. D'Azevedo, V. L. Eijkhout, and C. H. Romine, December 3, 1999. V. Eijkhout creates the algorithm
     initially to Chronopoulos and Gear.

     It requires two extra work vectors than the conventional implementation in PETSc.

     See also KSPPIPECG, KSPPIPECR, and KSPGROPPCG that use non-blocking reductions.

.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type, KSPPGMRES
@*/
PetscErrorCode  KSPCGUseSingleReduction(KSP ksp,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ierr = PetscTryMethod(ksp,"KSPCGUseSingleReduction_C",(KSP,PetscBool),(ksp,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGSetRadius"
/*@
    KSPCGSetRadius - Sets the radius of the trust region.

    Logically Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   radius - the trust region radius (Infinity is the default)

    Level: advanced

.keywords: KSP, NASH, STCG, GLTR, set, trust region radius
@*/
PetscErrorCode  KSPCGSetRadius(KSP ksp, PetscReal radius)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (radius < 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE, "Radius negative");
  PetscValidLogicalCollectiveReal(ksp,radius,2);
  ierr = PetscTryMethod(ksp,"KSPCGSetRadius_C",(KSP,PetscReal),(ksp,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGGetNormD"
/*@
    KSPCGGetNormD - Got norm of the direction.

    Collective on KSP

    Input Parameters:
+   ksp    - the iterative context
-   norm_d - the norm of the direction

    Level: advanced

.keywords: KSP, NASH, STCG, GLTR, get, norm direction
@*/
PetscErrorCode  KSPCGGetNormD(KSP ksp, PetscReal *norm_d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPCGGetNormD_C",(KSP,PetscReal*),(ksp,norm_d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGGetObjFcn"
/*@
    KSPCGGetObjFcn - Get objective function value.

    Collective on KSP

    Input Parameters:
+   ksp   - the iterative context
-   o_fcn - the objective function value

    Level: advanced

.keywords: KSP, NASH, STCG, GLTR, get, objective function
@*/
PetscErrorCode  KSPCGGetObjFcn(KSP ksp, PetscReal *o_fcn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ierr = PetscUseMethod(ksp,"KSPCGGetObjFcn_C",(KSP,PetscReal*),(ksp,o_fcn));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

