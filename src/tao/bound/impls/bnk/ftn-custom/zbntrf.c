#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>


#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define taobntrsettrustregionhookroutine_  TAOBNTRSETTRUSTREGIONHOOKROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taobntrsettrustregionhookroutine_  taobntrsettrustregionhookroutine
#endif

static struct {
  PetscFortranCallbackId trhook;
} _cb;

static PetscErrorCode ourtrustregionhookroutine(Tao tao, PetscReal prered, PetscReal actred, void *ctx)
{
    PetscObjectUseFortranCallback(tao,_cb.trhook,(Tao*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&tao,&prered,&actred,_ctx,&ierr));
}

EXTERN_C_BEGIN

PETSC_EXTERN void taobntrsettrustregionhookroutine_(Tao *tao, void (*func)(Tao*,PetscReal,PetscReal,void*,PetscErrorCode*), void *ctx, PetscErrorCode *ierr)
{
    CHKFORTRANNULLFUNCTION(func);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*tao,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.trhook,(PetscVoidFunction)func,ctx);
    if(!*ierr) *ierr = TaoBNTRSetTrustRegionHookRoutine(*tao,ourtrustregionhookroutine,ctx);
}

EXTERN_C_END


