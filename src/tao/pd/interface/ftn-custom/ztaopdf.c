#include <petsc/private/fortranimpl.h>
#include <petsc/private/taopdimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define taopdsetobjectiveroutine_            TAOPDSETOBJECTIVEROUTINE
  #define taopdsetgradientroutine_             TAOPDSETGRADIENTROUTINE
  #define taopdsetobjectiveandgradientroutine_ TAOPDSETOBJECTIVEANDGRADIENTROUTINE
  #define taopdview_                           TAOPDVIEW
  #define taopdsettype_                        TAOPDSETTYPE
  #define taopdviewfromoptions_                TAOPDVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

  #define taopdsetobjectiveroutine_            taopdsetobjectiveroutine
  #define taopdsetgradientroutine_             taopdsetgradientroutine
  #define taopdsetobjectiveandgradientroutine_ taopdsetobjectiveandgradientroutine
  #define taopdview_                           taopdview
  #define taopdsettype_                        taopdsettype
  #define taopdviewfromoptions_                taopdviewfromoptions
#endif

static int    OBJ     = 0;
static int    GRAD    = 1;
static int    OBJGRAD = 2;
static size_t NFUNCS  = 3;

static PetscErrorCode ourtaopdobjectiveroutine(TaoPD ls, Vec x, PetscReal *f, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoPD *, Vec *, PetscReal *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[OBJ]))(&ls, &x, f, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourtaopdgradientroutine(TaoPD ls, Vec x, Vec g, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoPD *, Vec *, Vec *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[GRAD]))(&ls, &x, &g, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourtaopdobjectiveandgradientroutine(TaoPD ls, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(TaoPD *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *))(((PetscObject)ls)->fortran_func_pointers[OBJGRAD]))(&ls, &x, f, &g, ctx, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void taopdsetobjectiveroutine_(TaoPD *ls, void (*func)(TaoPD *, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoPDSetObjective(*ls, 0, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[OBJ] = (PetscVoidFunction)func;
    *ierr                                          = TaoPDSetObjective(*ls, ourtaopdobjectiveroutine, ctx);
  }
}

PETSC_EXTERN void taopdsetgradientroutine_(TaoPD *ls, void (*func)(TaoPD *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoPDSetGradient(*ls, 0, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[GRAD] = (PetscVoidFunction)func;
    *ierr                                           = TaoPDSetGradient(*ls, ourtaopdgradientroutine, ctx);
  }
}

PETSC_EXTERN void taopdsetobjectiveandgradientroutine_(TaoPD *ls, void (*func)(TaoPD *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*ls, NFUNCS);
  if (!func) {
    *ierr = TaoPDSetObjectiveAndGradient(*ls, 0, ctx);
  } else {
    ((PetscObject)*ls)->fortran_func_pointers[OBJGRAD] = (PetscVoidFunction)func;
    *ierr                                              = TaoPDSetObjectiveAndGradient(*ls, ourtaopdobjectiveandgradientroutine, ctx);
  }
}

PETSC_EXTERN void taopdsettype_(TaoPD *ls, char *type_name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)

{
  char *t;

  FIXCHAR(type_name, len, t);
  *ierr = TaoPDSetType(*ls, t);
  if (*ierr) return;
  FREECHAR(type_name, t);
}

PETSC_EXTERN void taopdview_(TaoPD *ls, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = TaoPDView(*ls, v);
}

PETSC_EXTERN void taopdgetoptionsprefix_(TaoPD *ls, char *prefix, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *name;
  *ierr = TaoPDGetOptionsPrefix(*ls, &name);
  *ierr = PetscStrncpy(prefix, name, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, prefix, len);
}

PETSC_EXTERN void taopdappendoptionsprefix_(TaoPD *ls, char *prefix, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *name;
  FIXCHAR(prefix, len, name);
  *ierr = TaoPDAppendOptionsPrefix(*ls, name);
  if (*ierr) return;
  FREECHAR(prefix, name);
}

PETSC_EXTERN void taopdsetoptionsprefix_(TaoPD *ls, char *prefix, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(prefix, len, t);
  *ierr = TaoPDSetOptionsPrefix(*ls, t);
  if (*ierr) return;
  FREECHAR(prefix, t);
}

PETSC_EXTERN void taopdgettype_(TaoPD *ls, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;
  *ierr = TaoPDGetType(*ls, &tname);
  *ierr = PetscStrncpy(name, tname, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
PETSC_EXTERN void taopdviewfromoptions_(TaoPD *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = TaoPDViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
