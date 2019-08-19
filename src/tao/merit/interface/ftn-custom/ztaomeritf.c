#include <petsc/private/fortranimpl.h>
#include <petsc/private/taomeritimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)

#define taomeritview_                           TAOMERITVIEW
#define taomeritsettype_                        TAOMERITSETTYPE
#define taomeritgettype_                        TAOMERITGETTYPE

#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)

#define taomeritview_                           taomeritview
#define taomeritsettype_                        taomeritsettype
#define taomeritgettype_                        taomeritgettype

#endif

PETSC_EXTERN void PETSC_STDCALL taomeritsettype_(TaoMerit *merit, char* type_name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))

{
    char *t;

    FIXCHAR(type_name,len,t);
    *ierr = TaoMeritSetType(*merit,t);if (*ierr) return;
    FREECHAR(type_name,t);

}

PETSC_EXTERN void PETSC_STDCALL taomeritview_(TaoMerit *merit, PetscViewer *viewer, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(viewer,v);
    *ierr = TaoMeritView(*merit,v);
}

PETSC_EXTERN void PETSC_STDCALL taomeritgettype_(TaoMerit *merit, char* name PETSC_MIXED_LEN(len), PetscErrorCode *ierr  PETSC_END_LEN(len))
{
  const char *tname;
  *ierr = TaoMeritGetType(*merit,&tname);
  *ierr = PetscStrncpy(name,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}
