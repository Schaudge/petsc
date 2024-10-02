#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscBool         TaoTermRegisterAllCalled = PETSC_FALSE;
PetscFunctionList TaoTermList              = NULL;

PETSC_INTERN PetscErrorCode TaoTermCreate_TaoCallbacks(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_BRGNRegularizer(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_ADMMRegularizer(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_ADMMMisfit(TaoTerm);

/*@C
  TaoTermRegister - Register an impementation of `TaoTerm`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname - name of a new user-defined solver
- func  - routine to Create method context

  Example Usage:
.vb
   TaoTermRegister("my_term", MyTermCreate);
.ve

  Then, your term can be chosen with the procedural interface via
$     TaoTermType(term, "my_term")
  or at runtime via the option
$     -taoterm_type my_term

  Level: advanced

  Note:
  `TaoTermRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermRegister(const char sname[], PetscErrorCode (*func)(TaoTerm))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoTermList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermRegisterAll(void)
{
  PetscFunctionBegin;
  if (TaoTermRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TaoTermRegisterAllCalled = PETSC_TRUE;
  PetscCall(TaoTermRegister(TAOTERMTAOCALLBACKS, TaoTermCreate_TaoCallbacks));
  PetscCall(TaoTermRegister(TAOTERMBRGNREGULARIZER, TaoTermCreate_BRGNRegularizer));
  PetscCall(TaoTermRegister(TAOTERMADMMREGULARIZER, TaoTermCreate_ADMMRegularizer));
  PetscCall(TaoTermRegister(TAOTERMADMMMISFIT, TaoTermCreate_ADMMMisfit));
  PetscFunctionReturn(PETSC_SUCCESS);
}
