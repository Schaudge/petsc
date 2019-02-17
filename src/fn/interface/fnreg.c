/*
     Mechanism for register PetscFn  types
*/
#include <petsc/private/fnimpl.h>      /*I "petscfn.h" I*/

PetscBool PetscFnRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered PetscFn routines
*/
PetscFunctionList PetscFnList = 0;

/*@C
   PetscFnSetType - Builds PetscFn object for a particular nonlinear function type

   Collective on PetscFn

   Input Parameters:
+  fn      - the function object
-  fntype   - function type

   Options Database Key:
.  -petscfn_type  <method> - Sets the type; use -help for a list
    of available methods (for instance, seqaij)

   Notes:
   See "${PETSC_DIR}/include/petscfn.h" for available methods

  Level: intermediate

.keywords: PetscFn, PetscFnType, set, method

.seealso: PetscFnCreate(), PetscFnType, PetscFn
@*/
PetscErrorCode  PetscFnSetType(PetscFn fn, PetscFnType fntype)
{
  PetscErrorCode ierr,(*r)(PetscFn);
  PetscBool      sametype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);

  ierr = PetscObjectTypeCompare((PetscObject)fn,fntype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(PetscFnList,fntype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscFn type given: %s",fntype);

  if (fn->ops->destroy) {
    /* free the old data structure if it existed */
    ierr = (*fn->ops->destroy)(fn);CHKERRQ(ierr);
    fn->ops->destroy = NULL;
  }
  ierr = PetscObjectStateIncrease((PetscObject)fn);CHKERRQ(ierr);

  /* create the new data structure */
  ierr = (*r)(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscFnGetType - Gets the PetscFn type as a string from the object.

   Not Collective

   Input Parameter:
.  fn - the PetscFn object

   Output Parameter:
.  name - name of PetscFn type

   Level: intermediate

.keywords: PetscFn, PetscFnType, get, method, name

.seealso: PetscFnSetType()
@*/
PetscErrorCode  PetscFnGetType(PetscFn fn,PetscFnType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)fn)->type_name;
  PetscFunctionReturn(0);
}


/*@C
  PetscFnRegister -  - Adds a new PetscFn type

   Not Collective

   Input Parameters:
+  name - name of a new user-defined PetscFn type
-  routine_create - routine to create method context

   Notes:
   PetscFnRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   PetscFnRegister("my_fn",MyFnCreate);
.ve

   Then, your nonlinear function can be chosen with the procedural interface via
$     PetscFnSetType(PetscFn,"my_fn")
   or at runtime via the option
$     -petscfn_type my_mat

   Level: advanced

.keywords: PetscFn, register

.seealso: PetscFnRegisterAll()


  Level: advanced
@*/
PetscErrorCode  PetscFnRegister(const char sname[],PetscErrorCode (*function)(PetscFn))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscFnList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFnCreate_Shell(PetscFn);

PetscErrorCode PetscFnRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFnRegister(PETSCFNSHELL, PetscFnCreate_Shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
