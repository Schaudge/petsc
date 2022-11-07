#include  <petsc/private/localnetrpimpl.h>
PetscFunctionList NetRPList              = NULL;
PetscClassId      NETRP_CLASSID          = 0; 
PetscBool         NetRPRegisterAllCalled = PETSC_FALSE;

/* Add Creation Routines here */
PETSC_EXTERN NetRPCreate_Blank(NetRP);
PETSC_EXTERN NetRPCreate_Linearized(NetRP);
/*@C
  NetRPRegisterAll - Registers all of the Network Riemann Problems. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: 
@*/
PetscErrorCode  NetRPRegisterAll(void)
{
  PetscFunctionBegin;
  if (NetRPRegisterAllCalled) PetscFunctionReturn(0);
  NetRPRegisterAllCalled = PETSC_TRUE;
  PetscCall(NetRPRegister(NETRPBLANK, NetRPCreate_Blank));
  PetscCall(NNetRPRegister(NETRPLINEARIZED, NetRPCreate_Linearized));
  PetscFunctionReturn(0);
}

/*@C
  NetRPSetType - Sets the method to be used as the Network Riemann problem. 

  Collective on NetRS

  Input Parameters:
+ nrs   - The Riemann Problem context
- type - A known method

  Options Database Command:
. -nrp_type <type> - Sets the method; use -help for a list of available methods (for instance, linearized)

   Notes:
   See "petsc/include/petscnetrp.h" for available methods (for instance)
+  
   Level: intermediate

.seealso: 

@*/
PetscErrorCode  NetRPSetType(NetRP rp, NetRPType type)
{
  PetscErrorCode (*r)(NetRP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID,1);
  PetscValidCharPointer(type,2);
  PetscCall(PetscObjectTypeCompare((PetscObject) rp, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(NetRPList,type,&r));
  if (!r) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Network Riemann Problem type: %s", type);
  PetscCall(NetRPReset(rp));
  if (rp->ops->destroy) {
    PetscCall((*(rp)->ops->destroy)(rp));
  }
  PetscCall(PetscMemzero(rp->ops,sizeof(*rp->ops)));
  rp->setupcalled = PETSC_FALSE;
  PetscCall(PetscObjectChangeTypeName((PetscObject)rp, type));
  PetscCall((*r)(rp));
  PetscFunctionReturn(0);
}
/*@C
  NetRPGetType - Gets the Network Riemann Problem method type (as a string).

  Not Collective

  Input Parameter:
. rp - The Riemann Problem

  Output Parameter:
. type - The name of Riemann Problem method

  Level: intermediate

.seealso NetRPSetType()
@*/
PetscErrorCode  NetRPGetType(NetRP rs, NetRPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)rs)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  NetRPRegister - Adds a creation method to the NetworkRiemannProblem package.

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  NetRPRegister() may be called multiple times to add several user-defined riemann problems. 

  Sample usage:
.vb
  NetRSRegister("my_riemannproblem",  MyRiemannProblemCreate);
.ve

  Then, your riemannproblem type can be chosen with the procedural interface via
.vb
    NetRS rs;
    NetRSCreate(MPI_Comm, &rs);
    NetRSSetType(rs, "my_riemannproblem")
.ve
  or at runtime via the option
.vb
    -nrs_type my_riemannproblem 
.ve

  Level: advanced

.seealso: NetRSRegisterAll(), NetRSRegisterDestroy()
@*/
PetscErrorCode  NetRPRegister(const char sname[], PetscErrorCode (*function)(NetRP))
{
  PetscFunctionBegin;
  PetscCall(NetRPInitializePackage());
  PetscCall(PetscFunctionListAdd(&NetRPList,sname,function));
  PetscFunctionReturn(0);
}
