#include  <petsc/private/netrsimpl.h>
PetscFunctionList NetRSList              = NULL;
PetscClassId      NETRS_CLASSID          = 0; 
PetscBool         NetRSRegisterAllCalled = PETSC_FALSE;

/* Add Creation Routines here */
PETSC_EXTERN PetscErrorCode NRSCreate_Linear(NetRS);
PETSC_EXTERN PetscErrorCode NRSCreate_RS(NetRS);
PETSC_EXTERN PetscErrorCode NRSCreate_Outflow(NetRS);
PETSC_EXTERN PetscErrorCode NRSCreate_ExactSWE(NetRS);
PETSC_EXTERN PetscErrorCode NRSCreate_ExactSWEStar(NetRS);
PETSC_EXTERN PetscErrorCode NRSCreate_LinearStar(NetRS);


/*@C
  NetRSRegisterAll - Registers all of the Network Riemann Solvers. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: 
@*/
PetscErrorCode  NetRSRegisterAll(void)
{
  PetscFunctionBegin;
  if (NetRSRegisterAllCalled) PetscFunctionReturn(0);
  NetRSRegisterAllCalled = PETSC_TRUE;
  PetscCall(NetRSRegister(NETRSLINEAR, NRSCreate_Linear));
  PetscCall(NetRSRegister(NETRSRIEMANN,NRSCreate_RS));
  PetscCall(NetRSRegister(NETRSOUTFLOW,NRSCreate_Outflow));
  PetscCall(NetRSRegister(NETRSEXACTSWE,NRSCreate_ExactSWE));
  PetscCall(NetRSRegister(NETRSEXACTSWESTAR,NRSCreate_ExactSWEStar));
  PetscCall(NetRSRegister(NETRSLINEARSTAR,NRSCreate_LinearStar));
  PetscFunctionReturn(0);
}

/*@C
  NetRSSetType - Sets the method to be used as the Network Riemann solver. 

  Collective on NetRS

  Input Parameters:
+ nrs   - The RiemannSolver context
- type - A known method

  Options Database Command:
. -nrs_type <type> - Sets the method; use -help for a list of available methods (for instance, lax)

   Notes:
   See "petsc/include/petscnetrs.h" for available methods (for instance)
+  
   Level: intermediate

.seealso: 

@*/
PetscErrorCode  NetRSSetType(NetRS rs, NetRSType type)
{
  PetscErrorCode (*r)(NetRS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID,1);
  PetscValidCharPointer(type,2);
  PetscCall(PetscObjectTypeCompare((PetscObject) rs, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(NetRSList,type,&r));
  if (!r) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Network RiemannSolver type: %s", type);
  PetscCall(NetRSReset(rs));
  if (rs->ops->destroy) {
    PetscCall((*(rs)->ops->destroy)(rs));
  }
  PetscCall(PetscMemzero(rs->ops,sizeof(*rs->ops)));
  rs->setupcalled = PETSC_FALSE;
  PetscCall(PetscObjectChangeTypeName((PetscObject)rs, type));
  PetscCall((*r)(rs));
  PetscFunctionReturn(0);
}
/*@C
  NetRSGetType - Gets the Network RiemannSolver method type (as a string).

  Not Collective

  Input Parameter:
. rs - The RiemannSolver

  Output Parameter:
. type - The name of RiemannSolver method

  Level: intermediate

.seealso NetRSSetType()
@*/
PetscErrorCode  NetRSGetType(NetRS rs, NetRSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)rs)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  NetRSRegister - Adds a creation method to the NetworkRiemannSolver package.

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  NetRSRegister() may be called multiple times to add several user-defined tses.

  Sample usage:
.vb
  NetRSRegister("my_riemannsolver",  MyRiemannSolverCreate);
.ve

  Then, your riemannsolver type can be chosen with the procedural interface via
.vb
    NetRS rs;
    NetRSCreate(MPI_Comm, &rs);
    NetRSSetType(rs, "my_riemannsolver")
.ve
  or at runtime via the option
.vb
    -nrs_type my_riemannsolver 
.ve

  Level: advanced

.seealso: NetRSRegisterAll(), NetRSRegisterDestroy()
@*/
PetscErrorCode  NetRSRegister(const char sname[], PetscErrorCode (*function)(NetRS))
{
  PetscFunctionBegin;
  PetscCall(NetRSInitializePackage());
  PetscCall(PetscFunctionListAdd(&NetRSList,sname,function));
  PetscFunctionReturn(0);
}
