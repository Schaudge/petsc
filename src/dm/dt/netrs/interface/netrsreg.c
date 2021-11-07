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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (NetRSRegisterAllCalled) PetscFunctionReturn(0);
  NetRSRegisterAllCalled = PETSC_TRUE;
  ierr = NetRSRegister(NETRSLINEAR, NRSCreate_Linear);CHKERRQ(ierr);
  ierr = NetRSRegister(NETRSRIEMANN,NRSCreate_RS);CHKERRQ(ierr);
  ierr = NetRSRegister(NETRSOUTFLOW,NRSCreate_Outflow);CHKERRQ(ierr);
  ierr = NetRSRegister(NETRSEXACTSWE,NRSCreate_ExactSWE);CHKERRQ(ierr);
  ierr = NetRSRegister(NETRSEXACTSWESTAR,NRSCreate_ExactSWEStar);CHKERRQ(ierr);
  ierr = NetRSRegister(NETRSLINEARSTAR,NRSCreate_LinearStar);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject) rs, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(NetRSList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Network RiemannSolver type: %s", type);
  ierr = NetRSReset(rs);CHKERRQ(ierr);
  if (rs->ops->destroy) {
    ierr = (*(rs)->ops->destroy)(rs);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(rs->ops,sizeof(*rs->ops));CHKERRQ(ierr);
  rs->setupcalled = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)rs, type);CHKERRQ(ierr);
  ierr = (*r)(rs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NetRSInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&NetRSList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}