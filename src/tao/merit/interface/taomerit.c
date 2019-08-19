#include <petsctaomerit.h> /*I "petsctaomerit.h" I*/
#include <petsc/private/taomeritimpl.h>
#include <petsc/private/taoimpl.h>

PetscFunctionList TaoMeritList = NULL;

PetscClassId TAOMERIT_CLASSID=0;

PetscLogEvent TAOMERIT_Eval;

/*@C
  TaoMeritView - Prints information about the TaoMerit object

  Collective on TaoMerit

  InputParameters:
+ merit - the TaoMerit context
- viewer - visualization context

  Options Database Key:
. -tao_merit_view - Calls TaoMeritView() at the end of the optimization

  Notes:
  The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

  Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/

PetscErrorCode TaoMeritView(TaoMerit merit, PetscViewer viewer)
{
  PetscErrorCode          ierr;
  PetscBool               isascii, isstring;
  TaoMeritType            type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)merit), &viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(merit,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)merit, viewer);CHKERRQ(ierr);
    if (merit->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*merit->ops->view)(merit, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    /* ierr = PetscViewerASCIIPrintf(viewer,"maximum function evaluations=%D\n",merit->max_funcs);CHKERRQ(ierr); */
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = TaoMeritGetType(merit,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-3.3s",type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritCreate - Creates a TAO merit function object.  Algorithms in TAO will automatically 
  create one.

  Collective

  Input Parameter:
. tao - Tao algorithm context

  Output Parameter:
. newmerit - the new TaoMerit context

  Available methods include:
+ objective
. lagrangian
. augmented lagrangian
- logarithmic barrier


   Options Database Keys:
.   -tao_merit_type - select which method TaoMerit should use

   Level: beginner

.seealso: TaoMeritSetType(), TaoMeritEvaluate(), TaoMeritDestroy()
@*/

PetscErrorCode TaoMeritCreate(Tao tao, TaoMerit *newmerit)
{
  PetscErrorCode ierr;
  TaoMerit  merit;

  PetscFunctionBegin;
  PetscValidPointer(newmerit,2);
  *newmerit = NULL;

  ierr = TaoMeritInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(merit,TAOMERIT_CLASSID,"TaoMerit","Merit","Tao",PetscObjectComm((PetscObject)tao),TaoMeritDestroy,TaoMeritView);CHKERRQ(ierr);

  merit->setupcalled = PETSC_FALSE;
  merit->resetcalled = PETSC_FALSE;
  merit->last_alpha = 0.0;
  merit->last_eval = 0.0;
  merit->ops->setup=0;
  merit->ops->eval=0;
  merit->ops->view=0;
  merit->ops->setfromoptions=0;
  merit->ops->destroy=0;

  *newmerit = merit;
  PetscFunctionReturn(0);
}

/*@
  TaoMeritSetUp - Sets up the internal data structures for the later use
  of a Tao solver

  Collective on TaoMerit

  Input Parameters:
. merit - the TaoMerit context

  Notes:
  The user will not need to explicitly call TaoMeritSetUp(), as it will
  automatically be called in TaoSolve().  However, if the user
  desires to call it explicitly, it should come after TaoCreate()
  but before TaoSolve().

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritEvaluate(), TaoMeritDestroy()
@*/

PetscErrorCode TaoMeritSetUp(TaoMerit merit)
{
  PetscErrorCode ierr;
  const char     *default_type=TAOMERITOBJECTIVE;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  if (!merit->setupcalled) {
    ierr = VecDuplicate(merit->tao->solution, &merit->Xinit);CHKERRQ(ierr);
    ierr = VecDuplicate(merit->tao->solution, &merit->Xtrial);CHKERRQ(ierr);
    ierr = VecDuplicate(merit->tao->stepdirection, &merit->step);CHKERRQ(ierr);
    merit->setupcalled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@
  TaoMeritReset - Reset the merit function with a new optimization point 
  and a new search direction. Any future evaluations of the merit function 
  will happen using this point and step information.

  Collective on TaoMerit

  Input Parameter:
+ merit - TaoMerit context
. x0 - new initial point
- p - new search direction

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritEvaluate()
@*/
PetscErrorCode TaoMeritReset(TaoMerit merit, Vec x0, Vec p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,2);
  PetscValidHeaderSpecific(p,VEC_CLASSID,3);
  if (!merit->setupcalled) {
    ierr = TaoMeritSetUp(merit);CHKERRQ(ierr);
  }
  ierr = VecCopy(x0, merit->Xinit);CHKERRQ(ierr);
  ierr = VecCopy(x0, merit->Xtrial);CHKERRQ(ierr);
  ierr = VecCopy(p, merit->step);CHKERRQ(ierr);
  merit->last_alpha = 0.0;
  merit->last_eval = 0.0;
  if (merit->ops->reset) {
    ierr = (*merit->ops->reset)(merit,x0,p);CHKERRQ(ierr);
  }
  merit->resetcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TaoMeritEvaluate - Evaluate the merit function with a new step length

  Collective on TaoMerit

  Input Parameter:
+ merit - TaoMerit context
- alpha - step length

  Output Parameter:
. fval - merit function value evaluated at alpha

  Level: beginner

.seealso: TaoMeritCreate(), TaoMeritReset()
@*/
PetscErrorCode TaoMeritEvaluate(TaoMerit merit, PetscReal alpha, PetscReal *fval)
{
  PetscErrorCode ierr;

  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscLogEventBegin(TAOMERIT_Eval,merit,0,0,0);
  if (!merit->resetcalled) SETERRQ(PetscComm((PetscObject)merit), PETSC_ERR_ORDER, "Must call TaoMeritReset() with new point and step direction before evaluating the merit function");
  if (merit->ops->eval) {
    ierr = (*(merit->ops->eval))(merit, alpha, fval);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ARG_WRONGSTATE,"Evaluation function not defined for TaoMerit type");
  }
  PetscLogEventEnd(TAOMERIT_Eval,merit,0,0,0);
  PetscFunctionReturn(0);
}

/*@
  TaoMeritDestroy - Destroys the TaoMerit context that was created with
  TaoMeritCreate()

  Collective on TaoMerit

  Input Parameter
. merit - the TaoMerit context

  Level: beginner

.seealso: TaoMeritCreate(), TaoMeritEvaluate()
@*/
PetscErrorCode TaoMeritDestroy(TaoMerit *merit)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*merit) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*merit,TAOMERIT_CLASSID,1);
  if (--((PetscObject)*merit)->refct > 0) {*merit=0; PetscFunctionReturn(0);}
  ierr = VecDestroy(&(*merit)->Xinit);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->Xtrial);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->step);CHKERRQ(ierr);
  if ((*merit)->ops->destroy) {
    ierr = (*(*merit)->ops->destroy)(*merit);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(merit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetType - Sets the merit function type

  Collective on TaoMerit

  Input Parameters:
+ merit - the TaoMerit context
- type - merit function type

  Available methods include:
+ objective
. lagrangian
. aug-lag - augmented lagrangian
- log-barrier - objective plus logarithmic barrier term

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritGetType(), TaoMeritEvaluate()

@*/

PetscErrorCode TaoMeritSetType(TaoMerit merit, TaoMeritType type)
{
  PetscErrorCode ierr;
  PetscErrorCode (*r)(Tao, TaoMerit*);
  PetscBool      flg;
  Tao            tao = merit->tao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject)merit, type, &flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(TaoMeritList,type, (void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested TaoMerit type %s",type);
  if (merit->ops->destroy) {
    ierr = (*(merit)->ops->destroy)(&merit);CHKERRQ(ierr);
  }
  merit->ops->setup=0;
  merit->ops->eval=0;
  merit->ops->view=0;
  merit->ops->setfromoptions=0;
  merit->ops->destroy=0;
  merit->setupcalled = PETSC_FALSE;
  merit->resetcalled = PETSC_FALSE;
  ierr = (*r)(tao, merit);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)merit, type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoLMeritSetFromOptions - Sets various TaoMerit parameters from user
  options.

  Collective on TaoMerit

  Input Paremeter:
. merit - the TaoMerit context

  Options Database Keys:
+ -tao_merit_type <type> - The merit function type (objective, lagrangian, aug-lag, logbarrier)
- -tao_merit_view - display merit function information

  Level: beginner
@*/
PetscErrorCode TaoMeritSetFromOptions(TaoMerit merit)
{
  PetscErrorCode ierr;
  const char     *default_type=TAOMERITOBJECTIVE;
  char           type[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)merit);CHKERRQ(ierr);
  if (((PetscObject)merit)->type_name) {
    default_type = ((PetscObject)merit)->type_name;
  }
  /* Check for type from options */
  ierr = PetscOptionsFList("-tao_merit_type","Tao merit function type","TaoMeritSetType",TaoMeritList,default_type,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TaoMeritSetType(merit,type);CHKERRQ(ierr);
  } else if (!((PetscObject)merit)->type_name) {
    ierr = TaoMeritSetType(merit,default_type);CHKERRQ(ierr);
  }
  if (merit->ops->setfromoptions) {
    ierr = (*merit->ops->setfromoptions)(PetscOptionsObject,merit);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritGetType - Gets the type for the merit function

  Not Collective

  Input Parameter:
. merit - the TaoMerit context

  Output Paramter:
. type - the merit function type in effect

  Level: developer

@*/
PetscErrorCode TaoMeritGetType(TaoMerit merit, TaoMeritType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)merit)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   TaoMeritRegister - Adds a new merit function type to the registry

   Not collective

   Input Parameters:
+  sname - name of a new user-defined merit function
-  func - routine to Create method context

   Notes:
   TaoMeritRegister() may be called multiple times to add several user-defined merit functions.

   Sample usage:
.vb
   TaoMeritRegister("my_merit",MyMeritCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TaoMeritSetType(ls,"my_merit")
   or at runtime via the option
$     -tao_merit_type my_merit

   Level: developer

@*/
PetscErrorCode TaoMeritRegister(const char sname[], PetscErrorCode (*func)(Tao, TaoMerit*))
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TaoMeritInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&TaoMeritList, sname, (void (*)(void))func);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}