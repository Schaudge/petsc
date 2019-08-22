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

.seealso: TaoMeritSetType(), TaoMeritGetValue(), TaoMeritDestroy()
@*/

PetscErrorCode TaoMeritCreate(MPI_Comm comm, TaoMerit *newmerit)
{
  PetscErrorCode ierr;
  TaoMerit  merit;

  PetscFunctionBegin;
  PetscValidPointer(newmerit,2);
  *newmerit = NULL;

  ierr = TaoMeritInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(merit,TAOMERIT_CLASSID,"TaoMerit","Merit","Tao",comm,TaoMeritDestroy,TaoMeritView);CHKERRQ(ierr);

  merit->setupcalled = PETSC_FALSE;
  merit->resetcalled = PETSC_FALSE;
  merit->use_tao = PETSC_FALSE;

  merit->last_alpha = -1.0;
  merit->last_value = 0.0;
  merit->last_dirderiv = 0.0;

  merit->ops->setup=0;
  merit->ops->getvalue=0;
  merit->ops->getdirderiv=0;
  merit->ops->getvalueanddirderiv=0;
  merit->ops->view=0;
  merit->ops->setfromoptions=0;
  merit->ops->destroy=0;

  merit->ops->userobjective=0;
  merit->ops->usergradient=0;
  merit->ops->userobjandgrad=0;
  merit->ops->userhessian=0;
  merit->ops->usercnstreq=0;
  merit->ops->usercnstrineq=0;
  merit->ops->userjaceq=0;
  merit->ops->userjacineq=0;

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

.seealso: TaoMeritCreate(), TaoMeritGetValue(), TaoMeritDestroy()
@*/

PetscErrorCode TaoMeritSetUp(TaoMerit merit)
{
  PetscErrorCode ierr;
  const char     *default_type=TAOMERITOBJECTIVE;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  if (!merit->setupcalled) {
    ierr = VecDuplicate(merit->tao->solution, &merit->Xtrial);CHKERRQ(ierr);
    ierr = VecDuplicate(merit->tao->gradient, &merit->Gtrial);CHKERRQ(ierr);
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

.seealso: TaoMeritCreate(), TaoMeritGetValue()
@*/
PetscErrorCode TaoMeritReset(TaoMerit merit, Vec x0, Vec p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  if (!merit->setupcalled) {
    ierr = TaoMeritSetUp(merit);CHKERRQ(ierr);
  }
  merit->Xinit = x0;
  merit->step = p;
  merit->last_alpha = -1.0;
  merit->last_value = 0.0;
  merit->last_dirderiv = 0.0;
  if (merit->ops->reset) {
    ierr = (*merit->ops->reset)(merit, x0, p);CHKERRQ(ierr);
  }
  merit->resetcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  TaoMeritGetValue - Evaluate the merit function value at a new step length.

  Collective on TaoMerit

  Input Parameter:
+ merit - TaoMerit context
- alpha - step length

  Output Parameter:
. fval - merit function value evaluated at alpha

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritReset()
@*/
PetscErrorCode TaoMeritGetValue(TaoMerit merit, PetscReal alpha, PetscReal *fval)
{
  PetscErrorCode ierr;

  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscLogEventBegin(TAOMERIT_GetValue,merit,0,0,0);
  if (!merit->resetcalled) SETERRQ(PetscComm((PetscObject)merit), PETSC_ERR_ORDER, "Must call TaoMeritReset() with new point and step direction before evaluating the merit function");
  if (alpha == merit->last_alpha) {
    /* Avoid unnecessary computation if the step length is same as last evaluated step length */
    *fval = merit->last_value;
    PetscFunctionReturn(0);
  }
  if (merit->ops->getvalue) {
    ierr = (*(merit->ops->getvalue))(merit, alpha, &merit->last_value);CHKERRQ(ierr);
  } else if (merit->ops->getvalueanddirderiv) {
    ierr = (*(merit->ops->getvalueanddirderiv))(merit, alpha, &merit->last_value, NULL);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ARG_WRONGSTATE,"Merit function value not defined for TaoMerit type");
  }
  merit->last_alpha = alpha;
  *fval = merit->last_value;
  PetscLogEventEnd(TAOMERIT_GetValue,merit,0,0,0);
  PetscFunctionReturn(0);
}

/*@
  TaoMeritGetDirDeriv - Evaluate the merit function directional derivative at a new step length.
  The directional derivative is the dot product between the step direction and the gradient of the 
  merit function.

  Collective on TaoMerit

  Input Parameter:
+ merit - TaoMerit context
- alpha - step length

  Output Parameter:
. gtp - directional derivative evaluated at alpha

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritReset()
@*/
PetscErrorCode TaoMeritGetDirDeriv(TaoMerit merit, PetscReal alpha, PetscReal *gtp)
{
  PetscErrorCode ierr;

  PetscErrorCode ierr;

  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscLogEventBegin(TAOMERIT_GetDirDeriv,merit,0,0,0);
  if (!merit->resetcalled) SETERRQ(PetscComm((PetscObject)merit), PETSC_ERR_ORDER, "Must call TaoMeritReset() with new point and step direction before evaluating the merit function");
  if (alpha == merit->last_alpha) {
    /* Avoid unnecessary computation if the step length is same as last evaluated step length */
    *gtp = merit->last_dirderiv;
    PetscFunctionReturn(0);
  }
  if (merit->ops->getdirderiv) {
    ierr = (*(merit->ops->getvalue))(merit, alpha, &merit->last_dirderiv);CHKERRQ(ierr);
  } else if (merit->ops->getvalueanddirderiv) {
    ierr = (*(merit->ops->getvalueanddirderiv))(merit, alpha, NULL, &merit->last_dirderiv);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ARG_WRONGSTATE,"Directional derivative evaluation not defined for TaoMerit type");
  }
  merit->last_alpha = alpha;
  *gtp = merit->last_dirderiv;
  PetscLogEventEnd(TAOMERIT_GetDirDeriv,merit,0,0,0);
  PetscFunctionReturn(0);
}

/*@
  TaoMeritGetValueAndDirDeriv - Evaluate both the merit function value and its directional 
  derivative at a new step length. The directional derivative is the dot product between the 
  step direction and the gradient of the merit function.

  Collective on TaoMerit

  Input Parameter:
+ merit - TaoMerit context
- alpha - step length

  Output Parameter:
+ fval - merit function value evaluated at alpha
- gtp - directional derivative value at alpha

  Level: developer

.seealso: TaoMeritCreate(), TaoMeritReset()
@*/
PetscErrorCode TaoMeritGetValueAndDirDeriv(TaoMerit merit, PetscReal alpha, PetscReal *fval, PetscReal *gtp)
{
  PetscErrorCode ierr;

  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscLogEventBegin(TAOMERIT_GetValueAndDirDeriv,merit,0,0,0);
  if (!merit->resetcalled) SETERRQ(PetscComm((PetscObject)merit), PETSC_ERR_ORDER, "Must call TaoMeritReset() with new point and step direction before evaluating the merit function");
  if (alpha == merit->last_alpha) {
    /* Avoid unnecessary computation if the step length is same as last evaluated step length */
    *fval = merit->last_value;
    *gtp = merit->last_dirderiv;
    PetscFunctionReturn(0);
  }
  if (merit->ops->getvalueanddirderiv) {
    ierr = (*(merit->ops->getvalueanddirderiv))(merit, alpha, &merit->last_value, &merit->last_dirderiv);CHKERRQ(ierr);
  } else if ((merit->ops->getvalue) && (merit->ops->getdirderiv)) {
    ierr = (*(merit->ops->getvalue))(merit, alpha, &merit->last_value);CHKERRQ(ierr);
    ierr = (*(merit->ops->getdirderiv))(merit, alpha, &merit->last_dirderiv);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscComm((PetscObject)merit),PETSC_ERR_ARG_WRONGSTATE,"Function value and directional derivative evaluations are not defined for TaoMerit type");
  }
  merit->last_alpha = alpha;
  *fval = merit->last_value;
  *gtp = merit->last_dirderiv;
  PetscLogEventEnd(TAOMERIT_GetValueAndDirDeriv,merit,0,0,0);
  PetscFunctionReturn(0);
}

/*@
  TaoMeritDestroy - Destroys the TaoMerit context that was created with
  TaoMeritCreate()

  Collective on TaoMerit

  Input Parameter
. merit - the TaoMerit context

  Level: beginner

.seealso: TaoMeritCreate(), TaoMeritGetValue()
@*/
PetscErrorCode TaoMeritDestroy(TaoMerit *merit)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*merit) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*merit,TAOMERIT_CLASSID,1);
  if (--((PetscObject)*merit)->refct > 0) {
    *merit = 0;
    PetscFunctionReturn(0);
  }

  ierr = MatDestroy(&(*merit)->H);CHKERRQ(ierr);
  ierr = MatDestroy(&(*merit)->Hpre);CHKERRQ(ierr);
  ierr = MatDestroy(&(*merit)->Jeq);CHKERRQ(ierr);
  ierr = MatDestroy(&(*merit)->Jeq_pre);CHKERRQ(ierr);
  ierr = MatDestroy(&(*merit)->Jineq);CHKERRQ(ierr);
  ierr = MatDestroy(&(*merit)->Jineq_pre);CHKERRQ(ierr);

  ierr = VecDestroy(&(*merit)->Xinit);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->Ceq);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->Cineq);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->step);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->Xtrial);CHKERRQ(ierr);
  ierr = VecDestroy(&(*merit)->Gtrial);CHKERRQ(ierr);

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

  Level: beginner

.seealso: TaoMeritCreate(), TaoMeritGetType(), TaoMeritGetValue()

@*/

PetscErrorCode TaoMeritSetType(TaoMerit merit, TaoMeritType type)
{
  PetscErrorCode ierr;
  PetscErrorCode (*r)(TaoMerit);
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject)merit, type, &flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(TaoMeritList,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested TaoMerit type %s",type);
  if (merit->ops->destroy) {
    ierr = (*(merit->ops->destroy))(&merit);CHKERRQ(ierr);
  }
  merit->ops->setup=0;
  merit->ops->getvalue=0;
  merit->ops->getdirderiv=0;
  merit->ops->getvalueanddirderiv=0;
  merit->ops->view=0;
  merit->ops->setfromoptions=0;
  merit->ops->destroy=0;
  merit->setupcalled = PETSC_FALSE;
  merit->resetcalled = PETSC_FALSE;
  ierr = (*r)(merit);CHKERRQ(ierr);
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
+ sname - name of a new user-defined merit function
- func - routine to Create method context

  Notes:
  TaoMeritRegister() may be called multiple times to add several user-defined merit functions.

  Sample usage:
.vb
  TaoMeritRegister("my_merit",MyMeritCreate);
.ve

  Then, your solver can be chosen with the procedural interface via
$    TaoMeritSetType(ls,"my_merit")
  or at runtime via the option
$    -tao_merit_type my_merit

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

/*@C
  TaoMeritUseTaoCallbacks - Make the TaoMerit object use the same user defined 
  callbacks set into the Tao object

  Input Parameters:
+ merit - the TaoMerit context
- tao - the Tao context

  Level: intermediate

@*/
PetscErrorCode TaoMeritUseTaoCallbacks(TaoMerit merit, Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(tao,TAO_CLASSID,2);
  merit->tao = tao;
  merit->use_tao = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritIsUsingTaoCallbacks - Check if the TaoMerit object is using Tao callbacks

  Input Parameters:
+ merit - the TaoMerit context
- flg - PETSC_TRUE if tao callbacks are active

  Level: intermediate

@*/
PetscErrorCode TaoMeritUseTaoCallbacks(TaoMerit merit, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  *flg = merit->use_tao;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetObjectiveRoutine - Set the user callback for evaluating the objective function

  Input Parameters:
+ merit - the TaoMerit context
. X - Vec object for optimization variables
. func - function pointer for objective function evaluations
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetObjectiveRoutine(TaoMerit merit, Vec X, PetscErrorCode(*func)(TaoMerit, Vec, PetscReal*, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  if (merit->Xinit) {
    ierr = VecDestroy(&merit->Xinit);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  merit->Xinit = X;
  merit->ops->userobjective = func;
  merit->user_obj = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetGradientRoutine - Set the user callback for evaluating the gradient of the objective function

  Input Parameters:
+ merit - the TaoMerit context
. X - Vec object for optimization variables
. func - function pointer for gradient evaluations
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetGradientRoutine(TaoMerit merit, Vec X, PetscErrorCode(*func)(TaoMerit, Vec, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  if (merit->Xinit) {
    ierr = VecDestroy(&merit->Xinit);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  merit->Xinit = X;
  merit->ops->usergradient = func;
  merit->user_grad = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetObjectiveAndGradientRoutine - Set the user callback for evaluating the objective function 
  and its gradient at the same time

  Input Parameters:
+ merit - the TaoMerit context
. X - Vec object for optimization variables
. func - function pointer for objective function and gradient evaluations
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetObjectiveAndGradientRoutine(TaoMerit merit, Vec X, PetscErrorCode(*func)(TaoMerit, Vec, PetscReal*, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  if (merit->Xinit) {
    ierr = VecDestroy(&merit->Xinit);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  merit->Xinit = X;
  merit->ops->userobjandgrad = func;
  merit->user_objgrad = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetHessianRoutine - Set the user callback for evaluating the Hessian of the objective function

  Input Parameters:
+ merit - the TaoMerit context
. H - Mat object for the Hessian
. Hpre - Mat object for the Hessian preconditioner/pseudo-inverse
. func - function pointer for Hessian evaluations
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetHessianRoutine(TaoMerit merit, Mat H, Mat Hpre, PetscErrorCode(*func)(TaoMerit, Vec, Mat, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(H,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Hpre,MAT_CLASSID,3);
  ierr = MatDestroy(&merit->H);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)H);CHKERRQ(ierr);
  merit->H = H;
  ierr = MatDestroy(&merit->Hpre);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Hpre);CHKERRQ(ierr);
  merit->Hpre = Hpre;
  merit->ops->userhessian = func;
  merit->user_hess = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetEqualityConstraints - Set the user callback for evaluating the equality constraints

  Input Parameters:
+ merit - the TaoMerit context
. Ceq - Vec object for equality constraints
. func - function pointer for equality constraint evaluations
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetEqualityConstraints(TaoMerit merit, Vec Ceq, PetscErrorCode(*func)(TaoMerit, Vec, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(Ceq,VEC_CLASSID,2);
  if (merit->Ceq) {
    ierr = VecDestroy(&merit->Ceq);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)Ceq);CHKERRQ(ierr);
  merit->Ceq = Ceq;
  merit->ops->usercnstreq = func;
  merit->user_cnstreq = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetInequalityConstraints - Set the user callback for evaluating the inequality constraints

  Input Parameters:
+ merit - the TaoMerit context
. Cineq - Vec object for inequality constraints
. func - function pointer for inequality constraint evaluation
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetInequalityConstraints(TaoMerit merit, Vec Cineq, PetscErrorCode(*func)(TaoMerit, Vec, Vec, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(Cineq,VEC_CLASSID,2);
  if (merit->Cineq) {
    ierr = VecDestroy(&merit->Cineq);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)Cineq);CHKERRQ(ierr);
  merit->Cineq = Cineq;
  merit->ops->usercnstrineq = func;
  merit->user_cnstrineq = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetEqualityJacobian - Set the user callback for evaluating the Jacobian of the equality constraints

  Input Parameters:
+ merit - the TaoMerit context
. Jeq - Mat object for equality constraint Jacobian
. func - function pointer for equality constraint Jacobian evaluation
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetEqualityJacobian(TaoMerit merit, Mat Jeq, PetscErrorCode(*func)(TaoMerit, Vec, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(Jeq,MAT_CLASSID,2);
  ierr = MatDestroy(&merit->Jeq);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Jeq);CHKERRQ(ierr);
  merit->Jeq = Jeq;
  merit->ops->userjaceq = func;
  merit->user_jaceq = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritSetInequalityJacobian - Set the user callback for evaluating the Jacobian of the inequality constraints

  Input Parameters:
+ merit - the TaoMerit context
. Jeq - Mat object for inequality constraint Jacobian
. func - function pointer for inequality constraint Jacobian evaluation
- ctx - user application context

  Level: intermediate

@*/
PetscErrorCode TaoMeritSetInequalityJacobian(TaoMerit merit, Mat Jineq, PetscErrorCode(*func)(TaoMerit, Vec, Mat, void*), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(merit,TAOMERIT_CLASSID,1);
  PetscValidHeaderSpecific(Jineq,MAT_CLASSID,2);
  ierr = MatDestroy(&merit->Jineq);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Jineq);CHKERRQ(ierr);
  merit->Jineq = Jineq;
  merit->ops->userjacineq = func;
  merit->user_jacineq = ctx;
  PetscFunctionReturn(0);
}