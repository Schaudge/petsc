static char help[] =
 "Demonstrates how data from DMPlex may be mapped to perform computations on GPUs.\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include <petscdmadaptor.h>
/* We are solving the system of equations:
 * \vec{u} = -\grad{p}
 * \div{u} = f
 */

/* Exact solutions for linear velocity
   \vec{u} = \vec{x};
   p = -0.5*(\vec{x} \cdot \vec{x});
   */
static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt c;

  for (c = 0; c < Nc; ++c) u[c] = x[c];
  return 0;
}

static PetscErrorCode linear_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt d;

  u[0] = 0.;
  for (d=0; d<dim; ++d) u[0] += -0.5*x[d]*x[d];
  return 0;
}

static PetscErrorCode linear_divu(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  u[0] = dim;
  return 0;
}

/* fx_v are the residual functions for the equation \vec{u} = \grad{p}. f0_v is the term <v,u>.*/
static void f0_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt i;

  for (i=0; i<dim; ++i) f0[i] = u[uOff[0] + i];
}

/* f1_v is the term <v,-\grad{p}> but we integrate by parts to get <\grad{v}, -p*I> */
static void f1_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) {
    PetscInt d;

    for (d=0; d<dim; ++d) f1[c*dim + d] = (c==d) ? -u[uOff[1]] : 0;
  }
}

/* Residual function for enforcing \div{u} = f. */
static void f0_q_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar rhs,divu=0;
  PetscInt    i;

  (void)linear_divu(dim,t,x,dim,&rhs,NULL);;
  for (i=0; i< dim; ++i) divu += u_x[uOff_x[0]+i*dim+i];
  f0[0] = divu-rhs;
}

/* Boundary residual. Dirichlet boundary for u means u_bdy=p*n */
static void f0_bd_u_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;
  PetscInt    d;

  (void)linear_p(dim,t,x,dim,&pressure,NULL);
  for (d=0; d<dim; ++d) f0[d] = pressure*n[d];
}

/* gx_yz are the jacobian functions obtained by taking the derivative of the y residual w.r.t z*/
static void g0_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g0[c*dim + c] = 1.0;
}

static void g1_qu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g1[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g1[c*dim + c] = 1.0;
}

static void g2_vp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g2[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g2[c*dim + c] = -1.0;
}

typedef struct
{
  PetscBool simplex;
  PetscInt  dim;
} UserCtx;

/* Process command line options and initialize the UserCtx struct */
static PetscErrorCode ProcessOptions(MPI_Comm comm,UserCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Default to  2D, triangle mesh.*/
  user->simplex = PETSC_TRUE;
  user->dim     = 2;

  ierr = PetscOptionsBegin(comm,"","DMPlex GPU Tutorial","PetscSpace");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex","Whether to use simplices (true) or tensor-product (false) cells in " "the mesh","ex10.c",user->simplex,
                          &user->simplex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Number of solution dimensions","ex10.c",user->dim,&user->dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm,UserCtx *user,DM *mesh)
{
  PetscErrorCode   ierr;
  DMLabel          label;
  const char       *name  = "marker";
  DM               dmDist = NULL;
  PetscPartitioner part;

  PetscFunctionBegin;
  /* Create box mesh from user parameters */
  ierr = DMPlexCreateBoxMesh(comm,user->dim,user->simplex,NULL,NULL,NULL,NULL,PETSC_TRUE,mesh);CHKERRQ(ierr);

  /* Make sure the mesh gets properly distributed if running in parallel */
  ierr = DMPlexGetPartitioner(*mesh,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh,0,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr  = DMDestroy(mesh);CHKERRQ(ierr);
    *mesh = dmDist;
  }

  /* Mark the boundaries, we will need this later when setting up the system of
   * equations */
  ierr = DMCreateLabel(*mesh,name);CHKERRQ(ierr);
  ierr = DMGetLabel(*mesh,name,&label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(*mesh,1,label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(*mesh,label);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *mesh,"Mesh");CHKERRQ(ierr);

  /* Get any other mesh options from the command line */
  ierr = DMSetApplicationContext(*mesh,user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*mesh);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh,NULL,"-dm_view");CHKERRQ(ierr);

  ierr = DMDestroy(&dmDist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup the system of equations that we wish to solve */
static PetscErrorCode SetupProblem(DM dm,UserCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;
  const PetscInt id=1;

  PetscFunctionBegin;
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  /* All of these are independent of the user's choice of solution */
  ierr = PetscDSSetResidual(prob,0,f0_v,f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob,1,f0_q_linear,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,g0_vu,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,1,NULL,NULL,g2_vp,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,1,0,NULL,g1_qu,NULL,NULL);CHKERRQ(ierr);

  ierr = PetscDSAddBoundary(prob,DM_BC_NATURAL,"Boundary Integral","marker",0,0,NULL,(void (*)(void))NULL,1,&id,user);CHKERRQ(ierr);
  ierr = PetscDSSetBdResidual(prob,0,f0_bd_u_linear,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob,0,linear_u,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob,1,linear_divu,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create the finite element spaces we will use for this system */
static PetscErrorCode SetupDiscretization(DM mesh,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx *user)
{
  DM             cdm = mesh;
  PetscFE        u,divu;
  const PetscInt dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create FE objects and give them names so that options can be set from
   * command line. Each field gets 2 instances (i.e. velocity and velocity_sum)created twice so that we can compare between approaches. */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,dim,user->simplex,"velocity_",-1,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,1,user->simplex,"divu_",-1,&divu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)divu,"divu");CHKERRQ(ierr);

  ierr = PetscFECopyQuadrature(u,divu);CHKERRQ(ierr);

  /* Associate the FE objects with the mesh and setup the system */
  ierr = DMSetField(mesh,0,NULL,(PetscObject)u);CHKERRQ(ierr);
  ierr = DMSetField(mesh,1,NULL,(PetscObject)divu);CHKERRQ(ierr);
  ierr = DMCreateDS(mesh);CHKERRQ(ierr);
  ierr = (*setup)(mesh,user);CHKERRQ(ierr);

  while (cdm) {
    ierr = DMCopyDisc(mesh,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm,&cdm);CHKERRQ(ierr);
  }

  /* The Mesh now owns the fields, so we can destroy the FEs created in this
   * function */
  ierr = PetscFEDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&divu);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Copy of SNESSolve. May eventually modify this to do more of the non-linear solve process onto the GPU.*/
#if 0
static PetscErrorCode SNESSolve_Ex10(SNES snes,Vec b,Vec x)
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscInt          grid;
  Vec               xcreated = NULL;
  DM                dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (x) PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (x) PetscCheckSameComm(snes,1,x,3);
  if (b) PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  if (b) PetscCheckSameComm(snes,1,b,2);

  /* High level operations using the nonlinear solver */
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscInt          num;
    PetscBool         flg;
    static PetscBool  incall = PETSC_FALSE;

    if (!incall) {
      /* Estimate the convergence rate of the discretization */
      ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) snes),((PetscObject)snes)->options, ((PetscObject) snes)->prefix, "-snes_convergence_estimate", &viewer, &format, &flg);CHKERRQ(ierr);
      if (flg) {
        PetscConvEst conv;
        DM           dm;
        PetscReal   *alpha; /* Convergence rate of the solution error for each field in the L_2 norm */
        PetscInt     Nf;

        incall = PETSC_TRUE;
        ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
        ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
        ierr = PetscCalloc1(Nf, &alpha);CHKERRQ(ierr);
        ierr = PetscConvEstCreate(PetscObjectComm((PetscObject) snes), &conv);CHKERRQ(ierr);
        ierr = PetscConvEstSetSolver(conv, (PetscObject) snes);CHKERRQ(ierr);
        ierr = PetscConvEstSetFromOptions(conv);CHKERRQ(ierr);
        ierr = PetscConvEstSetUp(conv);CHKERRQ(ierr);
        ierr = PetscConvEstGetConvRate(conv, alpha);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
        ierr = PetscConvEstRateView(conv, alpha, viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = PetscConvEstDestroy(&conv);CHKERRQ(ierr);
        ierr = PetscFree(alpha);CHKERRQ(ierr);
        incall = PETSC_FALSE;
      }
      /* Adaptively refine the initial grid */
      num  = 1;
      ierr = PetscOptionsGetInt(NULL, ((PetscObject) snes)->prefix, "-snes_adapt_initial", &num, &flg);CHKERRQ(ierr);
      if (flg) {
        DMAdaptor adaptor;

        incall = PETSC_TRUE;
        ierr = DMAdaptorCreate(PetscObjectComm((PetscObject)snes), &adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorSetSolver(adaptor, snes);CHKERRQ(ierr);
        ierr = DMAdaptorSetSequenceLength(adaptor, num);CHKERRQ(ierr);
        ierr = DMAdaptorSetFromOptions(adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorSetUp(adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorAdapt(adaptor, x, DM_ADAPTATION_INITIAL, &dm, &x);CHKERRQ(ierr);
        ierr = DMAdaptorDestroy(&adaptor);CHKERRQ(ierr);
        incall = PETSC_FALSE;
      }
      /* Use grid sequencing to adapt */
      num  = 0;
      ierr = PetscOptionsGetInt(NULL, ((PetscObject) snes)->prefix, "-snes_adapt_sequence", &num, NULL);CHKERRQ(ierr);
      if (num) {
        DMAdaptor adaptor;

        incall = PETSC_TRUE;
        ierr = DMAdaptorCreate(PetscObjectComm((PetscObject)snes), &adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorSetSolver(adaptor, snes);CHKERRQ(ierr);
        ierr = DMAdaptorSetSequenceLength(adaptor, num);CHKERRQ(ierr);
        ierr = DMAdaptorSetFromOptions(adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorSetUp(adaptor);CHKERRQ(ierr);
        ierr = DMAdaptorAdapt(adaptor, x, DM_ADAPTATION_SEQUENTIAL, &dm, &x);CHKERRQ(ierr);
        ierr = DMAdaptorDestroy(&adaptor);CHKERRQ(ierr);
        incall = PETSC_FALSE;
      }
    }
  }
  if (!x) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&xcreated);CHKERRQ(ierr);
    x    = xcreated;
  }
  ierr = SNESViewFromOptions(snes,NULL,"-snes_view_pre");CHKERRQ(ierr);

  for (grid=0; grid<snes->gridsequence; grid++) {ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes)));CHKERRQ(ierr);}
  for (grid=0; grid<snes->gridsequence+1; grid++) {

    /* set solution vector */
    if (!grid) {ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);}
    ierr          = VecDestroy(&snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol = x;
    ierr          = SNESGetDM(snes,&dm);CHKERRQ(ierr);

    /* set affine vector if provided */
    if (b) { ierr = PetscObjectReference((PetscObject)b);CHKERRQ(ierr); }
    ierr          = VecDestroy(&snes->vec_rhs);CHKERRQ(ierr);
    snes->vec_rhs = b;

    if (snes->vec_rhs && (snes->vec_func == snes->vec_rhs)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Right hand side vector cannot be function vector");
    if (snes->vec_func == snes->vec_sol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be function vector");
    if (snes->vec_rhs  == snes->vec_sol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"Solution vector cannot be right hand side vector");
    if (!snes->vec_sol_update /* && snes->vec_sol */) {
      ierr = VecDuplicate(snes->vec_sol,&snes->vec_sol_update);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)snes,(PetscObject)snes->vec_sol_update);CHKERRQ(ierr);
    }
    ierr = DMShellSetGlobalVector(dm,snes->vec_sol);CHKERRQ(ierr);
    ierr = SNESSetUp(snes);CHKERRQ(ierr);

    if (!grid) {
      if (snes->ops->computeinitialguess) {
        ierr = (*snes->ops->computeinitialguess)(snes,snes->vec_sol,snes->initialguessP);CHKERRQ(ierr);
      }
    }

    if (snes->conv_hist_reset) snes->conv_hist_len = 0;
    if (snes->counters_reset) {snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;}

    ierr = PetscLogEventBegin(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
    ierr = (*snes->ops->solve)(snes);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
    if (!snes->reason) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
    snes->domainerror = PETSC_FALSE; /* clear the flag if it has been set */

    if (snes->lagjac_persist) snes->jac_iter += snes->iter;
    if (snes->lagpre_persist) snes->pre_iter += snes->iter;

    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes),((PetscObject)snes)->options,((PetscObject)snes)->prefix,"-snes_test_local_min",NULL,NULL,&flg);CHKERRQ(ierr);
    if (flg && !PetscPreLoadingOn) { ierr = SNESTestLocalMin(snes);CHKERRQ(ierr); }
    ierr = SNESReasonViewFromOptions(snes);CHKERRQ(ierr);

    if (snes->errorifnotconverged && snes->reason < 0) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_NOT_CONVERGED,"SNESSolve has not converged");
    if (snes->reason < 0) break;
    if (grid <  snes->gridsequence) {
      DM  fine;
      Vec xnew;
      Mat interp;

      ierr = DMRefine(snes->dm,PetscObjectComm((PetscObject)snes),&fine);CHKERRQ(ierr);
      if (!fine) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"DMRefine() did not perform any refinement, cannot continue grid sequencing");
      ierr = DMCreateInterpolation(snes->dm,fine,&interp,NULL);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(fine,&xnew);CHKERRQ(ierr);
      ierr = MatInterpolate(interp,x,xnew);CHKERRQ(ierr);
      ierr = DMInterpolate(snes->dm,interp,fine);CHKERRQ(ierr);
      ierr = MatDestroy(&interp);CHKERRQ(ierr);
      x    = xnew;

      ierr = SNESReset(snes);CHKERRQ(ierr);
      ierr = SNESSetDM(snes,fine);CHKERRQ(ierr);
      ierr = SNESResetFromOptions(snes);CHKERRQ(ierr);
      ierr = DMDestroy(&fine);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes)));CHKERRQ(ierr);
    }
  }
  ierr = SNESViewFromOptions(snes,NULL,"-snes_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(snes->vec_sol,(PetscObject)snes,"-snes_view_solution");CHKERRQ(ierr);
  ierr = DMMonitor(snes->dm);CHKERRQ(ierr);

  ierr = VecDestroy(&xcreated);CHKERRQ(ierr);
  ierr = PetscObjectSAWsBlock((PetscObject)snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode ConstructElementMatrix(DM dm,PetscScalar** elementMat) {
/* Use PetscFEIntegrateJacobian to get element matrix from a single cell. */
  PetscDS prob;
  PetscInt fieldI,fieldJ,numFields;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMGetNumFields(dm,&numFields);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);

  for(fieldI=0; fieldI<numFields; ++fieldI){
    for(fieldJ=0; fieldJ<numFields; ++fieldJ){
      PetscFE testField,basisField;  
      PetscQuadrature quad;
      PetscFEGeom* geom;
      PetscInt dim;

      ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
      ierr = DMGetField(dm,fieldI,NULL,(PetscObject *) &testField);CHKERRQ(ierr);
      ierr = DMGetField(dm,fieldJ,NULL,(PetscObject *) &basisField);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(testField, &quad);CHKERRQ(ierr);
      ierr = PetscFEGeomCreate(quad,1,dim,PETSC_FALSE,&geom);CHKERRQ(ierr);


      //ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN,fieldI,fieldJ,1,)
    }
  }

  PetscFunctionReturn(0);
}

/* Assign element matrix into global matrix */
static PetscErrorCode InsertElementMatrix(DM dm,Mat* globalMat,PetscInt p,const PetscScalar elementMat[]) {
  PetscSection lSec,gSec;
  PetscInt nDoF,poffset;
  PetscInt closureSize;
  PetscInt* closure;
  PetscInt vStart,vEnd,i,idxOffset=0,j;
  PetscInt idxSize=0;
  IS nodeis;
  PetscInt* idx;
  
  PetscErrorCode ierr;

   /* We will loop only over points in the local section. But need to insert into the matrix using global indices.*/
  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,&lSec);CHKERRQ(ierr); /* Maps local mesh points to their DoFs */
  ierr = DMGetGlobalSection(dm,&gSec);CHKERRQ(ierr); /* Maps all mesh points to their DoFs */
  ierr = PetscSectionView(lSec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSectionView(gSec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart,&vEnd);CHKERRQ(ierr);
  /* First need to determine the number of entries that we are modifying */
  for (i=0; i<closureSize; ++i){
    if (closure[i] >= vStart && closure[i] < vEnd){
      ierr = PetscSectionGetDof(gSec,closure[i],&nDoF);CHKERRQ(ierr);
      idxSize += nDoF;
    }
  }
  ierr = PetscCalloc1(idxSize,&idx);CHKERRQ(ierr);

  /* Now we can put the effected indices into an array */
  for (i=0; i<closureSize; ++i){
    if (closure[i] >= vStart && closure[i] < vEnd){
      ierr = PetscSectionGetDof(gSec,closure[i],&nDoF);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(gSec,closure[i],&poffset);CHKERRQ(ierr);
      /* idxOffset tracks where the next entry in idx should go, poffset tells us where the DoFs are in global indexing */
      for (j=idxOffset; j<idxOffset+nDoF; ++j){
        idx[j] = poffset+j-idxOffset;
      }
    }
  }

  ierr = MatSetValues(*globalMat,idxSize,idx,idxSize,idx,elementMat,ADD_VALUES);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode BuildSystemMatrix(DM dm, Mat* systemMat){
  /* Declare basis function coefficients on the reference element. Assuming 2d uniform quadrilaterals. */
  PetscInt cStart,cEnd,c;
  PetscFE field;
  PetscTabulation tab;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetField(dm,0,NULL,&field); CHKERRQ(ierr);
  ierr = PetscFEGetCellTabulation(field,&tab);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,systemMat);CHKERRQ(ierr);
  ierr = MatSetOption(*systemMat,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart,&cEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; ++c){
    ierr = InsertElementMatrix(dm,systemMat,c,tab->T[0]);CHKERRQ(ierr);
  }
  MatAssemblyBegin(*systemMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatAssemblyEnd(*systemMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


int main(int argc,char **argv)
{
  UserCtx         user;
  DM              dm;
  SNES            snes;
  KSP             ksp;  
  PC              pc;
  Vec             u,u1;
  Mat             sysMat,JMat;
  IS*             fieldIS;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);

  /* Set up a snes */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm,SetupProblem,&user);CHKERRQ(ierr);
  ierr = DMCreateFieldIS(dm,NULL,NULL,&fieldIS);CHKERRQ(ierr);

/*
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sysMat,sysMat);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,NULL,fieldIS[0]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,NULL,fieldIS[1]);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
 */ 



  /* Set the system and solve. */
  ierr = DMCreateGlobalVector(dm,&u);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"solution_snes");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes,u,NULL,NULL);CHKERRQ(ierr);
  
  ierr = BuildSystemMatrix(dm,&sysMat);CHKERRQ(ierr);
  ierr = MatViewFromOptions(sysMat,NULL,"-sysMat_view");CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u,NULL,"-solution_snes_view");CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&JMat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatViewFromOptions(JMat,NULL,"-JMat_view");CHKERRQ(ierr);
/*
  ierr = DMCreateGlobalVector(dm,&u1);CHKERRQ(ierr);
  ierr = VecSet(u1,0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u1,"solution_ksp");CHKERRQ(ierr);
  ierr = KSPSolve(ksp,u1,u1);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u1,NULL,"-solution_ksp_view");CHKERRQ(ierr);
*/

  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
 // ierr = VecDestroy(&u1);CHKERRQ(ierr);
 // ierr = MatDestroy(&sysMat);CHKERRQ(ierr);
 // ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 2d_lagrange
    requires: 
    args: -dim 2 \
      -simplex false \
      -dm_vec_type cuda \
      -dm_mat_type aijcusparse \
      -velocity_petscspace_degree 1 \
      -velocity_petscspace_type poly \
      -velocity_petscspace_components 2\
      -velocity_petscdualspace_type lagrange \
      -divu_petscspace_degree 0 \
      -divu_petscspace_type poly \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit\
      -pc_fieldsplit_type schur \
      -pc_fieldsplit_schur_precondition full\
TEST*/
