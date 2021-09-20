static char help[] = "1D droplet formation model using finite-element discretization. \n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscviewerhdf5.h>


typedef struct {
  PetscReal   nu;    /* Kinematic viscosity */
  PetscReal   rho;   /* Fluid density */
  PetscReal   gamma; /* Coefficient of the surface tension. */
  PetscReal   g;     /* Gravitational force per unit mass*/
  PetscReal   u_0;   /* Inlet velocity */
  PetscReal   h_0;   /* Inlet radius */
  PetscReal   dl_dt; /* Droplet length time derivative OR Bottom velocity */
  PetscReal   V_t;   /* Target Drop volume */
  PetscReal   V_old;   /* Start Drop volume */
  PetscReal   length;/* Drop length */
  PetscReal   cellsize;   /* Cellsize */
  PetscReal   fs;       /* Shear force (N/m^2) */
  PetscBool   Bool; /*corrector step */
  PetscReal   factor; /* Floor value of length to radius ratio */
} Parameter;

typedef struct {
  /* Problem definition */
  PetscBag    bag;     /* Holds problem parameters */
  PetscInt    cells[1]; /* Initial mesh division */
  PetscReal   dtInit;  /* Initial timestep */
  PetscInt    bd;
} AppCtx;

/* Initial conditions */
static PetscErrorCode Initial_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

 	PetscScalar h_0 = param->h_0;
  u[0] = PetscSqrtReal(h_0*h_0 - x[0]*x[0]);
  return 0;
}

static PetscErrorCode Initial_u(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  PetscScalar  u_0 = param->u_0;
  u[0] = 0.0;

  return 0;
}

static PetscErrorCode Initial_s(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

 	PetscScalar h_0 = param->h_0;
  u[0] = -x[0]/(PetscSqrtReal(h_0*h_0 - x[0]*x[0]));
  return 0;
}

/* Boundary conditions */
static PetscErrorCode Inlet_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  PetscScalar  h_0   = param->h_0;
  u[0] = h_0;

  return 0;
}

static PetscErrorCode Inlet_h_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Inlet_u(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

  PetscScalar  u_0 = param->u_0;
  u[0] = u_0;
  return 0;
}

static PetscErrorCode Inlet_u_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Inlet_s(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Inlet_s_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Bottom_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Bottom_h_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}


static PetscErrorCode Bottom_u(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

  u[0] = param->dl_dt;
  return 0;
}

static PetscErrorCode Bottom_u_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Bottom_s(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  // AppCtx        *user = (AppCtx *) ctx;
  // Parameter     *param;
  // PetscErrorCode ierr;
  // ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  // u[0] = - 10000*h_0;
  u[0] = -5;
  return 0;
}

static PetscErrorCode Bottom_s_t(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscReal curvature_z(PetscReal h, PetscReal s, PetscReal sx, PetscReal sxx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = -(s*sx/(h*PetscSqrtReal(PetscPowReal(a,3)))) - (s/(h*h*PetscSqrtReal(a))) + (3*s*sx*sx/(PetscSqrtReal(PetscPowReal(a,5)))) - (sxx/(PetscSqrtReal(PetscPowReal(a,3))));
  // PetscReal curve = -(hx*sx/(h*PetscSqrtReal(PetscPowReal(a,3)))) - (hx/(h*h*PetscSqrtReal(a))) + (3*hx*sx*sx/(PetscSqrtReal(PetscPowReal(a,5)))) - (sxx/(PetscSqrtReal(PetscPowReal(a,3))));

  return curve;
}

static PetscReal curvature0_q(PetscReal h, PetscReal s, PetscReal sx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = -(s*sx/(h*PetscSqrtReal(PetscPowReal(a,3)))) - (s/(h*h*PetscSqrtReal(a)));
  // PetscReal curve = -(hx*sx/(h*PetscSqrtReal(PetscPowReal(a,3)))) - (hx/(h*h*PetscSqrtReal(a)));

  return curve;
}
static PetscReal curvature1_q(PetscReal s, PetscReal sx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = (sx/(PetscSqrtReal(PetscPowReal(a,3))));

  return curve;
}
/*
Residual functions.
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{

  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal g = PetscRealPart(constants[3]);
  const PetscReal fs = PetscRealPart(constants[8]);
  PetscScalar  dpdz;

  // dpdz = curvature0_q(u[uOff[1]], u_x[uOff_x[1]], u_x[uOff_x[2]]);
  dpdz = curvature0_q(u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]]);
  // dpdz = curvature_z(u[uOff[1]], u_x[uOff_x[1]], u_x[uOff_x[2]], u_x[uOff_x[2]+3]);
  // dpdz = curvature_z(u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]], u_x[uOff_x[2]+3]);

  // printf("fs = %g\n", fs);

  if(fs==0.0) f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - g;
  else f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - 2*fs/(rho*u[uOff[1]]);

  // if(fs==0.0) f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u[uOff[2]]*u_x[uOff_x[0]]/u[uOff[1]]) - g;
  // else f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u[uOff[2]]*u_x[uOff_x[0]]/u[uOff[1]]) - 2*fs/(rho*u[uOff[1]]);
}
// Use 's' only in dpdz terms in place of dh_dz.
static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal dpdz, lambda, Pe, beta, full_dpdz, strong_res;
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal g = PetscRealPart(constants[3]);
  const PetscReal h = PetscRealPart(constants[7]);
  const PetscReal fs = PetscRealPart(constants[8]);

  dpdz = curvature1_q(u[uOff[2]], u_x[uOff_x[2]]);
  // dpdz = curvature1_q(u_x[uOff_x[1]], u_x[uOff_x[2]]);
  Pe = (u[uOff[0]] - (6*nu/u[uOff[1]])*u_x[uOff_x[1]])*h/(6*nu);
  // Pe = (u[uOff[0]] - (6*nu/u[uOff[1]])*u[uOff[2]])*h/(6*nu);
  // Pe = (u[uOff[0]] - (6*nu/u[uOff[1]])*u[uOff[2]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[0]] - (6*nu/u[uOff[1]])*u_x[uOff_x[1]])*beta*h/(PetscAbsReal(u[uOff[0]] - (6*nu/u[uOff[1]])*u_x[uOff_x[1]])*2.0);
  // lambda = (1.0)*(u[uOff[0]] - (6*nu/u[uOff[1]])*u[uOff[2]])*beta*h/(PetscAbsReal(u[uOff[0]] - (6*nu/u[uOff[1]])*u[uOff[2]])*2.0);
  // lambda = (1.0)*(beta*h)/(PetscAbsReal(u[uOff[0]] - (6*nu/u[uOff[1]])*u_x[uOff_x[1]])*2.0);

  /* Try full SUPG here */
  // f1[0] = 3*nu*u_x[uOff_x[0]];
  f1[0] = 3*nu*u_x[uOff_x[0]] + gamma*dpdz/rho;
  // full_dpdz = curvature_z(u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]], u_x[uOff_x[2]+3]);
  full_dpdz = curvature_z(u[uOff[1]], u_x[uOff_x[1]], u_x[uOff_x[2]], u_x[uOff_x[2]+3]);
  if(fs==0.0) strong_res = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*full_dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - 3*nu*u_x[uOff_x[0]+3] - g;
  else strong_res = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*full_dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - 3*nu*u_x[uOff_x[0]+3] - 2*fs/(rho*u[uOff[1]]);

  // if(fs==0.0) strong_res = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*full_dpdz/rho - (6*nu*u[uOff[2]]*u_x[uOff_x[0]]/u[uOff[1]]) - 3*nu*u_x[uOff_x[0]+3] - g;
  // else strong_res = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*full_dpdz/rho - (6*nu*u[uOff[2]]*u_x[uOff_x[0]]/u[uOff[1]]) - 3*nu*u_x[uOff_x[0]+3] - 2*fs/(rho*u[uOff[1]]);

  // printf("h = %g\t s = %g\t s_x = %g\t s_xx = %g\n",u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]], u_x[uOff_x[0]+3] );
  f1[0] += lambda*strong_res;
}

/* f0_v = dh_dt + v*s + dv_dz * h/2*/
static void f0_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[uOff[1]] + u[uOff[0]]*u_x[uOff_x[1]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]];
  // f0[0] = u_t[uOff[1]] + u[uOff[0]]*u[uOff[2]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]];
}

/* Add Upwinding, f1 = \lambda*(0.5*h*dv_dz) */
static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[7]);

  lambda = (1.0)*h/2.0;
  f1[0] = lambda*(u_t[uOff[1]] + u[uOff[0]]*u_x[uOff_x[1]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]]);
  // f1[0] = lambda*(u_t[uOff[1]] + u[uOff[0]]*u[uOff[2]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]]);
}

/* f0_w = s - dh_dz */
static void f0_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[uOff[2]] - u_x[uOff_x[1]];
}

static void g0_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift + u_x[uOff_x[2]];
}

static void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g1[0] = u[uOff[2]] - 6*nu*u_x[uOff_x[0]]/u[uOff[0]];
}

static void g0_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g0[0]  = 6*nu*u_x[uOff_x[0]]*u_x[uOff_x[2]]/(u[uOff[0]]*u[uOff[0]]);
  g0[0] += (gamma/rho)*(u[uOff[1]]*u_x[uOff_x[1]])/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))); // from first term
  g0[0] += (gamma/rho)*(2.0*u[uOff[1]])/(u[uOff[0]]*u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(1+u[uOff[1]]*u[uOff[1]])); // from second term
}

static void g1_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g1[0] = -6*nu*u_x[uOff_x[2]]/u[uOff[0]];
}

static void g0_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g0[0]  = -(gamma/rho)*((u_x[uOff_x[1]])/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))) - (3.0*u[uOff[1]]*u[uOff[1]]*u_x[uOff_x[1]])/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),5)))); // from first term
  g0[0] -= (gamma/rho)*((1.0)/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(1+u[uOff[1]]*u[uOff[1]])) - (u[uOff[1]]*u[uOff[1]])/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3)))); // from second term
}

static void g1_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g1[0] = -(gamma/rho)*(u[uOff[1]])/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))); // from first term
}

static void g3_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];
  const PetscReal nu = PetscRealPart(constants[0]);

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g3[0] = 3*nu + lambda*(u[uOff[2]] - 6*nu*u_x[uOff_x[0]]/u[uOff[0]]); /* include SUPG */
  // g3[0] = 3*nu; /* No SUPG */
}

static void g2_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal nu = PetscRealPart(constants[0]);

  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g2[0] = -(gamma/rho)*(3.0*u[uOff[1]]*u_x[uOff_x[1]])/(PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),5)));

  /* Add SUPG terms */
  g2[0] -= (gamma*lambda/rho)*((u_x[uOff_x[1]])/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))) - (3.0*u[uOff[1]]*u[uOff[1]]*u_x[uOff_x[1]])/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),5)))); // from first term
  g2[0] -= (gamma*lambda/rho)*((1.0)/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(1+u[uOff[1]]*u[uOff[1]])) - (u[uOff[1]]*u[uOff[1]])/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3)))); // from second term
  g2[0] += (gamma*lambda/rho)*((3*u_x[uOff_x[1]]*u_x[uOff_x[1]]/PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))) - (9*u[uOff[1]]*u[uOff[1]]*u_x[uOff_x[1]]*u_x[uOff_x[1]]/PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),5)))); // from third term
  g2[0] += (gamma*lambda/rho)*(3*u[uOff[1]]*u_x[uOff_x[1]+3]/PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),5))); //from fourth term
}

static void g3_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal nu = PetscRealPart(constants[0]);

  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g3[0] = (gamma/rho)*(1.0)/(PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3)));

  /* Add SUPG terms */
  g3[0] += (gamma*lambda/rho)*(-(u[uOff[1]]/(u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3)))) + (6*u[uOff[1]]*u_x[uOff_x[1]]/PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))));
}

/* Jacobians from SUPG terms*/
static void g2_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];
  const PetscReal nu = PetscRealPart(constants[0]);

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g2[0] = lambda*(u_tShift + u_x[uOff_x[2]]);
}

static void g2_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  // beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g2[0]  = lambda*(6*nu*u_x[uOff_x[0]]*u_x[uOff_x[2]]/(u[uOff[0]]*u[uOff[0]]));
  g2[0] += (gamma*lambda/rho)*(u[uOff[1]]*u_x[uOff_x[1]])/(u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(PetscPowReal((1+u[uOff[1]]*u[uOff[1]]),3))); // from first term
  g2[0] += (gamma*lambda/rho)*(2.0*u[uOff[1]])/(u[uOff[0]]*u[uOff[0]]*u[uOff[0]]*PetscSqrtReal(1+u[uOff[1]]*u[uOff[1]])); // from second term
}

static void g3_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal beta, Pe, lambda;
  const PetscReal h = constants[7];
  const PetscReal nu = PetscRealPart(constants[0]);

  Pe = (u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*h/(6*nu);

  // beta = (1/tanh(Pe)) - (1/Pe); /* 1st order */
  // beta = PetscSqrtReal(Pe*Pe/(1+Pe*Pe)); /* 2nd order */
  beta = PetscSqrtReal(Pe*Pe/(9+Pe*Pe)); /* 4th order */
  lambda = (1.0)*(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*beta*h/(PetscAbsReal(u[uOff[2]] - (6*nu/u[uOff[0]])*u_x[uOff_x[0]])*2.0);

  g3[0] = -lambda*(6*nu*u_x[uOff_x[2]]/u[uOff[0]]);
}



static void g0_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift + 0.5*u_x[uOff_x[2]];
}

static void g1_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  g1[0] = u[uOff[2]];
}

static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_x[uOff_x[0]];
}

static void g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  g1[0] = 0.5*u[uOff[0]];
}

/* Jacobians for stabilization */
static void g2_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[7];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*(u_tShift + 0.5*u_x[uOff_x[2]]);
}
static void g3_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[7];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*u[uOff[2]];
}
static void g2_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[7];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*u_x[uOff_x[2]];
}
static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[7];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*0.5*u[uOff[0]];
}



static void g0_ws(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static void g1_wh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  g1[0] = -1.0;
}


static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n=1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscOptionsBegin(comm, "", "1D Droplet Problem Options", "DMPLEX");CHKERRQ(ierr);

  options->cells[0] = 25;
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "droplet.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *user)
{
  PetscBag       bag;
  Parameter     *param;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->bag, "par", "Droplet parameters");CHKERRQ(ierr);
  bag  = user->bag;
  ierr = PetscBagRegisterReal(bag, &param->nu,    1.0e-6,  "nu",    "Kinematic viscosity(m^2/sec)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->rho,   997.0,   "rho",   "Fluid density(kg/m^3)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->gamma, 0.0728,  "gamma", "Coefficient of surface tension(kg/sec^2)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->g,     9.81,    "g",     "Gravitational force per unit mass(m/sec^2)");CHKERRQ(ierr);

  ierr = PetscBagRegisterReal(bag, &param->u_0,   1.0,     "u_0",   "Inlet velocity(m/s)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->h_0,   1.0,     "h_0",   "Inlet radius(m)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->dl_dt, 0.0,     "dl_dt", "Update in Length of a drop per time-step");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->length, param->h_0,  "length", "Length of a drop at time t");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->V_t,   0.0,    "V_t",   "Target drop volume");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->V_old, 2*PETSC_PI*(param->h_0)*(param->h_0)*(param->h_0)/3,    "V_old",   "Initial drop volume");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->fs,    0.0,    "fs",   "Shear force per unit area");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->cellsize, 0.0,    "cellsize",   "Cell size");CHKERRQ(ierr);
  ierr = PetscBagRegisterBool(bag, &param->Bool, PETSC_TRUE,    "Correction",   "Correction step");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->factor, 1.0,    "factor",   "Floor value of length to radius ratio");CHKERRQ(ierr);

  ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  Parameter     *param;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  PetscReal upper = param->h_0;

  ierr = DMPlexCreateBoxMesh(comm, 1, PETSC_FALSE, user->cells, NULL, &upper, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  // ierr = DMPlexCreateBoxMesh(comm, 1, PETSC_FALSE, NULL, NULL, &upper, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS          prob;
  // const PetscInt   id1=1, id2=2;
  PetscInt         id;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

  /* Residual terms */

  ierr = PetscDSSetResidual(prob, 0, f0_q, f1_q);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_v, f1_v);CHKERRQ(ierr);
  // ierr = PetscDSSetResidual(prob, 1, f0_v, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 2, f0_w, NULL);CHKERRQ(ierr);

  /* Jacobian terms without SUPG */ /*
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_qh, g1_qh, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, g0_qs, g1_qs, g2_qs, g3_qs);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 2, g0_qu, g1_qu, NULL, g3_qu);CHKERRQ(ierr);

  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_wh, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 1, g0_ws, NULL,  NULL, NULL);CHKERRQ(ierr);

  ierr = PetscDSSetJacobian(prob, 2, 0, g0_vh, g1_vh,  NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 2, 2, g0_vu, g1_vu,  NULL, NULL);CHKERRQ(ierr);
  */

  /* Jacobian terms with SUPG terms */
  //
  // ierr = PetscDSSetJacobian(prob, 0, 0, g0_qh, g1_qh, g2_qh, g3_qh);CHKERRQ(ierr);
  // ierr = PetscDSSetJacobian(prob, 0, 1, g0_qs, g1_qs, g2_qs, g3_qs);CHKERRQ(ierr);
  // ierr = PetscDSSetJacobian(prob, 0, 2, g0_qu, g1_qu, g2_qu, g3_qu);CHKERRQ(ierr);
  //
  // ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_wh, NULL, NULL);CHKERRQ(ierr);
  // ierr = PetscDSSetJacobian(prob, 1, 1, g0_ws, NULL,  NULL, NULL);CHKERRQ(ierr);
  //
  // ierr = PetscDSSetJacobian(prob, 2, 0, g0_vh, g1_vh,  g2_vh, g3_vh);CHKERRQ(ierr);
  // ierr = PetscDSSetJacobian(prob, 2, 2, g0_vu, g1_vu,  g2_vu, g3_vu);CHKERRQ(ierr);


  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[9];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[0] = param->nu;
    constants[1] = param->rho;
    constants[2] = param->gamma;
    constants[3] = param->g;
    constants[4] = param->dl_dt;
    constants[5] = param->u_0;
    constants[6] = param->h_0;
    constants[7] = param->cellsize;
    constants[8] = param->fs;

    ierr = PetscDSSetConstants(prob, 9, constants);CHKERRQ(ierr);
  }

  /* Setup Boundary Conditions */
  // PetscInt       bd;
  // DMLabel        label;
  // ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  id = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Inlet velocity",  "marker", 0, 0, NULL, (void (*)(void)) Inlet_u, (void (*)(void)) Inlet_u_t, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Inlet radius",    "marker", 1, 0, NULL, (void (*)(void)) Inlet_h, (void (*)(void)) Inlet_h_t, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Inlet slope",     "marker", 2, 0, NULL, (void (*)(void)) Inlet_s, (void (*)(void)) Inlet_s_t, 1, &id, user);CHKERRQ(ierr);

  id = 2;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Bottom velocity",  "marker", 0, 0, NULL, (void (*)(void)) Bottom_u, (void (*)(void)) Bottom_u_t, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Bottom radius",    "marker", 1, 0, NULL, (void (*)(void)) Bottom_h, (void (*)(void)) Bottom_h_t, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Bottom slope",     "marker", 2, 0, NULL, (void (*)(void)) Bottom_s, (void (*)(void)) Bottom_s_t, 1, &id, user);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscFE         fe[3];
  MPI_Comm        comm;
  PetscInt        dim;
  PetscBool       simplex = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "rad_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "radius");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "slope_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "slope");CHKERRQ(ierr);

  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 2, NULL, (PetscObject) fe[2]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);

  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[2]);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM               dm;
  PetscReal        t;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  void            *ctxs[3];
  AppCtx          *ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  funcs[0] = Initial_u;
  funcs[1] = Initial_h;
  funcs[2] = Initial_s;
  ctxs[0] = ctx;
  ctxs[1] = ctx;
  ctxs[2] = ctx;
  ierr = DMProjectFunction(dm, t, funcs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FieldFunction_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords, locX;
  DM                 dm;
  DMField            coordField;
  PetscDS            ds;
  PetscFE            fe;
  PetscQuadrature    q;
  PetscFEGeom       *cgeom;
  PetscSF            cellSF=NULL;
  IS                 cellIS;
  PetscTabulation    T;
  PetscScalar       *coeff = NULL;
  const PetscSFNode *cellsRem;
  PetscInt           N, cell, Ncl, fc, f;
  PetscReal          xRef;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  // Locate point in original mesh
  ierr = VecGetDM(uV, &dm);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, uV, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal *), &xRef);

  if (N != 1) {
    AppCtx     *user = (AppCtx *) ctx;
    Parameter  *param;

    ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
    u[0] = Bottom_u(dim, time, x, Nc, u, user);

    PetscFunctionReturn(0);
  }

  cell = cellsRem[0].index;
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  // Create geometry
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS);CHKERRQ(ierr);
  ierr = DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);

  /* Interpolate field values */
  ierr = DMPlexVecGetClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesToReference(dm, cell, 1, x, &xRef);CHKERRQ(ierr);
  ierr = PetscFECreateTabulation(fe, 1, 1, &xRef, 0, &T);CHKERRQ(ierr);
  if (Nc != T->Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %D != %D Number of field cmponents", Nc, T->Nc);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;

    for (fc = 0; fc < Nc; ++fc) {
      u[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        // u[fc] += coeff[(Ncl-Nb) + f]*basis[(0*Nb + f)*Nc + fc];
        u[fc] += coeff[f]*basis[(0*Nb + f)*Nc + fc];
      }
    }
  }
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  ierr = PetscFEPushforward(fe, cgeom, 1, u);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&cgeom);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode FieldFunction_h(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords, locX;
  DM                 dm;
  DMField            coordField;
  PetscDS            ds;
  PetscFE            fe;
  PetscQuadrature    q;
  PetscFEGeom       *cgeom;
  PetscSF            cellSF=NULL;
  IS                 cellIS;
  PetscTabulation    T;
  PetscScalar       *coeff = NULL;
  const PetscSFNode *cellsRem;
  PetscInt           N, cell, Ncl, fc, f;
  PetscReal          xRef;
  PetscErrorCode     ierr;


  PetscFunctionBeginUser;
  /* Locate point in original mesh */
  ierr = VecGetDM(uV, &dm);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal *), &xRef);

  if (N != 1) {
    void     *ctx;

    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    ierr = Bottom_h(dim, time, x, Nc, u, ctx);
    PetscFunctionReturn(0);
  }
  cell = cellsRem[0].index;
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);

   /* Create geometry */
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, 1, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS);CHKERRQ(ierr);
  ierr = DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);

  /* Get cell coefficients and Interpolate field values */
  ierr = DMCreateLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, uV, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesToReference(dm, cell, 1, x, &xRef);CHKERRQ(ierr);
  ierr = PetscFECreateTabulation(fe, 1, 1, &xRef, 0, &T);CHKERRQ(ierr);
  if (Nc != T->Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %D != %D Number of field cmponents", Nc, T->Nc);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;

    for (fc = 0; fc < Nc; ++fc) {
      u[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        u[fc] += coeff[(Ncl-2*Nb+1) + f]*basis[(0*Nb + f)*Nc + fc];
      }
    }
  }
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  ierr = PetscFEPushforward(fe, cgeom, 1, u);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&cgeom);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode FieldFunction_s(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords, locX;
  DM                 dm;
  DMField            coordField;
  PetscDS            ds;
  PetscFE            fe;
  PetscQuadrature    q;
  PetscFEGeom       *cgeom;
  PetscSF            cellSF=NULL;
  IS                 cellIS;
  PetscTabulation    T;
  PetscScalar       *coeff = NULL;
  const PetscSFNode *cellsRem;
  PetscInt           N, cell, Ncl, fc, f;
  PetscReal          xRef;
  PetscErrorCode     ierr;


  PetscFunctionBeginUser;
  /* Locate point in original mesh */
  ierr = VecGetDM(uV, &dm);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal *), &xRef);

  if (N != 1) {
    void     *ctx;

    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    ierr = Bottom_s(dim, time, x, Nc, u, ctx);

    PetscFunctionReturn(0);
  }
  cell = cellsRem[0].index;
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);

   /* Create geometry */
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, 2, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS);CHKERRQ(ierr);
  ierr = DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);

  /* Get cell coefficients and Interpolate field values */
  ierr = DMCreateLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, uV, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesToReference(dm, cell, 1, x, &xRef);CHKERRQ(ierr);
  ierr = PetscFECreateTabulation(fe, 1, 1, &xRef, 0, &T);CHKERRQ(ierr);
  if (Nc != T->Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %D != %D Number of field cmponents", Nc, T->Nc);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;

    for (fc = 0; fc < Nc; ++fc) {
      u[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        u[fc] += coeff[(Ncl-Nb) + f]*basis[(0*Nb + f)*Nc + fc];
      }
    }
  }
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  ierr = PetscFEPushforward(fe, cgeom, 1, u);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&cgeom);CHKERRQ(ierr);
  ierr = DMPlexVecRestoreClosure(dm, NULL, locX, cell, &Ncl, &coeff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscReal volume(DM dm, Vec u, PetscReal time)
{
  Vec                locX, coordinates;
  PetscDS            ds;
  PetscFE            fe;
  AppCtx             *user;
  Parameter          *param;
  PetscQuadrature    q;
  PetscInt           cStart, cEnd, c, nq, feDim, N;
  PetscReal          *z=NULL, *h = NULL, v=0.0;
  const PetscReal    *points_w;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, u, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &z);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinates, &z);CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, 1, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &feDim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, NULL, &nq, NULL, &points_w);CHKERRQ(ierr);
  // Get velocity

  for(c=cStart; c<cEnd; c++)
  {
    PetscInt    i, Ncl;
    PetscReal   detJ, Vol=0.0;

    ierr = DMPlexVecGetClosure(dm, NULL, locX, c, &Ncl, &h);CHKERRQ(ierr);
    for(i=0; i<nq; i++){
      Vol += points_w[i]*PETSC_PI*(h[(Ncl-(2*feDim)+1)+i]*h[(Ncl-(2*feDim)+1)+i]);
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, locX, c, &Ncl, &h);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFEM(dm, c, q, NULL, NULL, NULL, &detJ);CHKERRQ(ierr);
    v += Vol*detJ;
  }
  ierr = VecDestroy(&locX);
  PetscFunctionReturn(v);
}

static PetscErrorCode TSAdaptChoose_Volume(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea, PetscReal *wlter)
{
  DM                dm;
  PetscReal         time, dt, step, Flow_in, V_new, V_lost, e=0.10;
  AppCtx            *user;
  Parameter         *param;
  Vec               u;
  PetscInt          N;
  PetscDS           prob;
  const PetscInt    id=2;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &step);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);

  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSUpdateBoundary(prob, 2, DM_BC_ESSENTIAL, "Bottom velocity",  "marker", 0, 0, NULL, (void (*)(void)) Bottom_u, (void (*)(void)) Bottom_u_t, 1, &id, user);CHKERRQ(ierr);
  V_new = volume(dm, u, time);
  V_lost = 100*(param->V_t - V_new)/param->V_t;

/* Check if new volume is close to the tagret volume. */
  if((PetscAbsReal(V_lost)<=e)) {
    *accept  = PETSC_TRUE;
    printf("V_target = %g \t V_new = %g \t Volume_lost = %g %% \t tip velocity = %2.10f \t Predicted length = %2.10f\n \n", param->V_t, V_new, V_lost , param->dl_dt, param->length );
  }
  else {
    *accept  = PETSC_FALSE;

    DM                dmNew;
    Vec               U, coordinates;
    PetscScalar       *z, endpoint;
    PetscInt          N;
    PetscErrorCode    (*feFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
    void              *fectxs[3];

    ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetSize(coordinates, &N);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &z);CHKERRQ(ierr);
    endpoint = z[N-1];
    ierr = VecRestoreArray(coordinates, &z);CHKERRQ(ierr);

  /* Correct the length and boundary condition. Then scale and project again using corrected length */

    param->length = endpoint*(param->V_t/V_new);
    param->dl_dt = (param->length - param->h_0)/(time + dt);
    printf("\n CORRECTION because of Volume_lost = %g %% \n Previus length = %2.10f \t Corrected length = %2.10f\n\n", V_lost, endpoint, param->length);

    {
      PetscDS           probNew;
      ierr = DMGetDS(dm, &probNew);CHKERRQ(ierr);
      ierr = PetscDSUpdateBoundary(probNew, 2, DM_BC_ESSENTIAL, "Bottom velocity",  "marker", 0, 0, NULL, (void (*)(void)) Bottom_u, (void (*)(void)) Bottom_u_t, 1, &id, user);CHKERRQ(ierr);
    }

    ierr = DMClone(dm, &dmNew);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, dmNew);CHKERRQ(ierr);

    ierr = VecScale(coordinates,param->length/endpoint);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmNew, coordinates);CHKERRQ(ierr);

    feFuncs[0] = FieldFunction_u;
    feFuncs[1] = FieldFunction_h;
    feFuncs[2] = FieldFunction_s;
    fectxs[0]  = (void *) u;
    fectxs[1]  = (void *) u;
    fectxs[2]  = (void *) u;
    ierr = DMCreateGlobalVector(dmNew, &U);CHKERRQ(ierr);
    ierr = DMProjectFunction(dmNew, time, feFuncs, fectxs, INSERT_ALL_VALUES, U);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dmNew);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, U);
  }

  *next_h  = h;  /* Reuse the same time step */
  *next_sc = 0;  /* Reuse the same order scheme */
  *wlte    = -1; /* Weighted local truncation error was not evaluated */
  *wltea   = -1; /* Weighted absolute local truncation error was not evaluated */
  *wlter   = -1; /* Weighted relative local truncation error was not evaluated */
  PetscFunctionReturn(0);
}

/* Monitor routine */
static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  DM               dm;
  PetscErrorCode   ierr;
  Vec              coordinates;
  PetscInt         stepi;
  PetscReal        time;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);

  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "length");CHKERRQ(ierr);
  // ierr = VecViewFromOptions(coordinates, NULL, "-len_vec_view");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
  // ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
  {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "length.out", &viewer);CHKERRQ(ierr);
    ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.out", &viewer);CHKERRQ(ierr);
    ierr = VecView(u, viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* Calculate the target volume and predicted length before a time step using previous successful step. Then project on a new mesh. */
PetscErrorCode PreStep(TS ts)
{
  DM                dm, dmNew;
  PetscReal         dt, time, Flow_in;
  AppCtx            *user;
  Parameter         *param;
  PetscInt          stepi, N;
  Vec               u, coordinates, U;
  PetscScalar       *z, endpoint;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  // if (!stepi) user->dtInit = dt;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);

  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &z);CHKERRQ(ierr);
  endpoint = z[N-1];
  ierr = VecRestoreArray(coordinates, &z);CHKERRQ(ierr);

  Flow_in = PETSC_PI*param->h_0*param->h_0*param->u_0*dt;
  param->V_t = param->V_old + Flow_in;
  param->length = param->length*(param->V_t/param->V_old);  // param->dl_dt = (param->length - endpoint)/(dt);
  param->dl_dt = (param->length - param->h_0)/(time+dt);
  param->V_old = param->V_t;

  PetscReal X = endpoint-(param->h_0), n=01.0;
  {
    const PetscScalar *Consts;
    PetscInt          NC;
    PetscDS           prob;

    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSGetConstants(prob, &NC, &Consts);CHKERRQ(ierr);
    PetscScalar       *c = Consts;
    param->cellsize = param->length/(N-1);
    *(c+7) = param->cellsize;

    /*
    For low viscosity fluid, set the viscosity to higher value and gradually reduce it. This is to smooth out the initial large perturbations.
    */

    if(X<(n*(param->h_0))) {
      *(c+0) = -((param->nu*9)/(n*(param->h_0)))*X + param->nu*10;
      // *(c+0) = param->nu*100;
    }
    // else *(c+0) = param->nu;
    // else if (X>(n*(param->h_0)) && X<((n+1.0)*(param->h_0))) {
    //   *(c+0) = -((param->nu*99)/((n+1.0)*(param->h_0)))*X + param->nu*100;
    // }
    else if(param->Bool){
        *(c+0) = param->nu;
        // ierr = TSSetTimeStep(ts, dt*0.1);CHKERRQ(ierr);
        param->Bool = PETSC_FALSE;
    }

    ierr = PetscDSSetConstants(prob, NC, c);CHKERRQ(ierr);
    printf(" nu = %g \t Inlet Velocity = %g \n", *(c+0), *(c+5));
  }


/* Refine the mesh if the length is n*(initial_length), where n is an integer. */
  PetscReal scale = PetscFloorReal(param->length/param->h_0);
  printf("scale = %g \t factor = %g\n",scale, param->factor );
  // if(scale>(param->factor))
  if((stepi) && !(stepi%2))
  {
    Vec        coordsNew;
    PetscInt   NN;

    ierr = DMRefine(dm, PETSC_COMM_WORLD, &dmNew);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmNew);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dmNew, &coordsNew);CHKERRQ(ierr);
    ierr = VecGetSize(coordsNew, &NN);CHKERRQ(ierr);
    printf("Coordinates size = %d \n", NN );
    ierr = VecScale(coordsNew,param->length/endpoint);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmNew, coordsNew);CHKERRQ(ierr);
    // param->factor = scale;
  }
  else{
    ierr = DMClone(dm, &dmNew);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, dmNew);CHKERRQ(ierr);
    ierr = VecScale(coordinates,param->length/endpoint);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmNew, coordinates);CHKERRQ(ierr);
  }

/* Projection */
  PetscErrorCode    (*feFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
  void              *fectxs[3];
  feFuncs[0] = FieldFunction_u;
  feFuncs[1] = FieldFunction_h;
  feFuncs[2] = FieldFunction_s;
  fectxs[0]  = (void *) u;
  fectxs[1]  = (void *) u;
  fectxs[2]  = (void *) u;
  ierr = DMCreateGlobalVector(dmNew, &U);CHKERRQ(ierr);
  ierr = DMProjectFunction(dmNew, time, feFuncs, fectxs, INSERT_ALL_VALUES, U);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dmNew, stepi, time);CHKERRQ(ierr);


  /******** Testing refinement *********/
  /*
  if (X>(n*(param->h_0)) && (param->Bool)){
    Vec        uf;
    DM         dmf=NULL;
    MPI_Comm   comm;

    ierr = PetscObjectGetComm((PetscObject)dmNew, &comm);CHKERRQ(ierr);
    ierr = DMRefine(dmNew, comm, &dmf);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dmf, "Refined Mesh");CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmf);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dmf, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dmf, &uf);CHKERRQ(ierr);

    feFuncs[0] = FieldFunction_u;
    feFuncs[1] = FieldFunction_h;
    feFuncs[2] = FieldFunction_s;
    fectxs[0]  = (void *) U;
    fectxs[1]  = (void *) U;
    fectxs[2]  = (void *) U;
    ierr = DMProjectFunction(dmf, time, feFuncs, fectxs, INSERT_ALL_VALUES, uf);CHKERRQ(ierr);
    ierr = DMDestroy(&dmNew);CHKERRQ(ierr);
    param->Bool = PETSC_FALSE;
    ierr = TSReset(ts);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dmf);CHKERRQ(ierr);
    else {
    ierr = TSSetSolution(ts, uf);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, dt*0.1);CHKERRQ(ierr);

  }
    ierr = TSReset(ts);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dmNew);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, U);CHKERRQ(ierr);
  }
*/
  /*************************************/

  ierr = TSReset(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, U);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dmNew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc, char **argv)
{
  DM              dm;   /* problem definition */
  TS              ts;   /* timestepper */
  Vec             u;    /* solution */
  AppCtx          user; /* user-defined work context */
  PetscReal       t;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
  ierr = SetupParameters(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

/* Chnge the mesh density at varius locations */ /*
  {
    // DM         dm2;
    Vec        c, d, d1, d2, e;
    PetscReal  a=0;
    PetscReal  b=1.2;
    AppCtx     *ctx;
    Parameter  *param;

    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    ierr = PetscBagGetData(ctx->bag, (void **) &param);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dm, &c);CHKERRQ(ierr);
    ierr = VecDuplicate(c, &d);
    ierr = VecDuplicate(c, &d1);
    ierr = VecDuplicate(c, &d2);
    ierr = VecDuplicate(c, &e);
    ierr = VecCopy(c, d1);
    ierr = VecCopy(c, d2);
    ierr = VecSet(e, 1.0);

// Dense mesh at the top
    // ierr = VecScale(d1, (2*a+1)/(param->h_0));
    // ierr = VecScale(d2, -(2*a+1)/(param->h_0));
    // ierr = VecAXPY(d1, (b-2*a), e);
    // ierr = VecAXPY(d2, (b+2*a), e);
    // ierr = VecPointwiseDivide(d, d1, d2);
    // ierr = VecLog(d);
    // ierr = VecAXPBY(d, a, ((param->h_0 - a)/(PetscLogReal((b+1)/(b-1)))), e);

// Dense in the middle
    // ierr = VecDestroy(&d1);CHKERRQ(ierr);
    // ierr = VecDestroy(&d2);CHKERRQ(ierr);
    // ierr = VecDuplicate(d, &d1);
    // ierr = VecDuplicate(d, &d2);
    // ierr = VecCopy(d, d1);
    // ierr = VecCopy(d, d2);
    // b = 1.25;

// Dense mesh at the bottom
    ierr = VecScale(d1, -1/(param->h_0));
    ierr = VecScale(d2, 1/(param->h_0));
    ierr = VecAXPY(d1, (b+1), e);
    ierr = VecAXPY(d2, (b-1), e);
    ierr = VecPointwiseDivide(d, d1, d2);
    ierr = VecLog(d);
    ierr = VecAXPBY(d, param->h_0, -((param->h_0)/(PetscLogReal((b+1)/(b-1)))), e);

    // ierr = VecView(d, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dm, d);CHKERRQ(ierr);

    ierr = VecDestroy(&d1);CHKERRQ(ierr);
    ierr = VecDestroy(&d2);CHKERRQ(ierr);
    ierr = VecDestroy(&e);CHKERRQ(ierr);
    ierr = VecDestroy(&d);CHKERRQ(ierr);
  }
  */

  /* Setup problem */
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
/*
  {
    PetscInt  i;
    PetscDS   ds;
    const char *name[6];
    ierr = DMGetDS(dm, &ds);
    for (i=0; i<6; i++) {
      ierr = PetscDSGetBoundary(ds, i, NULL, &name[i], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
      printf("BC number = %d \t BC name = %s\n",i, name[i]);
    }
  }
*/
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM , &user);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_INTERPOLATE);CHKERRQ(ierr);
  // ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
  ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, MonitorError, &user, NULL);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = TSSetPreStep(ts, PreStep);CHKERRQ(ierr);

  {
    TSAdapt   adapt;
    ierr = TSGetAdapt(ts, &adapt);CHKERRQ(ierr);
    adapt->ops->choose = TSAdaptChoose_Volume;
  }
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* TEST

test1: Water-Glycerol (25-75)
  args: -h_0 0.0026 -u_0 0.01 -nu 0.00005 -rho 1205 -gamma 0.0675 -cells 50 -dm_plex_separate_marker -ts_max_steps 5 -ts_dt 1e-4 \
  -vel_petscspace_degree 3 -rad_petscspace_degree 3 -slope_petscspace_degree 2 -ts_fd_color  -dm_ds_jet_degree 2 \
  -ts_type beuler -pc_type lu -snes_monitor -ts_monitor -ksp_gmres_restart 500 \
  -ksp_error_if_not_converged -ksp_converged_reason -ksp_monitor_true_residual -snes_converged_reason -snes_max_funcs 1000000

  ################ Properties to use ###############
  #### Water:
  # rho = 997
  # nu  = 1e-6
  # gamma = 0.0728
  ### Glycerol:
  # rho = 1260
  # nu  = 0.00112
  # gamma =  0.0634
  ### 5-95 Water-Glycerol ###
  # rho = 1250
  # nu = 0.0005
  # gamma = 0.065

*/
