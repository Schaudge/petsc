static char help[] = "1D droplet formation model using finite-element discretization. \n\n\n";

/*
For visualization, use
-dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append
*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>
#include <petscbag.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscviewerhdf5.h>

typedef struct {
  PetscReal   nu;            /* Kinematic viscosity */
  PetscReal   rho;           /* Fluid density */
  PetscReal   gamma;         /* Coefficient of the surface tension. */
  PetscReal   g;             /* Gravitational force per unit mass */
  PetscReal   u_0;           /* Inlet velocity */
  PetscReal   h_0;           /* Inlet radius */
  PetscReal   cellsize;      /* Cellsize */
  PetscReal   fs;            /* Shear force (N/m^2) */
  PetscReal   x_p;
} Parameter;

typedef struct {
  PetscBag  bag;      /* Holds problem parameters */
  PetscReal V_old;    /* Starting drop volume */
  PetscReal V_t;      /* Target drop volume */
  PetscReal l;        /* Droplet length */
  PetscReal dl_dt;    /* Droplet length time derivative OR bottom velocity */
  PetscReal s_bottom; /* Bottom s (or dh_dz) for projection */
  PetscInt  bd_in;    /* BC number for inlet condition */
  PetscInt  bd_end;   /* BC number for droplet bottom condition */
  PetscInt  cells[1]; /* Initial mesh division */
  PetscBool necking;  /* Necking test */
  PetscBool Bool;
  PetscReal factor;   /* For adaptivity */
} AppCtx;

/* Initial conditions */
static PetscErrorCode Initial_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;
  PetscErrorCode ierr;
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

  PetscScalar h_0 = param->h_0;
 	PetscScalar l_0 = user->l;
  /* Make sure to change the initial volume and length according to the initial curvature */

  // u[0] = PetscCbrtReal(h_0*h_0*h_0 - x[0]*x[0]*x[0]); /* cubic curve */
  u[0] = (h_0/l_0)*(PetscSqrtReal(l_0*l_0 - x[0]*x[0])); /* hemi ellipse curve (h_0=l_0 gives you a hemi sphere) */
  return 0;
}

static PetscErrorCode Initial_u(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
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
 	PetscScalar l_0 = user->l;
  // u[0] = -(x[0]*x[0])/(PetscCbrtReal((h_0*h_0*h_0 - x[0]*x[0]*x[0])*(h_0*h_0*h_0 - x[0]*x[0]*x[0]))); /* cubic curve derivative */
  if(x[0]<l_0) u[0] = -h_0*x[0]/(l_0*PetscSqrtReal(l_0*l_0 - x[0]*x[0])); /* hemi ellipse curve derivative */
  else u[0] = -10.0;
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
  u[0] = ((AppCtx *) ctx)->dl_dt;
  return 0;
}

static PetscErrorCode Bottom_s(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = ((AppCtx *) ctx)->s_bottom;
  return 0;
}

static PetscReal curvature0_q(PetscReal h, PetscReal s, PetscReal sx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = -(s*sx/(h*PetscSqrtReal(PetscPowReal(a,3)))) - (s/(h*h*PetscSqrtReal(a)));
  return curve;
}
static PetscReal curvature1_q(PetscReal s, PetscReal sx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = (sx/(PetscSqrtReal(PetscPowReal(a,3))));

  return curve;
}
static PetscReal curvature0_bd_q(PetscReal s, PetscReal sx)
{
  PetscReal a = 1 + s*s;

  PetscReal curve = -(sx/(PetscSqrtReal(PetscPowReal(a,3))));
  return curve;
}

static void id_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}
static void id_r(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[1];
}
static void id_s(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[2];
}
static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal x_p =  PetscRealPart(constants[8]);
  if(x[0]<x_p) {f0[0] = 0.0;}
  else {f0[0] = PETSC_PI * PetscSqr(u[1]);}
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
  const PetscReal fs = PetscRealPart(constants[7]);

  PetscScalar  dpdz;

  dpdz = curvature0_q(u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]]);
  if(!fs){
    const PetscReal g = PetscRealPart(constants[3]);
    f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - g;
  }
  else {
    f0[0] = u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho - (6*nu*u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]]) - 2*fs/(rho*u[uOff[1]]);
  }
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal dpdz;
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho = PetscRealPart(constants[1]);

  dpdz = curvature1_q(u[uOff[2]], u_x[uOff_x[2]]);
  f1[0] = 3*nu*u_x[uOff_x[0]] + gamma*dpdz/rho;
}

static void f0_bd_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho = PetscRealPart(constants[1]);
  PetscReal       dpdz_bd = curvature0_bd_q(u[uOff[2]], u_x[uOff_x[2]]);

  f0[0] = -3.0*nu*u_x[uOff_x[0]] - gamma*dpdz_bd/rho;
}

static void f0_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[uOff[1]] + u[uOff[0]]*u_x[uOff_x[1]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]];
}

static void f1_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);

  lambda = (01.0)*h/2.0;
  f1[0] = lambda*(u_t[uOff[1]] + u[uOff[0]]*u_x[uOff_x[1]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]]);
}

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
  g0[0] = u_tShift + u_x[uOff_x[0]];
}

static void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g1[0] = u[uOff[0]] - (6*nu/u[uOff[1]])*u_x[uOff_x[1]];
}

static void g3_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g3[0] = 3*nu;
}

static void g1_bd_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g1[0] = -3.0*nu;
}

static void g0_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal nu = PetscRealPart(constants[0]);
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal fs = PetscRealPart(constants[7]);

  g0[0]  = (6*nu/(u[uOff[1]]*u[uOff[1]]))*u_x[uOff_x[0]]*u_x[uOff_x[1]];
  if (fs) g0[0] += 2*fs/(rho*u[uOff[1]]*u[uOff[1]]);
  g0[0] += (gamma/rho)*(u[uOff[2]]*u_x[uOff_x[2]])/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))); // from first term
  g0[0] += (gamma/rho)*(2.0*u[uOff[2]])/(u[uOff[1]]*u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(1+u[uOff[2]]*u[uOff[2]])); // from second term
}

static void g1_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu = PetscRealPart(constants[0]);

  g1[0] = -(6*nu/u[uOff[1]])*u_x[uOff_x[0]];
}

static void g0_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g0[0]  = -(gamma/rho)*((u_x[uOff_x[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))) - (3.0*u[uOff[2]]*u[uOff[2]]*u_x[uOff_x[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)))); // from first term
  g0[0] -= (gamma/rho)*((1.0)/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(1+u[uOff[2]]*u[uOff[2]])) - (u[uOff[2]]*u[uOff[2]])/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)))); // from second term
}

static void g1_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g1[0] = -(gamma/rho)*((u[uOff[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))));
}

static void g2_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g2[0] = -(gamma/rho)*(3.0*u[uOff[2]]*u_x[uOff_x[2]])/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)));
}

static void g3_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal rho = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g3[0] = (gamma/rho)*(1.0)/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)));
}

static void g0_bd_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho = PetscRealPart(constants[1]);

  g0[0] = (gamma/rho)*(3.0*u[uOff[2]]*u_x[uOff_x[2]])/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)));
}
static void g1_bd_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho = PetscRealPart(constants[1]);

  g1[0] = -(gamma/rho)*(1)/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)));;
}

static void g0_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift + 0.5*u_x[uOff_x[0]];
}

static void g1_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  g1[0] = u[uOff[0]];
}

static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_x[uOff_x[1]];
}

static void g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  g1[0] = 0.5*u[uOff[1]];
}

/* Jacobians for stabilization terms */
static void g2_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[6];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*(u_tShift + 0.5*u_x[uOff_x[0]]);
}

static void g3_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[6];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*u[uOff[0]];
}

static void g2_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[6];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*u_x[uOff_x[1]];
}

static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal lambda;
  const PetscReal h = constants[6];

  lambda = (1.0)*h/2.0;
  g1[0] = lambda*0.5*u[uOff[1]];
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
  options->cells[0] = 50;
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "droplet.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(PetscBag bag, AppCtx *user)
{
  Parameter     *param;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagGetData(bag, (void **) &param);CHKERRQ(ierr);
  ierr = PetscBagSetName(bag, "par", "Droplet parameters");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->nu,    1.0e-6,    "nu",         "Kinematic viscosity(m^2/sec)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->rho,   997.0,     "rho",        "Fluid density(kg/m^3)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->gamma, 0.0728,    "gamma",      "Coefficient of surface tension(kg/sec^2)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->g,     9.81,      "g",          "Gravitational force per unit mass(m/sec^2)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->u_0,   1.0,       "u_0",        "Inlet velocity(m/s)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->h_0,   0.0026,    "h_0",        "Inlet radius(m)");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->fs,    0.0,       "fs",         "Shear force per unit area");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->cellsize, 0.0,    "cellsize",   "Cell size");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &param->x_p, 0.0,    "x_p",   "Cell size");CHKERRQ(ierr);
  ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr);
  user->necking = PETSC_FALSE;
  user->Bool = PETSC_TRUE;
  user->factor = 1.0;
  user->l = param->h_0;
  /* Assuming a hemisphere */
  user->V_old = 0.5 * (4.*PETSC_PI/3.) * PetscSqr(param->h_0)*(user->l);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, 1, PETSC_FALSE, user->cells, NULL, &user->l, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  PetscInt       id;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  /* Residual terms */
  ierr = PetscDSSetResidual(ds, 0, f0_q, f1_q);CHKERRQ(ierr);
  ierr = PetscDSSetBdResidual(ds, 0, f0_bd_q, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 1, f0_v, f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 2, f0_w, NULL);CHKERRQ(ierr);

  /* Jacobian terms without SUPG */
  ierr = PetscDSSetBdJacobian(ds, 0, 0, NULL, g1_bd_qu, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, g0_qu, g1_qu, NULL,  g3_qu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 1, g0_qh, g1_qh, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 2, g0_qs, g1_qs, g2_qs, g3_qs);CHKERRQ(ierr);
  ierr = PetscDSSetBdJacobian(ds, 0, 2, g0_bd_qs, g1_bd_qs, NULL,  NULL);CHKERRQ(ierr);

  ierr = PetscDSSetJacobian(ds, 1, 0, g0_vu, g1_vu, g2_vu, g3_vu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 1, 1, g0_vh, g1_vh, g2_vh, g3_vh);CHKERRQ(ierr);

  ierr = PetscDSSetJacobian(ds, 2, 1, NULL,  g1_wh, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 2, 2, g0_ws, NULL,  NULL,  NULL);CHKERRQ(ierr);

  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[9];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
    constants[0] = param->nu;
    constants[1] = param->rho;
    constants[2] = param->gamma;
    constants[3] = param->g;
    constants[4] = param->u_0;
    constants[5] = param->h_0;
    constants[6] = param->cellsize;
    constants[7] = param->fs;
    constants[8] = param->x_p;
    ierr = PetscDSSetConstants(ds, 9, constants);CHKERRQ(ierr);
  }

  /* Setup Boundary Conditions */
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  id = 1;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Inlet velocity",  label, 1, &id, 0, 0, NULL, (void (*)(void)) Inlet_u, (void (*)(void)) Inlet_u_t, user, &(user->bd_in));CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Inlet radius",    label, 1, &id, 1, 0, NULL, (void (*)(void)) Inlet_h, (void (*)(void)) Inlet_h_t, user, NULL);CHKERRQ(ierr);
  id = 2;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Bottom radius",   label, 1, &id, 1, 0, NULL, (void (*)(void)) Bottom_h, (void (*)(void)) Bottom_h_t, user, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscFE        fe[3];
  MPI_Comm       comm;
  PetscInt       dim;
  PetscBool      simplex = PETSC_FALSE;
  PetscErrorCode ierr;

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
  /* Locate point in original mesh */
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
  /* Create geometry */
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

static PetscErrorCode TSAdaptChoose_Volume(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea, PetscReal *wlter)
{
  DM             dm;
  AppCtx        *user;
  Parameter     *param;
  Vec            u;
  PetscDS        ds;
  DMLabel        label;
  PetscReal      time, dt;
  PetscScalar    integral[3], Vnew=0.0, V_p=0.0, rerr, rtol = 1.e-3;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
  /* Calculate Volume */
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(ds, 1, volume);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, integral, user);CHKERRQ(ierr);
  if(param->x_p) {V_p = integral[1]; printf("x_p = %g\n",param->x_p );}
  else Vnew = integral[1];

  rerr = (user->V_t - Vnew)/user->V_t;
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);

/* Check if new volume is close to the target volume. */
  if (PetscAbsReal(rerr) <= rtol || V_p) {
    *accept = PETSC_TRUE;
    if(V_p) {ierr = PetscPrintf(PETSC_COMM_WORLD, "\n \n ###### \n \t\t The droplt pinched off. \n"
     "\t The pinch-off volume = %g (%2.2f %% of total volume) \t The pinch-off time = %g \n ###### \n \n", V_p, 100.0*V_p/user->V_t, time);}
    else ierr = PetscPrintf(PETSC_COMM_WORLD, "V_target = %g  V_new = %g  V_lost = %g%%  tip velocity = %2.10f  Predicted length = %2.10f\n\n", user->V_t, Vnew, rerr*100., user->dl_dt, user->l);CHKERRQ(ierr);
  } else {
    *accept = PETSC_FALSE;

    DM               dmNew;
    Vec              U, coordinates;
    PetscErrorCode (*feFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
    void            *fectxs[3];
    PetscPointFunc   funcs[3] = {id_v, id_r, id_s};
    PetscScalar      l_upper, l_lower, stretch; /* Factor by which we increase the length */

    /* Correct the length and boundary condition using bisection. Then scale and project again using corrected length */
    stretch      = user->V_t / Vnew;
    if (stretch<1.0) {
      l_upper = user->l;
      l_lower = (user->l)*stretch;
      user->dl_dt /= stretch;
      user->s_bottom /= stretch;
    }
    else {
      l_upper = (user->l)*stretch;
      l_lower = user->l;
      user->dl_dt *= stretch;
      user->s_bottom *= stretch;
    }
    user->l = (l_lower + l_upper)/2.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCORRECTION due to V_lost = %g%% (V_target = %g  V_new = %g) \t update factor = %g  \t  Corrected length = %2.10f in [%g, %g]\n\n", rerr*100., user->V_t, Vnew, stretch,  user->l, l_lower, l_upper);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMClone(dm, &dmNew);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, dmNew);CHKERRQ(ierr);
    ierr = VecScale(coordinates, stretch);CHKERRQ(ierr);
    ierr = DMSetCoordinates(dmNew, coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dmNew, "Stretched Mesh");CHKERRQ(ierr);

#if 1
    feFuncs[0] = FieldFunction_u;
    feFuncs[1] = FieldFunction_h;
    feFuncs[2] = FieldFunction_s;
    fectxs[0]  = (void *) u;
    fectxs[1]  = (void *) u;
    fectxs[2]  = (void *) u;
    ierr = DMCreateGlobalVector(dmNew, &U);CHKERRQ(ierr);
    ierr = DMProjectFunction(dmNew, time, feFuncs, fectxs, INSERT_ALL_VALUES, U);CHKERRQ(ierr);
#else
    // TODO the problem here is that it will fail the enclosing query, and we will need to point locate all the dual quad points
    ierr = DMProjectFieldLocal(dmNew, time, u, funcs, INSERT_ALL_VALUES, U);CHKERRQ(ierr);
#endif

    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dmNew);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, U);
    ierr = VecDestroy(&U);CHKERRQ(ierr);
  }

  *next_h  = h;  /* Reuse the same time step */
  *next_sc = 0;  /* Reuse the same order scheme */
  *wlte    = -1; /* Weighted local truncation error was not evaluated */
  *wltea   = -1; /* Weighted absolute local truncation error was not evaluated */
  *wlter   = -1; /* Weighted relative local truncation error was not evaluated */
  PetscFunctionReturn(0);
}

/* Calculate the target volume and predicted length before a time step using previous successful step. Then project on a new mesh. */
static PetscErrorCode PreStep(TS ts)
{
  DM             dm, dmNew;
  PetscReal      dt, time, Flow_in;
  AppCtx        *user;
  Parameter     *param;
  PetscInt       stepi, N;
  Vec            u, coordinates, U;
  PetscDS        ds;
  PetscScalar    stretch; /* Factor by which we update the length */
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);

  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &N);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

/* Test for necking */
  PetscScalar scale = 3.0;
  if((user->l > scale*param->h_0)) {
    const PetscScalar *v;
    PetscScalar  *ptr=NULL;
    PetscInt     n, pStart, pEnd, Nf;
    PetscSection s;

    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u, &v);
    for (n=pStart;n<(PetscInt) PetscFloorReal(4*N/5);n++){
        PetscInt dof, cdof, d;

        ierr = PetscSectionGetFieldDof(s, n, 1, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(s, n, 1, &cdof);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, n, 1, v, &ptr);
        for (d = 0; d < dof-cdof; ++d) {
          if (!user->necking) {
            PetscScalar h_neck = 0.8*param->h_0;
            PetscScalar h_blow = 2.0*param->h_0;
            if (ptr[d]<h_neck){
              user->necking = PETSC_TRUE;
              ierr = PetscPrintf(PETSC_COMM_WORLD, "\n \n \n ##### \t\t The necking begins \t\t ###### \n \n \n");
            }
            if(!user->necking && ptr[d]>h_blow){
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "The curvature is blowing up in the middle. Something is wrong.");
            }
          }
          /* How close we want to get to the pinch-off? (Depends on the mesh) */
          if(ptr[d]<(0.1)*param->h_0){
            const PetscScalar  *Consts, *z;
            PetscInt           NC;

            ierr = PetscDSGetConstants(ds, &NC, &Consts);CHKERRQ(ierr);
            PetscScalar       *c = (PetscScalar*) Consts;
            ierr = VecGetArrayRead(coordinates, &z);CHKERRQ(ierr);
            param->x_p = z[n];
            *(c+8) = param->x_p;
            ierr = VecRestoreArrayRead(coordinates, &z);CHKERRQ(ierr);
            ierr = PetscDSSetConstants(ds, NC, c);CHKERRQ(ierr);
            ierr = TSSetMaxSteps(ts, stepi);CHKERRQ(ierr);
          }
        }
    }
    if((user->l > (scale+1)*param->h_0) && !user->necking) SETERRQ( PETSC_COMM_SELF, PETSC_ERR_SUP, "Necking did not happen. Something is wrong.");
    ierr = VecRestoreArrayRead(u, &v);
  }

  Flow_in = PETSC_PI*PetscSqr(param->h_0)*param->u_0*dt;
  user->V_t   = user->V_old + Flow_in;
  user->V_old = user->V_t;
  {
    const PetscScalar  *Consts;
    PetscInt            NC;
    PetscInt            cEnd;
    const PetscScalar  *v;
    const PetscScalar  *ptr_u = NULL;
    const PetscScalar  *ptr_s = NULL;

    /* Update length using u_tip */
    ierr = VecGetArrayRead(u, &v);CHKERRQ(ierr);
    ierr = DMPlexGetSimplexOrBoxCells(dm,0,NULL,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalFieldRead(dm,2*cEnd,0, v, &ptr_u);
    ierr = DMPlexPointGlobalFieldRead(dm,2*cEnd,2, v, &ptr_s);
    stretch = (user->l + dt*(*ptr_u)) / (user->l);
    user->dl_dt = (*ptr_u)*stretch;
    user->s_bottom = *ptr_s/(stretch);
    user->l += dt*(*ptr_u);
    ierr = VecRestoreArrayRead(u, &v);CHKERRQ(ierr);

    /* Change cell size for SUPG */
    ierr = PetscDSGetConstants(ds, &NC, &Consts);CHKERRQ(ierr);
    PetscScalar       *c = (PetscScalar*) Consts;
    param->cellsize = user->l/(N-1);
    *(c+6) = 1.0*param->cellsize;

    /*
    For low viscosity fluid, set the viscosity to higher value and gradually reduce it.
    This is to smooth out the initial surface instabilities.
    */
    if (param->nu < 1e-4){
      PetscReal X = (user->l)-(param->h_0), n=02.0;
      if((user->l)<(n*(param->h_0))) {
        *(c+0) = -((param->nu*9)/(n*(param->h_0)))*X + param->nu*10; /* linear approach towards the true viscosity */
      }
      else{
        *(c+0) = param->nu;
      }
    }
    /* If reducing the timestep is required */
    // if (user->necking && user->Bool) {ierr = TSSetTimeStep(ts, 0.1*dt); user->Bool = PETSC_FALSE;}

    ierr = PetscDSSetConstants(ds, NC, c);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "N = %d \t nu = %g \t Tip Velocity = %g \t update factor = %g \t Predicted length = %g \n", N, *(c+0), user->dl_dt, stretch, user->l);
  }

  ierr = DMClone(dm, &dmNew);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, dmNew);CHKERRQ(ierr);

  ierr = VecScale(coordinates, stretch);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dmNew, coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmNew, "New Mesh");CHKERRQ(ierr);

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

  ierr = TSReset(ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dmNew);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorSolAndCoords(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            coordinates;
  char           coords_name[PETSC_MAX_PATH_LEN];
  char           sol_name[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscSNPrintf(coords_name, PETSC_MAX_PATH_LEN, "Length_%0.0f.out", user->factor);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, coords_name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, coords_name, &viewer);CHKERRQ(ierr);
  ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
  {
    ierr = PetscSNPrintf(sol_name, PETSC_MAX_PATH_LEN, "Numerical_Solution_%0.0f", user->factor);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, sol_name);CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm;   /* problem definition */
  TS              ts;   /* timestepper */
  Vec             u;    /* solution */
  AppCtx          user; /* user-defined work context */
  PetscReal       t;
  PetscBool       monitor_off=PETSC_FALSE; /* Determine if we need monitor.*/
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
  ierr = SetupParameters(user.bag, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  /* Setup problem */
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);

/* Rearrange the mesh coordinates to have a dense mesh. */
  {
    Vec        c, d, d1, d2, e;
    PetscReal  a=0;/* Do not change */
    PetscReal  b=1.1;
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
    ierr = VecScale(d1, (2*a+1)/(param->h_0));
    ierr = VecScale(d2, -(2*a+1)/(param->h_0));
    ierr = VecAXPY(d1, (b-2*a), e);
    ierr = VecAXPY(d2, (b+2*a), e);
    ierr = VecPointwiseDivide(d, d1, d2);
    ierr = VecLog(d);
    ierr = VecAXPBY(d, a, ((param->h_0 - a)/(PetscLogReal((b+1)/(b-1)))), e);

// Dense in the middle
    ierr = VecDestroy(&d1);CHKERRQ(ierr);
    ierr = VecDestroy(&d2);CHKERRQ(ierr);
    ierr = VecDuplicate(d, &d1);
    ierr = VecDuplicate(d, &d2);
    ierr = VecCopy(d, d1);
    ierr = VecCopy(d, d2);

// Dense mesh at the bottom
    ierr = VecScale(d1, -1/(param->h_0));
    ierr = VecScale(d2, 1/(param->h_0));
    ierr = VecAXPY(d1, (b+1), e);
    ierr = VecAXPY(d2, (b-1), e);
    ierr = VecPointwiseDivide(d, d1, d2);
    ierr = VecLog(d);
    ierr = VecAXPBY(d, param->h_0, -((param->h_0)/(PetscLogReal((b+1)/(b-1)))), e);

    ierr = DMSetCoordinates(dm, d);CHKERRQ(ierr);

    ierr = VecDestroy(&d1);CHKERRQ(ierr);
    ierr = VecDestroy(&d2);CHKERRQ(ierr);
    ierr = VecDestroy(&e);CHKERRQ(ierr);
    ierr = VecDestroy(&d);CHKERRQ(ierr);
  }

  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM , &user);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-monitor_off",&monitor_off);CHKERRQ(ierr);

  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
  ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  if(!monitor_off) ierr = TSMonitorSet(ts, MonitorSolAndCoords, &user, NULL);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = TSSetPreStep(ts, PreStep);CHKERRQ(ierr);
  {
    TSAdapt   adapt;
    ierr = TSGetAdapt(ts, &adapt);CHKERRQ(ierr);
    adapt->ops->choose = TSAdaptChoose_Volume;
  }
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, NULL);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Paraffin wax
  test:
    suffix: paraffin_0
    args: -h_0 0.0026 -u_0 0.005 -nu 0.00005 -rho 760 -gamma 0.035 \
          -cells 100 -dm_plex_separate_marker \
          -vel_petscspace_degree 3 -rad_petscspace_degree 3 -slope_petscspace_degree 2 \
          -ts_max_steps 10000 -ts_dt 1e-4 -ts_type beuler -ts_max_reject 20 -ts_monitor \
            -snes_converged_reason -snes_max_funcs 1000000  -snes_monitor \
              -ksp_gmres_restart 500 -ksp_error_if_not_converged -ksp_converged_reason -ksp_monitor_true_residual \
              -pc_type lu
              -monitor_off
TEST*/
