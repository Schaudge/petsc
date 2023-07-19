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
  PetscReal   nu_d;            /* Kinematic viscosity */
  PetscReal   rho_d;           /* Fluid density */
  PetscReal   nu_c;            /* Kinematic viscosity */
  PetscReal   rho_c;           /* Fluid density */
  PetscReal   gamma;         /* Coefficient of the surface tension. */
  PetscReal   g;             /* Gravitational force per unit mass */
  PetscReal   u_0;           /* Inlet velocity */
  PetscReal   h_0;           /* Inlet radius */
  PetscReal   cellsize;        /* Cellsize */
  PetscReal   R;            /* Shear velocity / Continuous phase velocity  */
  PetscReal   D;            /* Continuous phase pressure drop  */
  PetscReal   C;            /* R = C*h  */
  PetscReal   x_p;
} Parameter;

typedef struct {
  PetscBag  bag;      /* Holds problem parameters */
  PetscReal V_old;    /* Starting drop volume */
  PetscReal V_t;      /* Target drop volume */
  PetscReal l;        /* Droplet length */
  PetscReal dl_dt;    /* Droplet length time derivative OR bottom velocity */
  PetscReal s_bottom; /* Bottom s (or dh_dz) for projection */
  PetscReal factor;   /* For adaptivity */
  PetscReal N;
  PetscReal G;        /* The mass flux (kg/m^2 s) */
  PetscInt  cells[1]; /* Initial mesh division */
  PetscBool necking;  /* Necking test */
  PetscBool Bool;
  PetscBool dtRefine;
  PetscBool adapt;
  PetscBool PinchOffRefine;
  DMLabel   adaptLabel;
  PetscInt  N_adapts;
  Vec       locX;
} AppCtx;

/* Initial conditions */
static PetscErrorCode Initial_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;

  PetscCall(PetscBagGetData(user->bag, (void **) &param));

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

  PetscCall(PetscBagGetData(user->bag, (void **) &param));

  PetscScalar h_0 = param->h_0;
 	PetscScalar l_0 = user->l;
  // u[0] = -(x[0]*x[0])/(PetscCbrtReal((h_0*h_0*h_0 - x[0]*x[0]*x[0])*(h_0*h_0*h_0 - x[0]*x[0]*x[0]))); /* cubic curve derivative */
  if(x[0]<l_0) u[0] = -h_0*x[0]/(l_0*PetscSqrtReal(l_0*l_0 - x[0]*x[0])); /* hemi ellipse curve derivative */
  else u[0] = -10;
  return 0;
}

/* Boundary conditions */
static PetscErrorCode Inlet_h(PetscInt Dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  Parameter     *param;

  PetscCall(PetscBagGetData(user->bag, (void **) &param));
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

  PetscCall(PetscBagGetData(user->bag, (void **) &param));

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

static void u_radial(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = - u_x[uOff_x[0]]*u[uOff[1]]/2.0;
}
static void surface_energy(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal gamma = PetscRealPart(constants[2]);

  f0[0] = 2.0*PETSC_PI*gamma*u[uOff[1]]*PetscSqrtReal(1.0 + PetscSqr(u[uOff[2]]));
}

static void Curvature(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal h = u[uOff[1]];
  PetscReal sx = u_x[uOff_x[2]];
  PetscReal denom = 1 + u[uOff[2]]*u[uOff[2]];

  f0[0] = (1/(h*PetscSqrtReal(denom))) + (sx/(PetscSqrtReal(PetscPowReal(denom,3))));
}

static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal x_p =  PetscRealPart(constants[8]);
  if(x[0]<x_p) {f0[0] = 0.0;}
  else {f0[0] = PETSC_PI*PetscSqr(u[uOff[1]]);}
}
static void area(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal x_p =  PetscRealPart(constants[8]);
  if(x[0]<x_p) {f0[0] = 0.0;}
  else {f0[0] = 2.0*PETSC_PI*u[uOff[1]]*PetscSqrtReal(1.0 + PetscSqr(u[uOff[2]]));}
}
/*
Residual functions.
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{

  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal D = PetscRealPart(constants[11]);
  const PetscReal C = PetscRealPart(constants[12]);
  const PetscReal g = PetscRealPart(constants[3]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  PetscScalar  dpdz;

  dpdz = curvature0_q(u[uOff[1]], u[uOff[2]], u_x[uOff_x[2]]);
  f0[0] =  u_t[uOff[0]] + u[uOff[0]]*u_x[uOff_x[0]] + gamma*dpdz/rho_d;
  f0[0] += - (u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]])*6.0*nu_d*(1.0 + mu_ratio);
  // f0[0] += - (u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]])*6.0*nu_d*(1.0 + mu_ratio/(1.0 + PetscSqr(u[uOff[2]])));
  f0[0] += D/(2.0*rho_d*PetscLogReal(C));
  f0[0] += 2.0*D/rho_d;
  f0[0] += - (1.0 - rho_c/rho_d)*g;
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal dpdz = curvature1_q(u[uOff[2]], u_x[uOff_x[2]]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);
  lambda = (1.0)*h/2.0;
  f1[0] =  (3.0*nu_d + 2.0*(nu_c*rho_c/rho_d))*u_x[uOff_x[0]] + gamma*dpdz/rho_d;
  // f1[0] =  (3.0*nu_d)*u_x[uOff_x[0]] + gamma*dpdz/rho_d;
  f1[0] += lambda*(u[uOff[0]] - (6.0*nu_d*(1.0 + mu_ratio))*u_x[uOff_x[1]]/u[uOff[1]] )*u_x[uOff_x[0]];
}

static void f0_bd_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal dpdz_bd = curvature1_q(u[uOff[2]], u_x[uOff_x[2]]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);
  lambda = (1.0)*h/2.0;
  f0[0] =  - (3.0*nu_d + 2.0*(nu_c*rho_c/rho_d))*u_x[uOff_x[0]] - gamma*dpdz_bd/rho_d;
  // f0[0] =  - (3.0*nu_d)*u_x[uOff_x[0]] - gamma*dpdz_bd/rho_d;

  f0[0] += - lambda*(u[uOff[0]] - ( 6.0*nu_d*(1.0 + mu_ratio) )*u_x[uOff_x[1]]/u[uOff[1]])*u_x[uOff_x[0]];
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
  lambda = (1.0)*h/2.0;
  // f1[0] = lambda*(u[uOff[0]]*u_x[uOff_x[1]]);
  f1[0] = lambda*(u[uOff[0]]*u_x[uOff_x[1]] + 0.5*u[uOff[1]]*u_x[uOff_x[0]]);
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
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  g1[0] = u[uOff[0]] - 6.0*nu_d*(1.0 + mu_ratio)*u_x[uOff_x[1]]/u[uOff[1]];
  // g1[0] = u[uOff[0]] - 6.0*nu_d*(1.0 + mu_ratio/(1.0 + PetscSqr(u[uOff[2]])))*u_x[uOff_x[1]]/u[uOff[1]];
}

static void g3_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);
  lambda = (1.0)*h/2.0;

  // g3[0] = (3.0*nu_d);
  g3[0] = (3.0*nu_d + 2.0*(nu_c*rho_c/rho_d));
  g3[0] += lambda*(u[uOff[0]] - (6.0*nu_d*(1.0 + mu_ratio))*u_x[uOff_x[1]]/u[uOff[1]]);
}
static void g1_bd_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);
  lambda = (1.0)*h/2.0;

  // g1[0] = - (3.0*nu_d);
  g1[0] = - (3.0*nu_d+ 2.0*(nu_c*rho_c/rho_d));
  g1[0] += - lambda*(u[uOff[0]] - (6.0*nu_d*(1.0 + mu_ratio))*u_x[uOff_x[1]]/u[uOff[1]]);
}
/* stabilization Jacobians */
static void g2_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal lambda = (1.0)*h/2.0;

  g2[0] = lambda*u_x[uOff_x[0]];
}
static void g0_bd_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal lambda = (1.0)*h/2.0;

  g0[0] = -lambda*u_x[uOff_x[0]];
}

static void g2_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);
  PetscReal lambda = (1.0)*h/2.0;

  g2[0] = lambda*(6.0*nu_d)*u_x[uOff_x[1]]*u_x[uOff_x[0]]/(u[uOff[1]]*u[uOff[1]]);
}
static void g0_bd_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);
  PetscReal lambda = (1.0)*h/2.0;

  g0[0] = -lambda*(6.0*nu_d)*u_x[uOff_x[1]]*u_x[uOff_x[0]]/(u[uOff[1]]*u[uOff[1]]);
}
static void g3_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);
  PetscReal lambda = (1.0)*h/2.0;

  g3[0] = -lambda*(6.0*nu_d)*u_x[uOff_x[0]]/u[uOff[1]];
}
static void g1_bd_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal h = PetscRealPart(constants[6]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);
  PetscReal lambda = (1.0)*h/2.0;

  g1[0] = lambda*(6.0*nu_d)*u_x[uOff_x[0]]/u[uOff[1]];
}
/*********************/
static void g0_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);


  g0[0]  = 6.0*nu_d*( 1.0 + mu_ratio )*u_x[uOff_x[0]]*u_x[uOff_x[1]]/(u[uOff[1]]*u[uOff[1]]);
  // g0[0]  = 6.0*nu_d*( 1.0 + mu_ratio/(1.0 + PetscSqr(u[uOff[2]])) )*u_x[uOff_x[0]]*u_x[uOff_x[1]]/(u[uOff[1]]*u[uOff[1]]);
  g0[0] += (gamma/rho_d)*(u[uOff[2]]*u_x[uOff_x[2]])/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))); // from first term
  g0[0] += (gamma/rho_d)*(2.0*u[uOff[2]])/(u[uOff[1]]*u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(1+u[uOff[2]]*u[uOff[2]])); // from second term
}

static void g1_qh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  g1[0] = - 6.0*nu_d*( 1.0 + mu_ratio )*u_x[uOff_x[0]]/u[uOff[1]];
  // g1[0] = - 6.0*nu_d*( 1.0 + mu_ratio/(1.0 + PetscSqr(u[uOff[2]])) )*u_x[uOff_x[0]]/u[uOff[1]];
}

static void g0_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal nu_d = PetscRealPart(constants[0]);
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal nu_c = PetscRealPart(constants[9]);
  const PetscReal rho_c = PetscRealPart(constants[10]);
  const PetscReal gamma = PetscRealPart(constants[2]);
  PetscReal mu_ratio = nu_c*rho_c/(nu_d*rho_d);

  g0[0] = -(gamma/rho_d)*((u_x[uOff_x[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))) - (3.0*u[uOff[2]]*u[uOff[2]]*u_x[uOff_x[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)))); // from first term
  g0[0] -= (gamma/rho_d)*((1.0)/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(1+u[uOff[2]]*u[uOff[2]])) - (u[uOff[2]]*u[uOff[2]])/(u[uOff[1]]*u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)))); // from second term
  // g0[0] += (u_x[uOff_x[1]]*u_x[uOff_x[0]]/u[uOff[1]])*6.0*nu_d*(2.0*u[uOff[2]]*mu_ratio/PetscSqr(1.0 + PetscSqr(u[uOff[2]])));
}

static void g1_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g1[0] = -(gamma/rho_d)*((u[uOff[2]])/(u[uOff[1]]*PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3))));
}

static void g2_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g2[0] = -(gamma/rho_d)*(3.0*u[uOff[2]]*u_x[uOff_x[2]])/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)));
}

static void g3_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal rho_d = PetscRealPart(constants[1]);
  const PetscReal gamma = PetscRealPart(constants[2]);

  g3[0] = (gamma/rho_d)*(1.0)/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)));
}

static void g0_bd_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho_d = PetscRealPart(constants[1]);

  g0[0] = (gamma/rho_d)*(3.0*u[uOff[2]]*u_x[uOff_x[2]])/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),5)));
}
static void g1_bd_qs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscReal gamma = PetscRealPart(constants[2]);
  const PetscReal rho_d = PetscRealPart(constants[1]);

  g1[0] = -(gamma/rho_d)*(1)/(PetscSqrtReal(PetscPowReal((1+u[uOff[2]]*u[uOff[2]]),3)));;
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

static void g3_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);

  lambda = (1.0)*h/2.0;
  g3[0] = lambda*u[uOff[0]];
}

static void g2_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
    PetscReal lambda;
    const PetscReal h = PetscRealPart(constants[6]);

    lambda = (1.0)*h/2.0;
    g2[0] = lambda*u_x[uOff_x[1]];
}

static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal lambda;
  const PetscReal h = PetscRealPart(constants[6]);

  lambda = (1.0)*h/2.0;
  g3[0] = lambda*0.5*u[uOff[1]];
}

static void g2_vh(PetscInt dim, PetscInt Nf, PetscInt NfAux,
  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
    PetscReal lambda;
    const PetscReal h = PetscRealPart(constants[6]);

    lambda = (1.0)*h/2.0;
    g2[0] = lambda*0.5*u_x[uOff_x[0]];
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

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "1D Droplet Problem Options", "DMPLEX");
  options->cells[0] = 100;
  options->G = 0.0;
  PetscCall(PetscOptionsIntArray("-cells", "The initial mesh division", "droplet.c", options->cells, &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-G", &(options->G), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(PetscBag bag, AppCtx *user)
{
  Parameter     *param;
  PetscReal     Ca_c, Oh_d, mu_ratio;

  PetscFunctionBeginUser;
  PetscCall(PetscBagGetData(bag, (void **) &param));
  PetscCall(PetscBagSetName(bag, "par", "Droplet parameters"));
  PetscCall(PetscBagRegisterReal(bag, &param->nu_d,     1.0e-6,                                      "nu_d",       "Kinematic viscosity of diffuse phase (m^2/sec)"));
  PetscCall(PetscBagRegisterReal(bag, &param->rho_d,    997.0,                                       "rho_d",      "Diffuse phase density(kg/m^3)"));
  PetscCall(PetscBagRegisterReal(bag, &param->nu_c,     0.0,                                         "nu_c",       "Kinematic viscosity of continuous phase (m^2/sec)"));
  PetscCall(PetscBagRegisterReal(bag, &param->rho_c,    0.0,                                         "rho_c",      "Continuous phase density(kg/m^3)"));
  PetscCall(PetscBagRegisterReal(bag, &param->gamma,    0.0728,                                      "gamma",      "Coefficient of surface tension(kg/sec^2)"));
  PetscCall(PetscBagRegisterReal(bag, &param->g,        9.81,                                        "gr",          "Gravitational force per unit mass(m/sec^2)"));
  PetscCall(PetscBagRegisterReal(bag, &param->u_0,      0.01,                                        "u_0",        "Inlet velocity(m/s)"));
  PetscCall(PetscBagRegisterReal(bag, &param->h_0,      0.002,                                       "h_0",        "Inlet radius(m)"));
  PetscCall(PetscBagRegisterReal(bag, &param->R,        0.10,                                        "R",          "continuous phase velocity"));
  PetscCall(PetscBagRegisterReal(bag, &param->D,      - 8.0*param->nu_c*user->G/(param->R*param->R), "D",          "continuous phase pressure drop"));
  PetscCall(PetscBagRegisterReal(bag, &param->C,        0.0,                                        "C",          "C"));
  PetscCall(PetscBagRegisterReal(bag, &param->x_p,      0.0,                                         "x_p",        "Location of pinch-off"));
  PetscCall(PetscBagRegisterReal(bag, &param->cellsize, 0.0,                                         "cellsize",   "Cell size"));
  PetscCall(PetscBagSetFromOptions(bag));
  mu_ratio = (param->nu_c*param->rho_c)/(param->nu_d*param->rho_d);
  Oh_d     = (param->nu_d*param->rho_d)/PetscSqrtReal(param->rho_d*param->h_0*param->gamma);
  Ca_c     = (param->nu_c*user->G)/param->gamma;

  if(!param->C) param->C = 1.0 + (0.45*PetscTanhReal(2.5*Oh_d - 2.0) + 0.45)*( 20.0*PetscExpReal(-45.0*Ca_c*PetscPowReal(mu_ratio, -0.6)) + 0.045);

  user->necking = PETSC_FALSE;
  user->Bool = PETSC_FALSE;
  user->adapt = PETSC_FALSE;
  user->factor = 2.0;
  user->l = param->h_0;
  user->N = 0.9;
  // user->N = 3.0;
  user->N_adapts = 0;
  /* Assuming a hemisphere */
  user->V_old = 0.5 * (4.*PETSC_PI/3.) * PetscSqr(param->h_0)*(user->l);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateBoxMesh(comm, 1, PETSC_FALSE, user->cells, NULL, &user->l, NULL, PETSC_TRUE, dm));
  PetscCall(PetscObjectSetName((PetscObject) *dm, "Mesh"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  PetscInt       id;


  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  Parameter  *param;
  PetscCall(PetscBagGetData(user->bag, (void **) &param));

  /* Residual terms */
  PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
  PetscCall(PetscDSSetBdResidual(ds, 0, f0_bd_q, NULL));
  PetscCall(PetscDSSetResidual(ds, 1, f0_v, f1_v));
  PetscCall(PetscDSSetResidual(ds, 2, f0_w, NULL));

  /* Jacobian terms without SUPG */

  PetscCall(PetscDSSetBdJacobian(ds, 0, 0, g0_bd_qu, g1_bd_qu, NULL,  NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qu, g1_qu, g2_qu, g3_qu));

  PetscCall(PetscDSSetJacobian(ds, 0, 1, g0_qh, g1_qh, g2_qh,  g3_qh));
  PetscCall(PetscDSSetBdJacobian(ds, 0, 1, g0_bd_qh, g1_bd_qh, NULL,  NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 2, g0_qs, g1_qs, g2_qs, g3_qs));
  PetscCall(PetscDSSetBdJacobian(ds, 0, 2, g0_bd_qs, g1_bd_qs, NULL,  NULL));

  // PetscCall(PetscDSSetJacobian(ds, 1, 0, g0_vu, g1_vu, g2_vu, NULL));
  // PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_vh, g1_vh, NULL, g3_vh));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, g0_vu, g1_vu, g2_vu, g3_vu));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_vh, g1_vh, g2_vh, g3_vh));

  PetscCall(PetscDSSetJacobian(ds, 2, 1, NULL,  g1_wh, NULL,  NULL));
  PetscCall(PetscDSSetJacobian(ds, 2, 2, g0_ws, NULL,  NULL,  NULL));

  /* Setup constants */
  {
    PetscScalar constants[13];

    constants[0] = param->nu_d;
    constants[1] = param->rho_d;
    constants[2] = param->gamma;
    constants[3] = param->g;
    constants[4] = param->u_0;
    constants[5] = param->h_0;
    constants[6] = param->cellsize;
    constants[7] = param->R;
    constants[8] = param->x_p;
    constants[9] = param->nu_c;
    constants[10] = param->rho_c;
    constants[11] = param->D;
    constants[12] = param->C;
    PetscCall(PetscDSSetConstants(ds, 13, constants));
  }

  /* Setup Boundary Conditions */
  PetscCall(DMGetLabel(dm, "marker", &label));
  id = 1;
  PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Inlet velocity",  label, 1, &id, 0, 0, NULL, (void (*)(void)) Inlet_u, (void (*)(void)) Inlet_u_t, user, NULL));
  PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Inlet radius",    label, 1, &id, 1, 0, NULL, (void (*)(void)) Inlet_h, (void (*)(void)) Inlet_h_t, user, NULL));
  id = 2;
  PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Bottom radius",   label, 1, &id, 1, 0, NULL, (void (*)(void)) Bottom_h, (void (*)(void)) Bottom_h_t, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscFE        fe[3];
  MPI_Comm       comm;
  PetscInt       dim;
  PetscBool      simplex = PETSC_FALSE;


  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  /* Create finite element */
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "vel_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject) fe[0], "velocity"));

  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "rad_", PETSC_DEFAULT, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject) fe[1], "radius"));

  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "slope_", PETSC_DEFAULT, &fe[2]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[2]));
  PetscCall(PetscObjectSetName((PetscObject) fe[2], "slope"));

  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  PetscCall(DMSetField(dm, 2, NULL, (PetscObject) fe[2]));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, user));

  PetscCall(PetscFEDestroy(&fe[0]));
  PetscCall(PetscFEDestroy(&fe[1]));
  PetscCall(PetscFEDestroy(&fe[2]));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM               dm;
  PetscReal        t;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  void            *ctxs[3];
  AppCtx          *ctx;


  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMGetApplicationContext(dm, &ctx));
  funcs[0] = Initial_u;
  funcs[1] = Initial_h;
  funcs[2] = Initial_s;
  ctxs[0] = ctx;
  ctxs[1] = ctx;
  ctxs[2] = ctx;
  PetscCall(DMProjectFunction(dm, t, funcs, ctxs, INSERT_ALL_VALUES, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode FieldFunction_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords;
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
  PetscReal          xRef[3];


  PetscFunctionBeginUser;
  /* Locate point in original mesh */
  PetscCall(VecGetDM(uV, &dm));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords));
  PetscCall(DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(VecDestroy(&coords));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem));
  PetscCall(DMGetDS(dm, &ds));
  AppCtx     *user = (AppCtx *) ctx;
  PetscCall(DMGetApplicationContext(dm, &user));

  if (N != 1) {
    Parameter  *param;

    PetscCall(PetscBagGetData(user->bag, (void **) &param));
    PetscCall(Bottom_u(dim, time, x, Nc, u, user));
    PetscFunctionReturn(0);
  }
  cell = cellsRem[0].index;
  PetscCall(PetscSFDestroy(&cellSF));
  /* Create geometry */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *) &fe));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom));
  PetscCall(ISDestroy(&cellIS));

  /* Interpolate field values */
  PetscCall(DMPlexVecGetClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
  PetscCall(DMPlexCoordinatesToReference(dm, cell, 1, x, xRef));
  PetscCall(PetscFECreateTabulation(fe, 1, 1, xRef, 0, &T));
  if (Nc != T->Nc) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %d != %d Number of field cmponents", Nc, T->Nc);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;

    for (fc = 0; fc < Nc; ++fc) {
      u[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        u[fc] += coeff[f]*basis[(0*Nb + f)*Nc + fc];
      }
    }
    // {
    //   Parameter  *param;
    //   PetscCall(PetscBagGetData(user->bag, (void **) &param));
    //   /* Refinement using u' instead of u */
    //   if(user->Bool && user->N_adapts>0 && PetscAbsReal(u[0])>(param->u_0*user->N)){
    //       PetscCall(DMLabelSetValue(user->adaptLabel, cell, DM_ADAPT_REFINE));
    //       user->adapt = PETSC_TRUE;
    //   }
    // }
  }
  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(PetscFEPushforward(fe, cgeom, 1, u));
  PetscCall(PetscFEGeomDestroy(&cgeom));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
  PetscFunctionReturn(0);
}
static PetscErrorCode FieldFunction_h(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords;
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
  PetscReal          xRef[3];



  PetscFunctionBeginUser;
  /* Locate point in original mesh */
  PetscCall(VecGetDM(uV, &dm));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords));
  PetscCall(DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(VecDestroy(&coords));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem));
  AppCtx     *user = (AppCtx *) ctx;
  PetscCall(DMGetApplicationContext(dm, &user));

  if (N != 1) {
    PetscCall(Bottom_h(dim, time, x, Nc, u, user));
    PetscFunctionReturn(0);
  }
  cell = cellsRem[0].index;
  PetscCall(PetscSFDestroy(&cellSF));

   /* Create geometry */
  PetscCall(DMGetDS(dm, &ds));

  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(PetscDSGetDiscretization(ds, 1, (PetscObject *) &fe));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom));
  PetscCall(ISDestroy(&cellIS));

  /* Get cell coefficients and Interpolate field values */
  PetscCall(DMPlexVecGetClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
  PetscCall(DMPlexCoordinatesToReference(dm, cell, 1, x, xRef));
  PetscCall(PetscFECreateTabulation(fe, 1, 1, xRef, 0, &T));
  if (Nc != T->Nc) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %d != %d Number of field cmponents", Nc, T->Nc);
  {
    const PetscReal *basis = T->T[0];
    const PetscInt   Nb    = T->Nb;

    for (fc = 0; fc < Nc; ++fc) {
      u[fc] = 0.0;
      for (f = 0; f < Nb; ++f) {
        u[fc] += coeff[(Ncl-2*Nb+1) + f]*basis[(0*Nb + f)*Nc + fc];
      }
    }
    {
      Parameter  *param;
      PetscCall(PetscBagGetData(user->bag, (void **) &param));

      if(x[0]<0.9*user->l && u[0]<(0.01*param->h_0)){
        param->x_p = x[0];
      }
      if(user->Bool){
        // if(user->N_adapts==0 && x[0]>(0.1*user->l)){
        //   PetscCall(DMLabelSetValue(user->adaptLabel, cell, DM_ADAPT_REFINE));
        //   user->adapt = PETSC_TRUE;
        // }
        // else if(user->N_adapts==1 && (( x[0]>(0.15*user->l) && x[0]<(0.5*user->l) ) ||  x[0]>(0.9*user->l)) ){
       if(user->N_adapts==0 && (( x[0]>(0.1*user->l) && x[0]<(0.75*user->l) ) ||  x[0]>(0.9*user->l)) ){
          PetscCall(DMLabelSetValue(user->adaptLabel, cell, DM_ADAPT_REFINE));
          user->adapt = PETSC_TRUE;
        }
        // else if(user->N_adapts==1 && x[0]>0.02*user->l && x[0]<0.5*user->l){
        //   PetscCall(DMLabelSetValue(user->adaptLabel, cell, DM_ADAPT_REFINE));
        // }
        else if(user->N_adapts>=1 && x[0]<(0.8*user->l) && PetscAbsReal(u[0])<(param->h_0*user->N) ) {
          PetscCall(DMLabelSetValue(user->adaptLabel, cell, DM_ADAPT_REFINE));
          user->adapt = PETSC_TRUE;
        }
      }
    }
  }
  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(PetscFEPushforward(fe, cgeom, 1, u));
  PetscCall(PetscFEGeomDestroy(&cgeom));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
  PetscFunctionReturn(0);
}
static PetscErrorCode FieldFunction_s(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  Vec                uV = (Vec) ctx, coords;
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
  PetscReal          xRef[3];

  PetscFunctionBeginUser;
  /* Locate point in original mesh */
  PetscCall(VecGetDM(uV, &dm));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, x, &coords));
  PetscCall(DMLocatePoints(dm, coords, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(VecDestroy(&coords));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &N, NULL, &cellsRem));
  AppCtx     *user = (AppCtx *) ctx;
  PetscCall(DMGetApplicationContext(dm, &user));

  if (N != 1) {
    // void     *ctx;

    // PetscCall(DMGetApplicationContext(dm, &ctx));
    PetscCall(Bottom_s(dim, time, x, Nc, u, user));
    PetscFunctionReturn(0);
  }
  cell = cellsRem[0].index;
  PetscCall(PetscSFDestroy(&cellSF));

   /* Create geometry */
  PetscCall(DMGetDS(dm, &ds));

  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(PetscDSGetDiscretization(ds, 2, (PetscObject *) &fe));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, &cell, PETSC_COPY_VALUES, &cellIS));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, q, PETSC_FALSE, &cgeom));
  PetscCall(ISDestroy(&cellIS));

  /* Get cell coefficients and Interpolate field values */
  PetscCall(DMPlexVecGetClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
  PetscCall(DMPlexCoordinatesToReference(dm, cell, 1, x, xRef));
  PetscCall(PetscFECreateTabulation(fe, 1, 1, xRef, 0, &T));
  if (Nc != T->Nc) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of input components %d != %d Number of field cmponents", Nc, T->Nc);
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
  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(PetscFEPushforward(fe, cgeom, 1, u));
  PetscCall(PetscFEGeomDestroy(&cgeom));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, user->locX, cell, &Ncl, &coeff));
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
  PetscScalar    integral[3], Vnew=0.0, A_new=0.0, V_d=0.0, A_d=0.0, rerr, rtol = 5.0e-4;


  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(PetscBagGetData(user->bag, (void **) &param));
  PetscCall(TSGetSolution(ts, &u));
  /* Calculate Volume */
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 1, volume));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, integral, user));
  if(param->x_p) {
    V_d = integral[1];
    printf("\n \nNon-dimensional pinch-off location = %g\n", param->x_p/user->l );
  }
  else Vnew = integral[1];

  PetscCall(PetscDSSetObjective(ds, 1, area));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, integral, user));
  if(param->x_p) {
    A_d = integral[1];
  }
  else A_new = integral[1];

  rerr = (user->V_t - Vnew)/user->V_t;
  PetscCall(DMGetLabel(dm, "marker", &label));

  if ((PetscAbsReal(rerr) <= rtol || V_d)) {
    /* Check if new volume is close to the target volume. */
    *accept = PETSC_TRUE;
    if(V_d)
    {
      PetscPrintf(PETSC_COMM_WORLD, "The droplet length = %g \nThe droplet surface area = %g \nThe droplet volume = %g (%2.2f %% of total volume) \nThe droplet radius = %g \nThe pinch-off time = %g s \n ######"
    "\n \n The droplt pinched off \n \n", user->l, A_d, V_d, 100.0*V_d/user->V_t, PetscCbrtReal(3.0*V_d/(4.0*PETSC_PI)), time);

    PetscPrintf(PETSC_COMM_WORLD, "%g, %g, %g, %g, %g, %g \n", param->x_p/user->l, A_d, V_d, PetscCbrtReal(3.0*V_d/(4.0*PETSC_PI)), time, user->l/param->h_0);
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD, "\n V_target = %g  V_new = %g  V_lost = %g%% \n Surface area = %g \n\n", user->V_t, Vnew, rerr*100., A_new);
    }
  }
  else {
    *accept = PETSC_FALSE;
    DM               dmNew;
    Vec              U, coordinates;
    PetscErrorCode (*feFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
    void            *fectxs[3];
    // PetscPointFunc   funcs[3] = {id_v, id_r, id_s};
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
    stretch = (l_lower + l_upper)/(2.0*user->l);
    user->l *= stretch;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nCORRECTION due to V_lost = %g%% (V_target = %g  V_new = %g) \t update factor = %g  \t  Corrected length = %2.10f in [%g, %g]\n\n", rerr*100., user->V_t, Vnew, stretch,  user->l, l_lower, l_upper));

    PetscCall(DMCreateLocalVector(dm, &(user->locX)));
    PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, user->locX));
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, user->locX, time, NULL, NULL, NULL));

    PetscCall(DMGetCoordinates(dm, &coordinates));
    PetscCall(DMClone(dm, &dmNew));
    PetscCall(DMCopyDisc(dm, dmNew));
    PetscCall(VecScale(coordinates, stretch));
    PetscCall(DMSetCoordinates(dmNew, coordinates));
    PetscCall(PetscObjectSetName((PetscObject) dmNew, "Stretched Mesh"));

// #if 1
    feFuncs[0] = FieldFunction_u;
    feFuncs[1] = FieldFunction_h;
    feFuncs[2] = FieldFunction_s;
    fectxs[0]  = (void *) u;
    fectxs[1]  = (void *) u;
    fectxs[2]  = (void *) u;
    PetscCall(DMCreateGlobalVector(dmNew, &U));
    PetscCall(DMProjectFunction(dmNew, time, feFuncs, fectxs, INSERT_ALL_VALUES, U));
// #else
    // TODO the problem here is that it will fail the enclosing query, and we will need to point locate all the dual quad points
    // PetscCall(DMProjectFieldLocal(dmNew, time, u, funcs, INSERT_ALL_VALUES, U));
// #endif

    PetscCall(DMDestroy(&dm));
    PetscCall(TSSetDM(ts, dmNew));
    PetscCall(TSSetSolution(ts, U));
    PetscCall(VecDestroy(&U));
    {
      DM_Plex       *mesh = (DM_Plex*) dmNew->data;
      PetscCall(PetscGridHashDestroy(&mesh->lbox));
    }
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


  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetStepNumber(ts, &stepi));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(PetscBagGetData(user->bag, (void **) &param));
  PetscCall(TSGetSolution(ts, &u));

  PetscCall(DMGetCoordinates(dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(DMGetDS(dm, &ds));

/* Test for necking */
  PetscScalar scale = 2.0;
  if((user->l > scale*param->h_0)) {
    const PetscScalar *v;
    PetscScalar  *ptr=NULL;
    PetscInt     n;
    PetscSection s;
    PetscCall(DMGetLocalSection(dm, &s));
    PetscCall(VecGetArrayRead(u, &v));
    for (n=0;n<(PetscInt) PetscFloorReal(7*N/10);n++){
        PetscInt dof, cdof, d;

        PetscCall(PetscSectionGetFieldDof(s, n, 1, &dof));
        PetscCall(PetscSectionGetFieldConstraintDof(s, n, 1, &cdof));
        PetscCall(DMPlexPointGlobalFieldRead(dm, n, 1, v, &ptr));
        for (d = 0; d < dof-cdof; ++d) {
          PetscScalar h_neck;
          if (!user->necking) {
            h_neck = 0.9*param->h_0;
            user->factor = (user->l/param->h_0);
            if (ptr[d]<h_neck){
              user->necking = PETSC_TRUE;
              PetscPrintf(PETSC_COMM_WORLD, "\n \n \n ##### \t\t The necking begins \t\t ###### \n \n \n");
            }
            PetscScalar h_blow = 10*scale*param->h_0;
            if(!user->necking && ptr[d]>h_blow){
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "The curvature is blowing up in the middle. Something is wrong.");
            }
          }
        }
    }
    if((user->l > (10*scale)*param->h_0) && !user->necking) SETERRQ( PETSC_COMM_SELF, PETSC_ERR_SUP, "Necking did not happen. Something is wrong.");
    PetscCall(VecRestoreArrayRead(u, &v));
  }

  Flow_in = PETSC_PI*PetscSqr(param->h_0)*param->u_0*dt;
  user->V_t   = user->V_old + Flow_in;
  user->V_old = user->V_t;
  {
    const PetscScalar  *Consts;
    const PetscScalar  *v;
    PetscInt            NC;
    PetscInt            cEnd;
    const PetscScalar  *ptr_u = NULL;
    const PetscScalar  *ptr_s = NULL;

    /* Update length using u_tip */
    PetscCall(VecGetArrayRead(u, &v));
    PetscCall(DMPlexGetSimplexOrBoxCells(dm,0,NULL,&cEnd));
    PetscCall(DMPlexPointGlobalFieldRead(dm,2*cEnd,0, v, &ptr_u));
    PetscCall(DMPlexPointGlobalFieldRead(dm,2*cEnd,2, v, &ptr_s));
    stretch = (user->l + dt*(*ptr_u)) / (user->l);
    user->dl_dt = (*ptr_u)*stretch;
    user->s_bottom = (*ptr_s)*stretch;
    user->l *= stretch;
    PetscCall(VecRestoreArrayRead(u, &v));

    /* Change cell size for SUPG */
    PetscCall(PetscDSGetConstants(ds, &NC, &Consts));
    PetscScalar  *c = (PetscScalar*) Consts;
    PetscReal    cellsize;

    cellsize = user->l/(N-1);
    *(c+6) = (1.0)*cellsize/PetscPowReal(2,(user->N_adapts+4));
    /*
    For low viscosity fluid, set the viscosity to higher value and gradually reduce it.
    This is to smooth out the initial surface instabilities.
    */
    PetscReal X = ((user->l)-(param->h_0))/(param->h_0), n=02.0;
    if (param->nu_d < 1e-5){
      if(X < n) {
        *(c+0) = -param->nu_d*9*X/(n) + param->nu_d*10; /* linear approach towards the true viscosity */
      }
      else{
        *(c+0) = param->nu_d;
      }
    }

    // if (param->nu_c < 1e-5){
    //   if(X < n) {
    //     *(c+9) = -param->nu_c*99*X/(n) + param->nu_c*100; /* linear approach towards the true viscosity */
    //   }
    //   else{
    //     *(c+9) = param->nu_c;
    //   }
    // }
    // // n=03.0;
    // if (param->rho_c/param->rho_d > 10){
    //   if(X < n) {
    //     // *(c+10) = param->rho_c*0.9*X/(n) + param->rho_c*0.1; /* linear approach towards the true density */
    //     *(c+1) = -param->rho_d*99*X/(n) + param->rho_d*100; /* linear approach towards the true density */
    //   }
    //   else{
    //     // *(c+10) = param->rho_c;
    //     *(c+1) = param->rho_d;
    //   }
    // }

    if((user->l > 1.25*param->h_0) && stretch>1.0005) PetscCall(TSSetTimeStep(ts, 0.5*dt));
    if((param->h_0)>=0.0001 && (user->l > 1.5*param->h_0) && stretch<1.0001 && stretch>1.0) PetscCall(TSSetTimeStep(ts, 1.5*dt));
    PetscCall(PetscDSSetConstants(ds, NC, c));
    PetscPrintf(PETSC_COMM_WORLD, "N = %d (N_adapts = %d) \t G = %g, dp_dz = %g, \t \tTip Velocity = %g \t update factor = %g  \t Predicted length = %g (%g times h_0) \n\n", N, user->N_adapts, user->G, param->D, user->dl_dt, stretch, user->l, user->l/param->h_0);
    PetscPrintf(PETSC_COMM_WORLD, "D_effective = %g \t C = %g \t ln(C) = %g \t nu_c = %g \t nu_d = %g \t mu_ratio = %g \t rho_c = %g \t rho_d = %g \t rho_ratio =%g \t gamma = %g \n\n", param->D/(2.0*param->rho_d*PetscLogReal(param->C)) + 2.0*param->D/param->rho_d - (1 - *(c+10)/(*(c+1)))*param->g, param->C, PetscLogReal(param->C), *(c+9) , *(c+0), *(c+9)*(*(c+10))/(*(c+0)*(*(c+1))), *(c+10), (*(c+1)), *(c+10)/(*(c+1)), *(c+2) );
  }

  /* Decide when to adapt */
  if(user->necking && (user->l / param->h_0)>=user->factor){
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &user->adaptLabel));
    PetscCall(DMLabelSetDefaultValue(user->adaptLabel, DM_ADAPT_COARSEN));
    user->Bool = PETSC_TRUE;
  }
  PetscCall(DMCreateLocalVector(dm, &(user->locX)));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, user->locX));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, user->locX, time, NULL, NULL, NULL));

  /* Projection */
  PetscCall(DMClone(dm, &dmNew));
  PetscCall(DMCopyDisc(dm, dmNew));
  PetscCall(VecScale(coordinates, stretch));
  PetscCall(DMSetCoordinates(dmNew, coordinates));
  PetscCall(PetscObjectSetName((PetscObject) dmNew, "New Mesh"));

  PetscErrorCode    (*feFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
  void              *fectxs[3];
  feFuncs[0] = FieldFunction_u;
  feFuncs[1] = FieldFunction_h;
  feFuncs[2] = FieldFunction_s;
  fectxs[0]  = (void *) u;
  fectxs[1]  = (void *) u;
  fectxs[2]  = (void *) u;

  PetscCall(DMCreateGlobalVector(dmNew, &U));
  PetscCall(DMProjectFunction(dmNew, time, feFuncs, fectxs, INSERT_ALL_VALUES, U));
  PetscCall(DMDestroy(&dm));
  {
    DM_Plex       *mesh = (DM_Plex*) dmNew->data;
    PetscCall(PetscGridHashDestroy(&mesh->lbox));
  }
  if(param->x_p){
    PetscCall(DMGetDS(dmNew, &ds));
    const PetscScalar  *Consts;
    PetscInt            NC;
    PetscCall(PetscDSGetConstants(ds, &NC, &Consts));
    PetscScalar  *c = (PetscScalar*) Consts;
    *(c+8) = param->x_p;
    PetscCall(PetscDSSetConstants(ds, NC, c));
    PetscCall(TSSetMaxSteps(ts, stepi));
  }
  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, dmNew));
  PetscCall(TSSetSolution(ts, U));
  PetscCall(VecDestroy(&U));

  PetscFunctionReturn(0);
}
static PetscErrorCode PostStep(TS ts)
{
  DM             dm;
  AppCtx        *user;
  Parameter     *param;


  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(PetscBagGetData(user->bag, (void **) &param));

  if(user->adapt){
  /* Adaptively refine the mesh. No streching when adapting */
    DM                adm;
    Mat               Interp;
    Vec               u, au;
    PetscReal         dt, Body_force;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSGetTimeStep(ts, &dt));

    PetscCall(DMAdaptLabel(dm, user->adaptLabel, &adm));
    PetscCall(DMSetApplicationContext(adm, user));
    PetscCall(DMLabelDestroy(&(user->adaptLabel)));
    PetscCall(DMCreateInterpolation(dm, adm, &Interp, NULL));
    PetscCall(DMCreateGlobalVector(adm, &au));
    PetscCall(MatInterpolate(Interp, u, au));
    PetscCall(MatDestroy(&Interp));
    PetscCall(DMDestroy(&dm));
    PetscCall(TSReset(ts));
    PetscCall(TSSetDM(ts, adm));
    PetscCall(TSSetSolution(ts, au));
    PetscCall(VecDestroy(&au));
    {
      DM_Plex       *mesh = (DM_Plex*) adm->data;
      PetscCall(PetscGridHashDestroy(&mesh->lbox));
    }
    Body_force = PetscAbsReal(2.0*param->D/param->rho_d + param->D/(2.0*param->rho_d*PetscLogReal(param->C)) - (1 - param->rho_c/param->rho_d)*param->g);
    user->N -= 0.10;
    if (user->N < 0.0) user->N = 0.1;
    user->N_adapts += 1;
    user->factor += 0.750;

    // if (Body_force<=1.0*(param->g)) {
    //   user->factor += (param->h_0<0.001)? 0.35 : 0.70;
    //   printf("factor = %g\n", (param->h_0<0.001)? 0.35 : 0.70 );
    // }
    // else if (Body_force>1.0*(param->g) && Body_force<2.0*(param->g)) {
    //   user->factor += 1.50;
    //   printf("factor = 1.50\n" );
    // }
    // else if (Body_force>2.0*(param->g) && Body_force<4.0*(param->g)) {
    //   user->factor += 2.50;
    //   printf("factor = 2.50\n" );
    // }
    // else {
    //   user->factor += 3.50;
    //   printf("factor = 3.50\n" );
    // }

    PetscCall(TSSetTimeStep(ts, 0.5*dt));
    user->Bool = PETSC_FALSE;
    user->adapt = PETSC_FALSE;
  }
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


  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));

  PetscCall(DMGetCoordinates(dm, &coordinates));
  PetscCall(PetscSNPrintf(coords_name, PETSC_MAX_PATH_LEN, "Length_%d.out", user->N_adapts));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, coords_name));

  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, coords_name, &viewer));
  PetscCall(VecView(coordinates, viewer));

  PetscCall(PetscSNPrintf(sol_name, PETSC_MAX_PATH_LEN, "Numerical_Solution_%d", user->N_adapts));
  PetscCall(PetscObjectSetName((PetscObject) u, sol_name));
  PetscCall(VecViewFromOptions(u, NULL, "-sol_vec_view"));
  {
    PetscPointFunc funcs[3] = {u_radial, surface_energy, Curvature};
    Vec            curve;
    char           curve_name[PETSC_MAX_PATH_LEN];

    PetscCall(DMCreateGlobalVector(dm, &curve));
    PetscCall(DMProjectField(dm, crtime, u, funcs, INSERT_ALL_VALUES, curve));
    PetscCall(PetscSNPrintf(curve_name, PETSC_MAX_PATH_LEN, "Curvature_Derivative_%d", user->N_adapts));
    PetscCall(PetscObjectSetName((PetscObject) curve, curve_name));
    PetscCall(VecViewFromOptions(curve, NULL, "-curve_vec_view"));
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


  PetscCall(PetscInitialize(&argc, &argv, NULL,help);
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag));
  PetscCall(SetupParameters(user.bag, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(DMSetApplicationContext(dm, &user));
  /* Setup problem */
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject) u, "Numerical Solution"));

  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user));
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM , &user));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-monitor_off",&monitor_off));

  PetscCall(TSSetComputeInitialCondition(ts, SetInitialConditions)); /* Must come after SetFromOptions() */
  PetscCall(SetInitialConditions(ts, u));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMSetOutputSequenceNumber(dm, 0, t));
  PetscCall(DMTSCheckFromOptions(ts, u));
  if(!monitor_off) PetscCall(TSMonitorSet(ts, MonitorSolAndCoords, &user, NULL)));
  PetscCall(TSSetPreStep(ts, PreStep));
  {
    TSAdapt   adapt;
    PetscCall(TSGetAdapt(ts, &adapt));
    adapt->ops->choose = TSAdaptChoose_Volume;
  }
  PetscCall(TSSetPostStep(ts, PostStep));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(TSSolve(ts, NULL));
  PetscCall(DMTSCheckFromOptions(ts, u));
  PetscCall(TSGetDM(ts, &dm));

  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscBagDestroy(&user.bag));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Paraffin wax
  test:
    suffix: glycerol_gravity
    args: -h_0 0.00135 -u_0 0.05 -rho_d 1.20 -nu_d 1.5E-05 -gamma 0.0728 -rho_c 1000.00 -nu_c 1.0E-06 -R 0.0254 -G 0.00 -gr -9.81 \
          -cells 100 -dm_plex_separate_marker -dm_plex_transform_type refine_1d -dm_plex_hash_location \
          -vel_petscspace_degree 3 -rad_petscspace_degree 3 -slope_petscspace_degree 2 \
          -ts_max_steps 10000000 -ts_dt 1e-6 -ts_type beuler -ts_max_reject 20 -ts_monitor \
            -snes_converged_reason -snes_max_funcs 1000000  \
              -pc_type lu \ \
              -monitor_off
TEST*/
