const char help[] = "A test demonstrating stratum-dof grouping methods.\n";

#include <petscconvest.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/dmpleximpl.h>

const PetscReal DOMAIN_SPLIT = 0.0; // Used to switch value/form of the permeability tensor.
const PetscInt  RANDOM_SEED = 0; // Used to seed rng for mesh perturbations.
const PetscReal PUMP_HALF_SIDE_LENGTH = 2.5;
const PetscReal PUMP_STRENGTH = 2.5/PetscSqr(2*PUMP_HALF_SIDE_LENGTH); 

static PetscErrorCode smallSAXPY(PetscInt dim, const PetscScalar x[], PetscScalar alpha, PetscScalar* y){
  /* Updates y with y = ax+ y;*/
  PetscInt i;

  for (i = 0; i < dim; ++i){
    y[i] += alpha * x[i]; 
  }
  return 0;
}

static PetscErrorCode smallDot(PetscInt dim, const PetscScalar x[], const PetscScalar y[], PetscScalar* alpha){
  /* Updates  alpha += u\cdot v for small vectors.*/
  PetscInt i;

  for (i = 0; i < dim; ++i){
    *alpha += x[i] * y[i];
  }
  return 0;
}

static PetscErrorCode smallMatVec(PetscInt dim, const PetscScalar A[], const PetscScalar x[], PetscScalar *y) {
  /* Updates y with y += Ax */
  PetscErrorCode ierr;
  PetscInt j;

  for (j = 0; j < dim; ++j){
    ierr = smallSAXPY(dim,&A[j*dim],x[j],y);CHKERRQ(ierr);
  }
  return 0;
}

static PetscErrorCode smallMatMat(PetscInt dim, const PetscScalar A[], const PetscScalar B[], PetscScalar* C)
{
  /* Updates C with C += A*B */
  PetscErrorCode ierr;
  PetscInt j;

  for (j = 0; j<dim; ++j){
    ierr = smallMatVec(dim, A, &B[j*dim], &C[j*dim]);CHKERRQ(ierr);
  }
  return 0;
}

static PetscErrorCode smallDIP(PetscInt dim, const PetscScalar A[], const PetscScalar B[], PetscScalar* alpha)
{
  /* Updates alpha with alpha = A:B */
  PetscInt i,j;

  for (i = 0; i < dim; ++i){
    for (j = 0; j < dim; ++j){
      *alpha += A[j*dim + i] * B[i*dim +j];
    }
  }

  return 0;
}

/* Examples solve the system governed by:
 *
 * \vec{u} = -K\grad{p}
 * \div{\vec{u}} = f
 *
 * K is the the permeability tensor. We will often use K^{-1} which is the reluctivity tensor.
 */

static PetscErrorCode permeability_tensor(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,PetscScalar * u_x,void *ctx)
{
  /* Constructs the anisotropic, spacially varying permeability tensor using an eigenvalue decompositon. Q and Q_T are the left and right eigenbasis.
   * D holds the eigenvalues which controls the "strength" of the permeability. The varies continously, but not smoothly, in the domain. The
   * transition point is put close to half way, but not exactly so that it does not perfectly line up with the mesh. This function will also compute
   * the divergence of the permeability_tensor if u_x is not NULL.*/
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *Q,*Q_T,*D,*tmpu,*D_x,*tmpu_x;

  ierr = PetscCalloc2(dim*dim,&Q,dim*dim,&Q_T);CHKERRQ(ierr);
  if (u) {
    ierr = PetscCalloc2(dim*dim,&D,dim*dim,&tmpu);CHKERRQ(ierr);
  }
  if (u_x) {
    ierr = PetscCalloc2(dim*dim*dim,&D_x,dim,&tmpu_x);CHKERRQ(ierr);
  }
  switch (dim) {
  case 2:
    Q[0] = Q_T[0] = 0.6;
    Q[1] = Q_T[2] = 0.8;
    Q[2] = Q_T[1] = -0.8;
    Q[3] = Q_T[3] = 0.6;

    if (u) {
      D[0] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*12 : 18; //12. : 21.-20.*x[0]; */
      D[3] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*0.2 : 0.4;//0.2 + 50.*x[0] : 22.7; */
    }
    if (u_x) {
      /* The indexing scheme for D_x is sligtly different to facilitate MatVecs later on. D_x[k,i,j] = d/dx_i(D[j,k]) */
      D_x[0] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;   /*0 : -20; */
      D_x[6] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;  /*50. : 0; */
    }
    break;
  case 3:
    Q[0] = Q_T[0] = 1./3.;
    Q[1] = Q_T[3] = 2./3.;
    Q[2] = Q_T[6] = 2./3.;
    Q[3] = Q_T[1] = 0.;
    Q[4] = Q_T[4] = -2./PetscSqrtReal(8.);
    Q[5] = Q_T[7] = -2./PetscSqrtReal(8.);
    Q[6] = Q_T[2] = 8./PetscSqrtReal(72);
    Q[7] = Q_T[5] = -2./PetscSqrtReal(72);
    Q[8] = Q_T[8] = -2./PetscSqrtReal(72);
    if (u) {
      D[0] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*12. : 21.-20.*x[0]; */
      D[4] = (x[0] < DOMAIN_SPLIT) ? 1 : 1; /*0.2 + 50.*x[0] : 22.7; */
      D[8] = (x[2] < DOMAIN_SPLIT) ? 1 : 1;/*0.1 + x[0] + 2*x[1] + x[2]*x[2] : 0.3025 + x[0] + 2*x[1]; */
    }
    if (u_x) {
      D_x[0]  = (x[0] < DOMAIN_SPLIT) ? 0 : 0;/*0 : -20.; */
      D_x[12] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;/* 50. : 0.; */
      D_x[24] = 0;  /*1; */
      D_x[25] = 0;  /*2; */
      D_x[26] = (x[2] < DOMAIN_SPLIT) ? 0 : 0;/*2*x[2] : 0; */
    }
    break;
  }
  if (u) {
    ierr = PetscArrayzero(u,dim*dim);CHKERRQ(ierr);
    ierr = smallMatMat(dim,D,Q_T,tmpu);CHKERRQ(ierr);
    ierr = smallMatMat(dim,Q,tmpu,u);CHKERRQ(ierr);
    ierr = PetscFree2(D,tmpu);CHKERRQ(ierr);
  }
  if (u_x) {
    ierr = PetscArrayzero(u_x,dim);CHKERRQ(ierr);
    for (i = 0; i < dim; ++i) {
      ierr = smallDIP(dim,&D_x[i*dim*dim],Q_T,&tmpu_x[i]);CHKERRQ(ierr);
    }
    ierr = smallMatVec(dim,Q,tmpu_x,u_x);CHKERRQ(ierr);
    ierr = PetscFree2(D_x,tmpu_x);CHKERRQ(ierr);
  }

  ierr = PetscFree2(Q,Q_T);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode reluctivity_tensor(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,PetscScalar * u_x,void *ctx)
{
  /* Constructs the reluctivity tensor which is the inverse of the permeability. This function will also compute
   * the divergence of the reluctivity_tensor if u_x is not NULL.*/
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *Q,*Q_T,*D,*tmpu,*D_x,*tmpu_x;

  ierr = PetscCalloc2(dim*dim,&Q,dim*dim,&Q_T);CHKERRQ(ierr);
  if (u) {
    ierr = PetscCalloc2(dim*dim,&D,dim*dim,&tmpu);CHKERRQ(ierr);
  }
  if (u_x) {
    ierr = PetscCalloc2(dim*dim*dim,&D_x,dim,&tmpu_x);CHKERRQ(ierr);
  }
  switch (dim) {
  case 2:
    Q[0] = Q_T[0] = 0.6;
    Q[1] = Q_T[2] = 0.8;
    Q[2] = Q_T[1] = -0.8;
    Q[3] = Q_T[3] = 0.6;

    if (u) {
      D[0] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*1./12 : 1./18; //1./12. : 1./(21.-20*x[0]); */
      D[3] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*1./0.2 : 1/0.4; //1./(0.2 + 50.*x[0]) : 1./22.7; */
    }
    if (u_x) {
      /* The indexing scheme for D_x is sligtly different to facilitate MatVecs later on. D_x[k,i,j] = d/dx_i(D[j,k]) */
      D_x[0] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;  /*0 : 20./PetscSqr(21-20*x[0]); */
      D_x[6] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;   /*-50./PetscSqr(0.2+50.*x[0]) : 0; */
    }
    break;
  case 3:
    Q[0] = Q_T[0] = 1./3.;
    Q[1] = Q_T[3] = 2./3.;
    Q[2] = Q_T[6] = 2./3.;
    Q[3] = Q_T[1] = 0.;
    Q[4] = Q_T[4] = -2./PetscSqrtReal(8.);
    Q[5] = Q_T[7] = -2./PetscSqrtReal(8.);
    Q[6] = Q_T[2] = 8./PetscSqrtReal(72);
    Q[7] = Q_T[5] = -2./PetscSqrtReal(72);
    Q[8] = Q_T[8] = -2./PetscSqrtReal(72);
    if (u) {
      D[0] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*1./12. : 21.-20*x[0]; */
      D[4] = (x[0] < DOMAIN_SPLIT) ? 1 : 1;/*1./(0.2 + 50.*x[0]) : 1./22.7; */
      D[8] = (x[2] < DOMAIN_SPLIT) ? 1 : 1;/*1./(0.1 + x[0] + 2*x[1] + x[2]*x[2]) : 1./(0.3025 + x[0] + 2*x[1]); */
    }
    if (u_x) {
      D_x[0]  = (x[0] < DOMAIN_SPLIT) ? 0 : 0;/*0 : 20./PetscSqr(21-20*x[0]); */
      D_x[12] = (x[0] < DOMAIN_SPLIT) ? 0 : 0;/* -50./PetscSqr(0.2+50.*x[0]) : 0.; */
      D_x[24] = (x[2]<DOMAIN_SPLIT) ? 0 : 0;/*-1./PetscSqr(0.1 + x[0] + 2*x[1] + x[2]*x[2]) : -1./PetscSqr(0.3025 + x[0] + 2*x[1]); */
      D_x[25] = (x[2] < DOMAIN_SPLIT) ? 0 : 0;/* -2./PetscSqr(0.1 + x[0] + 2*x[1] + x[2]*x[2]) : -2./PetscSqr(0.3025 + x[0] + 2*x[1]); */
      D_x[26] = (x[2] < DOMAIN_SPLIT) ? 0 : 0;/* -2*x[2]/PetscSqr(0.1 + x[0] + 2*x[1] + x[2]*x[2]) :  0; */
    }
    break;
  }
  if (u) {
    ierr = PetscArrayzero(u,dim*dim);CHKERRQ(ierr);
    ierr = smallMatMat(dim,D,Q_T,tmpu);CHKERRQ(ierr);
    ierr = smallMatMat(dim,Q,tmpu,u);CHKERRQ(ierr);
    ierr = PetscFree2(D,tmpu);CHKERRQ(ierr);
  }
  if (u_x) {
    ierr = PetscArrayzero(u_x,dim);CHKERRQ(ierr);
    for (i = 0; i < dim; ++i) {
      ierr = smallDIP(dim,&D_x[i*dim*dim],Q,&tmpu_x[i]);CHKERRQ(ierr);
    }
    ierr = smallMatVec(dim,Q,tmpu_x,u_x);CHKERRQ(ierr);
    ierr = PetscFree2(D_x,tmpu_x);CHKERRQ(ierr);
  }

  ierr = PetscFree2(Q,Q_T);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar* u, void * ctx){
  PetscInt c;
  for (c=0; c<Nc; ++c) u[c] = 0.0;
  return 0;
}

static PetscErrorCode ten(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar* u, void * ctx){
  PetscInt c;
  for (c=0; c<Nc; ++c) u[c] = 10.0;
  return 0;
}
static PetscErrorCode hiHead(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar* u, void *ctx){
  u[0] = 105.00;
  return 0;
}

static PetscErrorCode loHead(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar* u, void *ctx){
  u[0] = 100.00;
  return 0;
}
/* We label solutions by the form of the potential/pressure, p: i.e. linear_u is the analytical form of u one gets when p is linear. */
/* 2D Linear Exact Functions
   p = x;
   \vec{u} = <-1, 0>;
   f = 0;
   \div{\vec{u}} = 0;
   */
static PetscErrorCode linear_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  u[0] = x[0];
  return 0;
}

static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  /* Need to set only the x-component i.e. c==0  */
  PetscErrorCode ierr;
  PetscScalar *K,*gradP;
  ierr = PetscCalloc2(dim*dim,&K,dim,&gradP);CHKERRQ(ierr);
  ierr = PetscArrayzero(u,dim);CHKERRQ(ierr);
  ierr = permeability_tensor(dim, time,x,Nc,K,NULL,ctx);CHKERRQ(ierr);
  gradP[0] = -1.0;
  ierr = smallMatVec(dim,K,gradP,u);CHKERRQ(ierr);

  ierr = PetscFree2(K,gradP);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode linear_source(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  PetscErrorCode ierr;
  PetscScalar *gradP,*divK;

  ierr = PetscCalloc2(dim,&divK,dim,&gradP);CHKERRQ(ierr);
  gradP[0] = -1.0;
  ierr = permeability_tensor(dim,time,x,Nc,NULL,divK,ctx);CHKERRQ(ierr);
  *u = 0.0;
  ierr = smallDot(dim,divK,gradP,u);CHKERRQ(ierr);
  ierr = PetscFree2(divK,gradP);CHKERRQ(ierr);
  return 0;
}

/* 2D Quadratic Exact Functions */
static PetscErrorCode quad_p(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar * u,void *ctx) {
  u[0] = PetscSqr(x[0]) + x[0]*x[1];
  return 0;
}

static PetscErrorCode quad_u(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar * u,void *ctx){
  PetscErrorCode ierr;
  PetscScalar *K,*gradP;
  ierr = PetscCalloc2(dim*dim,&K,dim,&gradP);CHKERRQ(ierr);
  ierr = PetscArrayzero(u,dim);CHKERRQ(ierr);
  ierr = permeability_tensor(dim, time,x,Nc,K,NULL,ctx);CHKERRQ(ierr);
  gradP[0] = -2.0 * x[0] - x[1];
  gradP[1] = -x[0];
  ierr = smallMatVec(dim,K,gradP,u);CHKERRQ(ierr);

  ierr = PetscFree2(K,gradP);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode quad_source(PetscInt dim,PetscReal time, const PetscReal x[],PetscInt Nc,PetscScalar * u, void *ctx){
  PetscErrorCode ierr;
  PetscScalar *K,*gradP,*divK,*gradgradP; 
  
  *u = 0;
  ierr = PetscCalloc4(dim*dim,&K,dim,&divK,dim,&gradP,dim*dim,&gradgradP);CHKERRQ(ierr);
  ierr = permeability_tensor(dim,time,x,Nc,K,divK,ctx);CHKERRQ(ierr);
  gradP[0] = -2.0*x[0] - x[1];
  gradP[1] = -x[0];
  gradgradP[0] = -2.0;
  gradgradP[1] = -1.0;
  gradgradP[2] = -1.0;
  ierr = smallDot(dim,divK,gradP,u);CHKERRQ(ierr);
  ierr = smallDIP(dim,K,gradgradP,u);CHKERRQ(ierr);


  ierr = PetscFree4(K,divK,gradP,gradgradP);CHKERRQ(ierr); 
  return 0;
}

/* 2D Sinusoidal Exact Functions
   p = sin(2*pi*x)*sin(2*pi*y);
   \vec{u} = <2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*cos(2*pi*y)*sin(2*pi*x);
   \div{\vec{u}} = -8*pi^2*sin(2*pi*x)*sin(2*pi*y);
   */
static PetscErrorCode sinusoid_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  u[0] = 1;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= PetscSinReal(2 * PETSC_PI * x[d]);
  return 0;
}

static PetscErrorCode sinusoid_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  PetscErrorCode ierr;
  PetscScalar *K,*gradP;
  PetscInt c,d;

  ierr = PetscCalloc2(dim*dim,&K, dim,&gradP);
  ierr = PetscArrayzero(u,dim);CHKERRQ(ierr);
  ierr = permeability_tensor(dim,time,x,Nc,K,NULL,ctx);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) {
    gradP[c] = -2.0*PETSC_PI;
    for (d = 0; d < dim; ++d) {
      if (d == c) gradP[c] *= PetscCosReal(2 * PETSC_PI * x[d]);
      else gradP[c] *= PetscSinReal(2 * PETSC_PI * x[d]);
    }
  }
  ierr = smallMatVec(dim,K,gradP,u);CHKERRQ(ierr);
  ierr = PetscFree2(K,gradP);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode sinusoid_source(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx)
{
  PetscErrorCode ierr;
  PetscScalar *K,*divK,*gradP,*gradgradP;
  PetscInt c,d,dd;

  *u = 0;
  ierr = PetscCalloc4(dim*dim,&K,dim,&divK,dim,&gradP,dim*dim,&gradgradP);CHKERRQ(ierr);
  ierr = permeability_tensor(dim,time,x,Nc,K,divK,ctx);CHKERRQ(ierr);
 
  for (c = 0; c < Nc; ++c) {
    gradP[c] = -2.0*PETSC_PI;
    for (d = 0; d < dim; ++d) {
      if (d == c) gradP[c] *= PetscCosReal(2 * PETSC_PI * x[d]);
      else gradP[c] *= PetscSinReal(2 * PETSC_PI * x[d]);

      gradgradP[c*dim + d] = -4.0*PetscSqr(PETSC_PI);
      for (dd = 0; dd < dim; ++dd) {
        if (dd == c && dd == d) gradgradP[c*dim + d] *= -1.0* PetscSinReal(2 * PETSC_PI * x[dd]);
        else if (dd == c || dd == d) gradgradP[c*dim +d] *= PetscCosReal(2*PETSC_PI*x[dd]);
        else gradgradP[c*dim + d] *= PetscSinReal(2*PETSC_PI*x[dd]);
      }
    }
  }

  ierr = smallDot(dim,divK,gradP,u);CHKERRQ(ierr);
  ierr = smallDIP(dim,K,gradgradP,u);CHKERRQ(ierr);

  ierr = PetscFree4(K,divK,gradP,gradgradP);CHKERRQ(ierr); 
  return 0;
}

static PetscErrorCode subsurface_source(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar * u,void * ctx){
  PetscBool inBox = PETSC_TRUE;
  PetscInt i;
  for (i = 0; i< dim; ++i){
   inBox = inBox && (x[i] < PUMP_HALF_SIDE_LENGTH && x[i] > -PUMP_HALF_SIDE_LENGTH);
  }
  u[0] = inBox ? PUMP_STRENGTH : 0;
  return 0;
}
/* Pointwise function for (v,u) */
static void f0_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],
                 const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],
                 const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscErrorCode ierr;
  PetscScalar *K_inv;

  ierr = PetscCalloc1(dim*dim,&K_inv);
  ierr = reluctivity_tensor(dim,t,x,dim,K_inv,NULL,NULL);
  ierr = PetscArrayzero(f0,dim);
  ierr = smallMatVec(dim,K_inv,&u[uOff[0]],f0);
  ierr = PetscFree(K_inv);CHKERRV(ierr);
}

/* This is the pointwise function that represents (\trace(\grad v),p) == (\grad
 * v : I*p) */
static void f1_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],
                 const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],
                 const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c)
      f1[c * dim + c] = -u[uOff[1]];
}

/* represents (\div u - f,q). */
static void f0_q_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                        const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                        const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,
                        const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar rhs = 0.0;
  PetscScalar divu;

  (void)linear_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  /* diagonal terms of the gradient */
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i * dim + i];
  f0[0] = divu - rhs;
}

static void f0_q_quad(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                        const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                        const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,
                        const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar rhs = 0.0;
  PetscScalar divu;

  (void)quad_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  /* diagonal terms of the gradient */
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i * dim + i];
  f0[0] = divu - rhs;
}
static void f0_q_sinusoid(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                          const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                          const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,
                          const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar rhs;
  PetscScalar divu;

  (void)sinusoid_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i * dim + i];
  f0[0] = divu - rhs;
}

static void f0_q_subsurface(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                          const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                          const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,
                          const PetscScalar constants[],PetscScalar f0[]){
  PetscInt i;
  PetscScalar rhs;
  PetscScalar divu;

  (void)subsurface_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i*dim+i];
  f0[0] = divu - rhs;
}

static void f0_linear_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                           const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                           const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,
                           const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;

  (void)linear_p(dim,t,x,dim,&pressure,NULL);
  for (PetscInt d = 0; d < dim; ++d) f0[d] = pressure * n[d];
}

static void f0_quad_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                           const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                           const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,
                           const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;

  (void)quad_p(dim,t,x,dim,&pressure,NULL);
  for (PetscInt d = 0; d < dim; ++d) f0[d] = pressure * n[d];
}

static void f0_sinusoid_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                             const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                             const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],
                             PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;

  (void)sinusoid_p(dim,t,x,dim,&pressure,NULL);
  for (PetscInt d = 0; d < dim; ++d) f0[d] = pressure * n[d];
}

static void f0_subsurface_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],
                             const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],
                             const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],
                             PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[]){
  PetscInt d;
  PetscScalar pressure;

  if (x[0] == -102.5){
    (void)hiHead(dim,t,x,1,&pressure,NULL);
  } else if (x[0] == 102.5){
    (void)loHead(dim,t,x,1,&pressure,NULL);
  }
  

  for (d=0; d< dim; ++d) f0[d] = pressure * n[d];
}

/* <v, u> */
static void g0_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],
                  const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],
                  const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],
                  PetscScalar g0[])
{
  PetscErrorCode ierr;
  ierr = reluctivity_tensor(dim,t,x,dim,g0,NULL,NULL);CHKERRV(ierr);
}

/* <-p,\nabla\cdot v> = <-pI,\nabla u> */
static void g2_vp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],
                  const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],
                  const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],
                  PetscScalar g2[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g2[c * dim + c] = -1.0;
}

/* <q, \nabla\cdot u> */
static void g1_qu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],
                  const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],
                  const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],
                  PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

typedef enum
{
  NONE = 0,
  RANDOM = 1,
  SKEW = 2,
  SKEWRAND = 3
} Perturbation;
const char* const PerturbationTypes[] =
{"none","random","skew","skewrand","Perturbation","",NULL};

typedef enum
{
  LINEAR = 0,
  SINUSOIDAL = 1,
  QUADRATIC = 2,
  SUBSURFACE_BENCHMARK=3
} Solution;
const char* const SolutionTypes[] = {"linear",
                                     "sinusoidal",
                                     "quadratic",
                                     "subsurface_benchmark",
                                     "Solution",
                                     "",
                                     NULL};

typedef struct
{
  PetscBool    simplex;
  PetscInt     dim;
  Perturbation mesh_transform;
  Solution     sol_form;
  PetscBool    showNorm;
  PetscBool    toFile;
  char         filename[128];
} UserCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm,UserCtx * user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  user->simplex        = PETSC_TRUE;
  user->dim            = 2;
  user->mesh_transform = NONE;
  user->sol_form       = LINEAR;
  user->showNorm       = PETSC_FALSE;
  user->toFile         = PETSC_FALSE;
  ierr = PetscStrncpy(user->filename,"",128);CHKERRQ(ierr);
  /* Define/Read in example parameters */
  ierr = PetscOptionsBegin(comm,"","Stratum Dof Grouping Options","DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex","Whether to use simplices (true) or tensor-product (false) cells in the mesh","ex38.c",user->simplex,
                          &user->simplex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Number of solution dimensions","ex38.c",user->dim,&user->dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mesh_transform","Method used to perturb the mesh vertices. Options are Skew,Random," "SkewRand,or None","ex38.c",
                          PerturbationTypes,(PetscEnum)user->mesh_transform,(PetscEnum*)&user->mesh_transform,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sol_form","Form of the exact solution. Options are Linear or Sinusoidal","ex38.c",SolutionTypes,(PetscEnum)user->sol_form,
                          (PetscEnum*)&user->sol_form,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-showNorm","Whether to print the norm of the difference between lumped and unlumped solutions.","ex38.c",user->showNorm,
                          &user->showNorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-toFile","Whether to print solution errors to file.","ex38.c",user->toFile,&user->toFile,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename","If printing results to file, the path to use.","ex38.c",user->filename,user->filename,128,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbMesh(DM * mesh,PetscScalar * coordVals,PetscInt ncoord,PetscInt dim,PetscRandom * ran)
{
  PetscErrorCode ierr;
  PetscReal      minCoords[3],maxCoords[3],maxPert,randVal,nodePerEdge;
  PetscScalar    phase,amp;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  ierr = DMGetLocalBoundingBox(*mesh,minCoords,maxCoords);CHKERRQ(ierr);

  /* Compute something ~= half an edge length. This is the most we can perturb
   * points and gaurantee that there won't be any topology issues. */
  nodePerEdge = PetscPowReal(ncoord,1./dim);
  maxPert = 0.3/nodePerEdge;
  for (int i = 0; i < ncoord; ++i) {
    for (int j = 0; j < dim; ++j) {
      ierr                    = PetscRandomGetValueReal(*ran,&randVal);CHKERRQ(ierr);
      amp                     = maxPert * (randVal - 0.5);
      coordVals[dim * i + j] += amp;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SkewMesh(DM * mesh,PetscScalar * coordVals,PetscInt ncoord,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscReal      * transMat;

  PetscFunctionBegin;
  ierr = PetscCalloc1(dim * dim,&transMat);CHKERRQ(ierr);

  /* Make a matrix representing a skew transformation */
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) transMat[i * dim + j] = 1;
      else if (j < i) transMat[i * dim + j] = 2 * (j + i);
      else transMat[i * dim + j] = 0;
    }

  /* Multiply each coordinate vector by our tranformation */
  for (int i = 0; i < ncoord; ++i) {
    PetscReal tmpcoord[3];
    for (int j = 0; j < dim; ++j) {
      tmpcoord[j] = 0;
      for (int k = 0; k < dim; ++k) tmpcoord[j] += coordVals[dim * i + k] * transMat[dim * k + j];
    }
    for (int l = 0; l < dim; ++l) coordVals[dim * i + l] = tmpcoord[l];
  }
  ierr = PetscFree(transMat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TransformMesh(UserCtx * user,DM * mesh,PetscRandom * ran)
{
  PetscErrorCode ierr;
  PetscInt       dim,ncoord;
  PetscScalar    * coordVals;
  Vec            coords;

  PetscFunctionBegin;
  ierr   = DMGetCoordinates(*mesh,&coords);CHKERRQ(ierr);
  ierr   = VecGetArray(coords,&coordVals);CHKERRQ(ierr);
  ierr   = VecGetLocalSize(coords,&ncoord);CHKERRQ(ierr);
  ierr   = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  ncoord = ncoord / dim;

  switch (user->mesh_transform) {
  case NONE:
    break;
  case RANDOM:
    ierr = PerturbMesh(mesh,coordVals,ncoord,dim,ran);CHKERRQ(ierr);
    break;
  case SKEW:
    ierr = SkewMesh(mesh,coordVals,ncoord,dim);CHKERRQ(ierr);
    break;
  case SKEWRAND:
    ierr = SkewMesh(mesh,coordVals,ncoord,dim);CHKERRQ(ierr);
    ierr = PerturbMesh(mesh,coordVals,ncoord,dim,ran);CHKERRQ(ierr);
    break;
  default:
    PetscFunctionReturn(-1);
  }
  ierr = VecRestoreArray(coords,&coordVals);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*mesh,coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * PetscSectionGetFieldChart - Get the chart range for points which store the
 * DoFs of a specified filed.
 *
 * Input Parameters:
 * + s - The PetscSection
 * - field - The index of the field of interest
 *
 *   Output Parameters:
 *   + pStart - Start index of the chart
 *   - pEnd - End index of the chart
 *
 *   Level:
 */
PetscErrorCode PetscSectionGetFieldChart(PetscSection s,PetscInt field,PetscInt *    pStart,PetscInt *    pEnd)
{
  PetscErrorCode ierr;
  PetscSection   fieldSec;
  PetscInt       cBegin,cEnd,nDof;

  PetscFunctionBegin;
  ierr = PetscSectionGetField(s,field,&fieldSec);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(fieldSec,&cBegin,&cEnd);CHKERRQ(ierr);

  for (PetscInt p = cBegin; p < cEnd; ++p) {
    ierr = PetscSectionGetDof(fieldSec,p,&nDof);CHKERRQ(ierr);
    if (nDof > 0) {
      *pStart = p;
      break;
    }
  }

  for (PetscInt p = cEnd - 1; p >= cBegin; --p) {
    ierr = PetscSectionGetDof(fieldSec,p,&nDof);CHKERRQ(ierr);
    if (nDof > 0) {
      *pEnd = p + 1;
      break;
    }
  }

  /* TODO: Handle case where no points in the current section have DoFs
   * belonging to the specified field. Possibly by returning negative values for
   * pStart and pEnd */

  PetscFunctionReturn(0);
}

/*
 * DMPlexGetFieldDepth - Find the stratum on which the desired
 * field's DoFs are currently assigned.
 *
 * Input Parameters:
 * + dm - The DM
 * - field - Index of the field on the DM
 *
 *   Output Parameters:
 *   - depth - The depth of the stratum that to which field's DoFs are assigned
 */
PetscErrorCode DMPlexGetFieldDepth(DM dm,PetscInt field,PetscInt * depth)
{
  PetscErrorCode ierr;
  PetscSection   localSec;
  PetscInt       maxDepth,fStart = -1,fEnd = -1,pStart,pEnd;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,&localSec);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldChart(localSec,field,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&maxDepth);CHKERRQ(ierr);

  for (*depth = 0; *depth <= maxDepth; ++(*depth)) {
    ierr = DMPlexGetDepthStratum(dm,*depth,&pStart,&pEnd);CHKERRQ(ierr);
    if (pStart == fStart && pEnd == fEnd) break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionInvertMapping(PetscSection s,IS is,PetscSection * newSec,IS * newIs)
{
  /* We take a map, implemented as a Section and IS, and swap the domain (points) and range (DoFs) to create a new map from the range into the
   * domain.*/
  PetscErrorCode ierr;
  PetscInt       sStart,sEnd,isStart,isEnd,point,pOff,pNum,j,newPoint,*newIsInd,totalDof=0,*pointOffTracker,newOff;
  const PetscInt * isInd;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s,&sStart,&sEnd);CHKERRQ(ierr);
  ierr = ISGetMinMax(is,&isStart,&isEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&isInd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)s),newSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*newSec,isStart,isEnd+1);CHKERRQ(ierr);

  for (point = sStart; point < sEnd; ++point) {
    /* Here we allocate the space needed for our map. Everytime we encounter a DoF in the range of the given map
     * (i.e. there is a point x which maps to the DoF y) we must add space for a DoF to our new map at the point y.*/
    ierr = PetscSectionGetOffset(s,point,&pOff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s,point,&pNum);CHKERRQ(ierr);
    for (j = pOff; j < pOff+pNum; ++j) {
      newPoint = isInd[j];
      PetscSectionAddDof(*newSec,newPoint,1);
      ++totalDof;
    }
  }

  ierr = PetscSectionSetUp(*newSec);CHKERRQ(ierr);
  ierr = PetscCalloc1(totalDof,&newIsInd);CHKERRQ(ierr);
  ierr = PetscCalloc1(isEnd-isStart,&pointOffTracker);CHKERRQ(ierr);
  /* At this point newSec will give the proper offset and numDof information */

  for (point = sStart; point < sEnd; ++point) {
    /* Now we assign values into the newIS to complete the mapping. When we encounter a point x that maps to y under the given
     * mapping we put x into our new IS and increment a counter to keep track of how many Dofs we have currently assigned to y. */

    ierr = PetscSectionGetOffset(s,point,&pOff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(s,point,&pNum);CHKERRQ(ierr);
    for (j = pOff; j < pOff+pNum; ++j) {
      PetscInt currOffset;

      newPoint                    = isInd[j];
      ierr                        = PetscSectionGetOffset(*newSec,newPoint,&newOff);CHKERRQ(ierr);
      currOffset                  = pointOffTracker[newPoint-isStart];
      newIsInd[newOff+currOffset] = point;
      ++pointOffTracker[newPoint-isStart];
    }
  }

  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),totalDof,newIsInd,PETSC_OWN_POINTER,newIs);

  ierr = PetscFree(pointOffTracker);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&isInd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * DMPlexGetStratumMap - Create a mapping in the form of a section and IS which
 * associates points from one stratum with points from another.
 *
 * Input Parameters:
 * + dm - The DM
 * . source - Depth value of the source stratum
 * - target - Depth value of the target stratum
 *
 *   Output Parameters:
 *   + s - PetscSection which contains the number of target points and offset
 *   for each point in source.
 *   - is - IS containing the indices of points in target stratum
 */
PetscErrorCode DMPlexGetStratumMap(DM dm,PetscInt source,PetscInt target,PetscSection * s,IS * is)
{
  PetscErrorCode ierr;
  PetscInt       pStart,pEnd,tStart,tEnd,nClosurePoints,*closurePoints = NULL,
                 isCount = 0,*idx,pOff,*pCount;
  PetscBool inCone = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,s);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,source,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,target,&tStart,&tEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*s,pStart,pEnd);CHKERRQ(ierr);
  if (source == target) {
    /* Each point maps to itself and only to itself, this is the trivial map
     */

    for (PetscInt p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetDof(*s,p,1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(*s);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,pEnd - pStart,pStart,1,is);CHKERRQ(ierr);
  } else {
    if (source < target) inCone = PETSC_FALSE;
    /* TODO: This routine currently relies on a number of calls to
     * DMPlexGetTransitiveClosure. Determine whether there is a more efficient
     * method and/or if this is an instance of reinventing the wheel due to
     * existence of PetscSectionGetClosureIndex */

    /* Count the number of target points for each source
     * so that the proper amount of memory can be allocated for the section and
     * IS */
    for (PetscInt p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);

      for (PetscInt cp = 0; cp < nClosurePoints; ++cp) {
        PetscInt closurePoint = closurePoints[2 * cp];
        /* Check if closure point is in target stratum */
        if (closurePoint >= tStart && closurePoint < tEnd) {
          /* Add a DoF to the section and increment IScount */
          ierr = PetscSectionAddDof(*s,p,1);CHKERRQ(ierr);
          ++isCount;
        }
      }

      ierr = DMPlexRestoreTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
    }

    ierr = PetscSectionSetUp(*s);CHKERRQ(ierr);
    ierr = PetscCalloc1(isCount,&idx);CHKERRQ(ierr);
    ierr = PetscCalloc1(pEnd-pStart,&pCount);CHKERRQ(ierr);

    /* Now that proper space is allocated assign the correct values to the IS
     * TODO: Check that this method of construction preserves the orientation*/
    for (PetscInt p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(*s,p,&pOff);CHKERRQ(ierr);

      for (PetscInt cp = 0; cp < nClosurePoints; ++cp) {
        PetscInt closurePoint = closurePoints[2 * cp];
        /* Check if closure point is in target stratum */
        if (closurePoint >= tStart && closurePoint < tEnd) idx[pOff + pCount[p-pStart]++] = closurePoint;
      }

      ierr = DMPlexRestoreTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
    }
    ierr = ISCreateGeneral(
      PETSC_COMM_WORLD,isCount,idx,PETSC_OWN_POINTER,is
      );CHKERRQ(ierr);
  }

  PetscFree(pCount);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * DMPlexGetStratumDofMap - Create a map consisting of a PetscSection and IS
 * from a specified stratum to the DoFs of the specified field.
 *
 * Input Parameters:
 * + dm - The DM.
 * . stratum - The depth value of the stratum to map from.
 * - field - The index of the field whose DoFs are to be assigned to stratum.
 *
 *   Output Parameters:
 *   + section - The section to be created consisting of points in the stratum.
 *   - is - The IS to be created which will contain indices of the field DoFs.
 */
PetscErrorCode DMPlexGetStratumDofMap(DM dm,PetscInt stratum,PetscInt field,PetscSection *section,IS *is)
{
  PetscErrorCode ierr;
  PetscInt
                 fieldDepth,pStart,pEnd,fStart,fEnd,*idx,dofCount=0,numDof,numPoints,pOff,stratSize,*stratDofCount,sOff,mapInd,sInd0,dofOff;
  const PetscInt *stratInds;
  PetscSection   stratumSec,localSec;
  IS             stratum2Stratum;

  PetscFunctionBegin;
  ierr      = DMGetLocalSection(dm,&localSec);CHKERRQ(ierr);
  ierr      = DMPlexGetFieldDepth(dm,field,&fieldDepth);CHKERRQ(ierr);
  ierr      = DMPlexGetDepthStratum(dm,fieldDepth,&fStart,&fEnd);CHKERRQ(ierr);
  ierr      = DMPlexGetDepthStratum(dm,stratum,&pStart,&pEnd);CHKERRQ(ierr);
  ierr      = PetscSectionCreate(PETSC_COMM_WORLD,section);CHKERRQ(ierr);
  ierr      = PetscSectionSetChart(*section,pStart,pEnd);CHKERRQ(ierr);
  stratSize = pEnd-pStart;
  ierr      =
    DMPlexGetStratumMap(dm,fieldDepth,stratum,&stratumSec,&stratum2Stratum);CHKERRQ(ierr);
  ierr = ISGetIndices(stratum2Stratum,&stratInds);CHKERRQ(ierr);

  for (PetscInt i = fStart; i <fEnd; ++i) {
    ierr = PetscSectionGetDof(localSec,i,&numDof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(stratumSec,i,&numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(stratumSec,i,&pOff);CHKERRQ(ierr);
    if (numDof==numPoints) {
      for (PetscInt j = 0; j<numDof; ++j) {
        ierr = PetscSectionAddDof(*section,stratInds[pOff + j],1);CHKERRQ(ierr);
      }
      dofCount += numDof;
    }
  }

  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscCalloc1(stratSize,&stratDofCount);CHKERRQ(ierr);
  ierr = PetscCalloc1(dofCount,&idx);CHKERRQ(ierr);

  for (PetscInt i = fStart; i < fEnd; ++i) {
    ierr = PetscSectionGetDof(localSec,i,&numDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSec,i,&dofOff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(stratumSec,i,&numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(stratumSec,i,&pOff);CHKERRQ(ierr);
    if (numDof==numPoints)
      for (PetscInt j = 0; j < numDof; ++j) {
        sInd0 = stratInds[pOff+j] - pStart;
        ierr  =
          PetscSectionGetOffset(*section,stratInds[pOff+j],&sOff);CHKERRQ(ierr);
        mapInd      = sOff + stratDofCount[sInd0]++;
        idx[mapInd] = dofOff+j;
      }
  }
  ierr = ISRestoreIndices(stratum2Stratum,&stratInds);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,dofCount,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  ierr = PetscFree(stratDofCount);CHKERRQ(ierr);
  ierr = ISDestroy(&stratum2Stratum);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&stratumSec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * PetscDualSpaceProjectConstants - Create matrices representing the projection of constant functions onto the DoFs.
 */
PetscErrorCode PetscDualSpaceProjectConstants(PetscDualSpace dual,PetscScalar* constants) {
  /* Set up variables in local scope */
  PetscQuadrature allQuad;
  Mat             allMat;
  PetscInt        dim,c,numComponents,totNumDoF=0,p,numPoints,dof;
  PetscScalar*    constEvals;
  PetscErrorCode  ierr;

  ierr = PetscDualSpaceGetAllData(dual,&allQuad,&allMat);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(dual,&numComponents);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(allQuad, &dim,NULL,&numPoints,NULL,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(allMat,&totNumDoF,&numPoints);CHKERRQ(ierr);

  ierr = PetscCalloc1(numPoints,&constEvals);CHKERRQ(ierr);
  
  for (c = 0; c < numComponents; ++c){
    ierr = PetscArrayzero(constEvals,numPoints);
    for (p = c; p < numPoints; p+=numComponents){
      constEvals[p] = 1.0;
    }
    ierr = PetscDualSpaceApplyAll(dual,constEvals,&constants[c*totNumDoF]);CHKERRQ(ierr);
  }

  ierr = PetscFree(constEvals);CHKERRQ(ierr);
  return 0;
}
/*
 * PetscFEIntegrateJacobian_WY - Integrate the jacobian functions and perform the DoF lumping a la Wheeler and Yotov.
 */
static PetscErrorCode PetscFEIntegrateJacobian_WY(PetscDS ds,PetscFEJacobianType jtype,PetscInt fieldI,PetscInt fieldJ,PetscInt Ne,PetscFEGeom *cgeom,
                                                  const PetscScalar coefficients[],const PetscScalar coefficients_t[],PetscDS dsAux,
                                                  const PetscScalar coefficientsAux[],PetscReal t,PetscReal u_tshift,PetscScalar elemMat[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscFEIntegrateJacobian_Basic(ds,jtype,fieldI,fieldJ,Ne,cgeom,coefficients,coefficients_t,dsAux,coefficientsAux,t,
                                        u_tshift,elemMat);CHKERRQ(ierr);
  if (fieldJ==fieldI) {
    PetscInt eOffset =
      0,totDim,e,offsetI,offsetJ,dim,f,g,gDofMin,gDofMax,dGroupMin,dGroupMax,vStart,vEnd;
    PetscInt        nGroups,nDoFs;
    const PetscInt  *groupDofInd,*dofGroupInd;
    PetscTabulation *T;
    PetscFE         fieldFE;
    PetscDualSpace  dsp;
    PetscSection    groupDofSect,dofGroupSect;
    IS              group2Dof,dof2Group;
    DM              refdm;
    PetscScalar     *group2Constant,*group2ConstantInv,*constantProjections;
    PetscScalar     *tmpElemMat;

    ierr = PetscDSGetTotalDimension(ds,&totDim);CHKERRQ(ierr);
    ierr = PetscMalloc1(totDim*totDim,&tmpElemMat);CHKERRQ(ierr);
    ierr = PetscDSGetFieldOffset(ds,fieldI,&offsetI);CHKERRQ(ierr);
    ierr = PetscDSGetFieldOffset(ds,fieldJ,&offsetJ);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(ds,&T);CHKERRQ(ierr);
    ierr = PetscDSGetSpatialDimension(ds,&dim);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(ds,fieldI,(PetscObject*)&fieldFE);CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fieldFE,&dsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDM(dsp,&refdm);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(refdm,0,&vStart,&vEnd);CHKERRQ(ierr);

    ierr = DMSetField(refdm,0,NULL,(PetscObject)fieldFE);CHKERRQ(ierr);
    ierr = DMPlexGetStratumDofMap(refdm,0,0,&groupDofSect,&group2Dof);CHKERRQ(ierr);
    ierr = PetscSectionInvertMapping(groupDofSect,group2Dof,&dofGroupSect,&dof2Group);CHKERRQ(ierr);

    ierr = ISGetMinMax(group2Dof,&gDofMin,&gDofMax);CHKERRQ(ierr);
    ierr = ISGetMinMax(dof2Group,&dGroupMin,&dGroupMax);CHKERRQ(ierr);
    ierr = ISGetIndices(group2Dof,&groupDofInd);CHKERRQ(ierr);
    ierr = ISGetIndices(dof2Group,&dofGroupInd);CHKERRQ(ierr);

    nDoFs   = gDofMax - gDofMin + 1;   /*Possibly unnecessary. Maybe some place where this is better to use than current setup. */
    nGroups = dGroupMax - dGroupMin +1;

    /*dof_to_group // which corner group I'm in
      group_to_dof // covered by group DofSect/group2Dof/groupDofInd
      currently use a combination of PetscSection/ IS, and a work array to access IS entries (maybe better way)
      group_to_constants // numberofgroups x d x d: think of it as a matrix C for each corner, whiere
      C_{i,j} is the coefficient of constant function j in its representation on shape function for the ith
      dof in: the corner group

     For example in the vertDofInd [0, 7, 1, 2, 3, 4, 5, 6]
     That corresponds to

       +5-------4+
       6         3
       |         |
       |         |
       7         2
       +0-------1+

     the C_{0,7} block for group {0,7}

     [ 0 -1 ]
     [-1  0 ]

     and the C_{1,2} block for group {1,2} is

     [ 0 -1 ]
     [ 1  0 ]


     M[[0, 7],[1, 2]] has been filled, we have to add it in to M[[0,7],[0,7]],

     C_{0,7} C^{-1}_{1,2}

     [ 0  -1 ] [ 0  1 ] = [ 1  0 ]
     [ -1  0 ] [-1  0 ]   [ 0 -1 ]
     ** This means DoF 0 and DoF 1 are in the same direction, DoF 7 and DoF 2 are in opposing directions. And DoF 0 orth. to DoF 2, same for 7 and 1
     M[[0,7],[0,7]] += M[[0,7],[1,2]] C_{0,7} C^{-1}_{1,2}  */

    /* Build coefficient matrices using dualspace data. */
    ierr = PetscCalloc3(nGroups*dim*dim,&group2Constant,nGroups*dim*dim,&group2ConstantInv,nGroups*dim*dim,&constantProjections);CHKERRQ(ierr);
    ierr = PetscDualSpaceProjectConstants(dsp,constantProjections);CHKERRQ(ierr);

    for (g=0; g < nGroups; ++g) {
      PetscInt nPointsInGroup,nOffset,point,globalInd;
      ierr = PetscSectionGetDof(groupDofSect,g+vStart,&nPointsInGroup);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(groupDofSect,g+vStart,&nOffset);CHKERRQ(ierr);
      for (point = 0; point<nPointsInGroup; ++point) {
        PetscInt comp;
        globalInd = groupDofInd[nOffset+point];
        for (comp = 0; comp<dim; ++comp) group2Constant[g*dim*dim + point*dim + comp] = constantProjections[comp*nGroups*dim + globalInd];
      }
    }

    for (g=0; g< nGroups; ++g) {
      PetscScalar detJ;
      if (dim==2) {
        DMPlex_Det2D_Scalar_Internal(&detJ,&group2Constant[g*dim*dim]);
        DMPlex_Invert2D_Internal(&group2ConstantInv[g*dim*dim],&group2Constant[g*dim*dim],detJ);
      } else if (dim==3) {
        DMPlex_Det3D_Scalar_Internal(&detJ,&group2Constant[g*dim*dim]);
        DMPlex_Invert3D_Internal(&group2ConstantInv[g*dim*dim],&group2Constant[g*dim*dim],detJ);
      }
    }

    for (e=0; e < Ne; ++e) {
      PetscInt groupI;
      ierr = PetscArrayzero(tmpElemMat,totDim*totDim);CHKERRQ(ierr);
      /* ierr = PetscPrintf(PETSC_COMM_WORLD,"ELEMENT %d\n",e);CHKERRQ(ierr); */

      /* Applying the lumping and storing result in tmp array (may not need tmp any more) */
      for (groupI = dGroupMin; groupI <= dGroupMax; ++groupI) {
        /* Group I is our target (range) for the lumping. I.e. the block diagonal elements that the lumping will preserve*/
        PetscInt    gIOff,numGIDof,DoFI,groupJ;
        PetscScalar *constI;
        ierr   = PetscSectionGetOffset(groupDofSect,groupI,&gIOff);CHKERRQ(ierr);
        ierr   = PetscSectionGetDof(groupDofSect,groupI,&numGIDof);CHKERRQ(ierr);
        constI = &group2ConstantInv[(groupI-dGroupMin)*dim*dim];

        for (groupJ = dGroupMin; groupJ <= dGroupMax; ++groupJ) {
          /* Group J represents the source (or domain) of the lumping. The DoFs belonging to Group J are the values that we will be adding into
           * the DoFs of Group I. */
          PetscInt    gJOff,numGJDof,DoFK;
          PetscScalar *constJ;

          ierr   = PetscSectionGetOffset(groupDofSect,groupJ,&gJOff);CHKERRQ(ierr);
          ierr   = PetscSectionGetDof(groupDofSect,groupJ,&numGJDof);CHKERRQ(ierr);
          constJ = &group2Constant[(groupJ-dGroupMin)*dim*dim];

          for (DoFI = gIOff; DoFI < gIOff+numGIDof; ++DoFI) {
            /* For each dof in group I serving as a row index for both source and target values. */
            PetscInt DoFJ,rowInd = groupDofInd[DoFI];

            for (DoFJ = gIOff; DoFJ < gIOff+numGIDof; ++DoFJ) {
              /* For each dof in group I serving as a column index for the target value. */
              PetscInt colIndTarget = groupDofInd[DoFJ];

              for (DoFK = gJOff; DoFK < gJOff+numGJDof; ++DoFK) {
                PetscInt k,colIndSource = groupDofInd[DoFK];
                /* Now we need to perform the matrix multiplication M[[GroupI],[GroupI]] += M[[GroupI],[GroupJ]]*constI*constJ;*/
                for (k = 0; k<numGIDof; ++k) {
                  /* k is a temporary index to facilitate the matrix multiply*/
                  tmpElemMat[rowInd*totDim +  colIndTarget] += elemMat[eOffset + rowInd*totDim + colIndSource] *
                                                               constJ[k*numGIDof + (DoFK-gJOff)] * constI[(DoFJ-gIOff)*numGIDof + k];
                }
              }
            }
          }
        }
      }

      /* Move data from tmp array back to original now that we can safely overwrite*/
      for (f = 0; f < T[fieldI]->Nb; ++f) {
        const PetscInt i = offsetI + f;
        for (g = 0; g < T[fieldJ]->Nb; ++g) {
          const PetscInt j = offsetJ + g;
          elemMat[eOffset+i*totDim+j] = tmpElemMat[i*totDim + j];
        }
      }
      eOffset += PetscSqr(totDim);
    }
    ierr = PetscFree(tmpElemMat);CHKERRQ(ierr);
    ierr = PetscFree2(group2Constant,group2ConstantInv);CHKERRQ(ierr);
    ierr = ISRestoreIndices(group2Dof,&groupDofInd);CHKERRQ(ierr);
    ierr = ISRestoreIndices(dof2Group,&dofGroupInd);CHKERRQ(ierr);
    ierr = ISDestroy(&group2Dof);CHKERRQ(ierr);
    ierr = ISDestroy(&dof2Group);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&groupDofSect);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&dofGroupSect);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm,UserCtx * user,DM * mesh)
{
  PetscErrorCode   ierr;
  PetscRandom      ran;
  DM               dmDist = NULL;
  PetscPartitioner part;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(comm,&ran);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(ran,RANDOM_SEED);CHKERRQ(ierr);
  ierr = PetscRandomSeed(ran);CHKERRQ(ierr);
  /* Create a mesh (2D vs. 3D) and (simplex vs. tensor) as determined by */
  /* parameters */
  /* TODO: make either a simplex or tensor-product mesh */
  /* Desirable: a mesh with skewing element transforms that will stress the */
  /* Piola transformations involved in assembling H-div finite elements */
  /* Create box mesh from user parameters */
  ierr = DMPlexCreateBoxMesh(comm,user->dim,user->simplex,NULL,NULL,NULL,NULL,PETSC_TRUE,mesh);CHKERRQ(ierr);

  ierr = DMPlexGetPartitioner(*mesh,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh,0,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr  = DMDestroy(mesh);CHKERRQ(ierr);
    *mesh = dmDist;
  }
  ierr = DMLocalizeCoordinates(*mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *mesh,"Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*mesh,user);CHKERRQ(ierr);
  ierr = TransformMesh(user,mesh,&ran);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*mesh);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh,NULL,"-dm_view");CHKERRQ(ierr);

  ierr = DMDestroy(&dmDist);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&ran);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm,UserCtx * user)
{
  PetscDS        prob;
  PetscErrorCode ierr;
  PetscInt       id = 1;
  PetscInt       cmp = 1;
  PetscInt       dim=user->dim;

  PetscFunctionBegin;
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob,0,f0_v,f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,g0_vu,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,1,NULL,NULL,g2_vp,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,1,0,NULL,g1_qu,NULL,NULL);CHKERRQ(ierr);

  switch (user->sol_form) {
  case LINEAR:
    ierr = PetscDSSetResidual(prob,1,f0_q_linear,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_linear_bd_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,linear_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,linear_p,NULL);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Boundary","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    break;
  case QUADRATIC:
    ierr = PetscDSSetResidual(prob,1,f0_q_quad,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_quad_bd_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,quad_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,quad_p,NULL);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Boundary","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    break;
  case SINUSOIDAL:
    ierr = PetscDSSetResidual(prob,1,f0_q_sinusoid,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_sinusoid_bd_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,sinusoid_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,sinusoid_p,NULL);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Boundary","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    break;
  case SUBSURFACE_BENCHMARK:
    ierr = PetscDSSetResidual(prob,1,f0_q_subsurface,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_subsurface_bd_u,NULL);CHKERRQ(ierr);
    id   = 1;
    cmp  = 1;
    ierr = DMAddBoundary (dm,DM_BC_ESSENTIAL,"Bottom","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    id   = dim==3 ? 5 : 2;
    ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Right","marker",0,0,NULL,NULL,NULL,1,&id,user);CHKERRQ(ierr);
    id   = dim==3 ? 2 : 3;
    ierr = DMAddBoundary(dm,DM_BC_ESSENTIAL,"Top","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    id   = dim==3 ? 6 : 4;
    ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Left","marker",0,0,NULL,NULL,NULL,1,&id,user);CHKERRQ(ierr);
    if (dim==3) {
      id   = 3;
      ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Front","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
      id   = 4;
      ierr = DMAddBoundary(dm,DM_BC_NATURAL,"Back","marker",0,0,NULL,(void (*)(void))zero,NULL,1,&id,user);CHKERRQ(ierr);
    }
    break;
  default:
    PetscFunctionReturn(-1);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM mesh,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx * user,PetscBool useLumping)
{
  DM             cdm = mesh;
  PetscFE        fevel,fepres;
  const PetscInt dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFECreateDefault (PetscObjectComm((PetscObject)mesh),dim,dim,user->simplex,"velocity_",PETSC_DEFAULT,&fevel);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fevel,"velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,1,user->simplex,"pressure_",PETSC_DEFAULT,&fepres);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fepres,"pressure");CHKERRQ(ierr);

  ierr = PetscFECopyQuadrature(fevel,fepres);CHKERRQ(ierr);

  if (useLumping) fevel->ops->integratejacobian = PetscFEIntegrateJacobian_WY;

  ierr = DMSetField(mesh,0,NULL,(PetscObject)fevel);CHKERRQ(ierr);
  ierr = DMSetField(mesh,1,NULL,(PetscObject)fepres);CHKERRQ(ierr);
  ierr = DMCreateDS(mesh);CHKERRQ(ierr);
  ierr = (*setup)(mesh,user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(mesh,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm,&cdm);CHKERRQ(ierr);
  }

  ierr = PetscFEDestroy(&fevel);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fepres);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  UserCtx         user;
  DM              mesh,mesh_WY;
  SNES            snes,snes_WY;
  Mat             jacobian,jacobian_WY;
  Vec             u,b,u_WY,b_WY,exactSol,errVec,errVec_WY,resVec,resVec_WY;
  PetscSection    gSec,lSec;
  PetscReal       diffNorm,resNorm_exact;
  PetscReal       *fieldDiff, *fieldDiff_WY,*fieldResNorm,*fieldResNorm_WY;
  PetscBool       solutionWithinTol;
  const PetscReal tol = 100*PETSC_SQRT_MACHINE_EPSILON;
  PetscErrorCode (**exacts)(PetscInt dim,PetscReal t,const PetscReal* x,PetscInt Nc,PetscScalar *u,void *ctx);
  PetscDS             prob;
  PetscInt       Nf,f,nIt,nIt_WY;
  PetscViewer view;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);
  if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);

  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&mesh);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&mesh_WY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"====Mesh====\n");CHKERRQ(ierr);
  ierr = DMView(mesh,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n====Mesh_WY====\n");CHKERRQ(ierr);
  ierr = DMView(mesh_WY,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);
  ierr = SetupDiscretization(mesh,SetupProblem,&user,PETSC_FALSE);CHKERRQ(ierr);
  ierr = SetupDiscretization(mesh_WY,SetupProblem,&user,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(mesh,&user,&user,&user);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(mesh_WY,&user,&user,&user);CHKERRQ(ierr);
  ierr = DMGetDS(mesh,&prob);CHKERRQ(ierr);
  ierr = DMGetNumFields(mesh,&Nf);CHKERRQ(ierr);

  ierr = PetscMalloc(Nf,&exacts);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf,&fieldDiff,Nf,&fieldDiff_WY);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf,&fieldResNorm,Nf,&fieldResNorm_WY);CHKERRQ(ierr);
  for (f=0; f<Nf; ++f) {
    ierr = PetscDSGetExactSolution(prob,f,&exacts[f],NULL);CHKERRQ(ierr);
  }

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_WY);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,mesh);CHKERRQ(ierr);
  ierr = SNESSetDM(snes_WY,mesh_WY);CHKERRQ(ierr);
  ierr = SNESSetAlwaysComputesFinalResidual(snes,PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESSetAlwaysComputesFinalResidual(snes_WY,PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"A_");CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_WY,"WY_");CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_WY);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(mesh,&u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh_WY,&u_WY);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&b);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh_WY,&b_WY);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&errVec);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh_WY,&errVec_WY);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&resVec);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh_WY,&resVec_WY);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&exactSol);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(mesh,&gSec);CHKERRQ(ierr);
  ierr = DMGetLocalSection(mesh,&lSec);CHKERRQ(ierr);


  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(u_WY,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  ierr = VecSet(b_WY,0.0);CHKERRQ(ierr);
  ierr = VecSet(errVec,0.0);CHKERRQ(ierr);
  ierr = VecSet(errVec_WY,0.0);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"====A_Solve====\n");CHKERRQ(ierr);
  ierr = SNESSolve(snes,b,u);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n====WY_Solve====\n");CHKERRQ(ierr);
  ierr = SNESSolve(snes_WY,b_WY,u_WY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);

  ierr = SNESGetJacobian(snes,&jacobian,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes_WY,&jacobian_WY,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = MatViewFromOptions(jacobian,NULL,"-jacobian_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(jacobian_WY,NULL,"-jacobian_WY_view");CHKERRQ(ierr);

  ierr = DMComputeExactSolution(mesh,0,exactSol,NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(exactSol,NULL,"-exact_view");CHKERRQ(ierr);
  ierr = DMSNESCheckResidual(snes,mesh,exactSol,-1,&resNorm_exact);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXACT RESIDUAL: %14.12g\n",resNorm_exact);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&nIt);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes_WY,&nIt_WY);CHKERRQ(ierr);
  if (user.sol_form != SUBSURFACE_BENCHMARK){
    ierr = DMComputeL2FieldDiff(mesh,0.0,exacts,NULL,u,fieldDiff);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(mesh_WY,0.0,exacts,NULL,u_WY,fieldDiff_WY);CHKERRQ(ierr);
  }
  ierr = SNESGetFunction(snes,&resVec,NULL,NULL);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes_WY,&resVec_WY,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSectionVecNorm(lSec,gSec,resVec,NORM_2,fieldResNorm);CHKERRQ(ierr);
  ierr = PetscSectionVecNorm(lSec,gSec,resVec_WY,NORM_2,fieldResNorm_WY);CHKERRQ(ierr);

  if (user.toFile) {
    ierr = PetscViewerCreate(MPI_COMM_WORLD,&view);CHKERRQ(ierr);
    ierr = PetscViewerSetType(view,PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(view,FILE_MODE_APPEND);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(view,user.filename);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(view,"==== Refine Level: ====\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"Unmodified System: \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"%14.12g\t%14.12g\t%14.12g\t%14.12g\t%D\n",fieldDiff[0],fieldDiff[1],fieldResNorm[0],fieldResNorm[1],nIt);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"Lumped Field Diff: \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"%14.12g\t%14.12g\t%14.12g\t%14.12g\t%D\n",fieldDiff_WY[0],fieldDiff_WY[1],fieldResNorm_WY[0],fieldResNorm_WY[1],nIt_WY);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  }

  ierr = VecAXPY(u,-1,u_WY);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&diffNorm);CHKERRQ(ierr);
  if (user.showNorm) {
    ierr = PetscPrintf(MPI_COMM_WORLD,"Norm of solution difference: %g\n",diffNorm);CHKERRQ(ierr);
  }
  solutionWithinTol = (diffNorm <= tol);
  ierr              = PetscPrintf(MPI_COMM_WORLD,"Solutions are witin tolerance?: %s\n",solutionWithinTol ? "True" : "False");CHKERRQ(ierr);

  /* Tear down */
  ierr = PetscFree(exacts);CHKERRQ(ierr);
  ierr = PetscFree2(fieldResNorm,fieldResNorm_WY);CHKERRQ(ierr);
  ierr = PetscFree2(fieldDiff,fieldDiff_WY);CHKERRQ(ierr);
  ierr = VecDestroy(&b_WY);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u_WY);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes_WY);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh_WY);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
testset:
  suffix: 2d_bdm
  requires: triangle
  args: -dim 2 \
  -velocity_petscspace_degree 1 \
  -velocity_petscdualspace_type bdm \
  -velocity_petscdualspace_lagrange_node_endpoints true \
  -A_snes_converged_reason \
  -A_snes_linesearch_type basic \
  -A_snes_rtol 1e-10 \
  -A_snes_atol 1e-10 \
  -A_snes_stol 1e-10 \
  -A_snes_max_it 500 \
  -A_ksp_rtol 1e-12 \
  -A_pc_type fieldsplit \
  -A_pc_fieldsplit_type schur \
  -A_pc_fieldsplit_schur_precondition full \
  -WY_snes_converged_reason \
  -WY_snes_linesearch_type basic \
  -WY_snes_rtol 1e-10 \
  -WY_snes_atol 1e-10 \
  -WY_snes_stol 1e-10 \
  -WY_snes_max_it 500 \
  -WY_ksp_rtol 1e-12 \
  -WY_pc_type fieldsplit \
  -WY_pc_fieldsplit_type schur \
  -WY_pc_fieldsplit_schur_precondition full \
  -showNorm true 
  test:
    suffix: linear
    args: -sol_form linear -mesh_transform none
  test:
    suffix: quadratic
    args: -sol_form quadratic -mesh_transform none
  test: 
    suffix: sinusoidal
    args: -sol_form sinusoidal -mesh_transform none

testset:
  suffix: 2d_bdmq
  args: -dim 2 \
  -simplex false \
  -velocity_petscspace_degree 1 \
  -velocity_petscdualspace_type bdm \
  -velocity_petscdualspace_lagrange_tensor 1 \
  -velocity_petscdualspace_lagrange_node_endpoints true \
  -A_ksp_rtol 1e-12 \
  -WY_ksp_rtol 1e-12 \
  -A_pc_type fieldsplit \
  -WY_pc_type fieldsplit \
  -A_pc_fieldsplit_type schur \
  -WY_pc_fieldsplit_type schur \
  -A_pc_fieldsplit_schur_precondition full \
  -WY_pc_fieldsplit_schur_precondition full
  test:
    suffix: linear
    args: -sol_form linear -mesh_transform none
  test:
    suffix: quadratic
    args: -sol_form quadratic -mesh_transform none
  test:
    suffix: sinusoidal
    args: -sol_form sinusoidal -mesh_transform none

testset:
  suffix: 3d_bdm
  requires: triangle
  args: -dim 3 \
  -velocity_petscspace_degree 1 \
  -velocity_petscdualspace_type bdm
  test:
    suffix: linear
    args: -sol_form linear -mesh_transform none

# Test set for subsurface flow cases.
# Eventually we should stop abusing the test harness and break
# these out into dedicated executables or external scripts .
# Domain is a square, 205m side length with 41 cells per side.
# Need Neumann conditions on top and bottom of domain, dirichlet on sides.
testset:
  suffix: subsurface_benchmark
  args: -dim 2 \
    -simplex false \
    -velocity_petscspace_degree 1 \
    -velocity_petscdualspace_type bdm \
    -velocity_petscdualspace_lagrange_tensor 1 \
    -velocity_petscdualspace_lagrange_node_endpoints true \
    -A_ksp_rtol 1e-12 \ 
    -WY_ksp_rtol 1e-12 \
    -A_pc_type fieldsplit \
    -WY_pc_type fieldsplit \
    -A_pc_fieldsplit_type schur \
    -WY_pc_fieldsplit_type schur \
    -A_pc_fieldsplit_schur_precondition full \
    -WY_pc_fieldsplit_schur_precondition full \
    -A_snes_converged_reason \
    -WY_snes_converged_reason
  test: 
    suffix: chang3.1
    args: -dm_plex_box_lower -102.5,-102.5 \
      -dm_plex_box_upper 102.5,102.5 \
      -dm_plex_box_faces 41,41 \
      -dm_plex_box_interpolate true \
      -dm_plex_separate_marker \
      -sol_form subsurface_benchmark -mesh_transform none

TEST*/
