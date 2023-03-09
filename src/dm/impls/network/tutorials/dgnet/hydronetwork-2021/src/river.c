#include "wash.h"
/*
 Subroutines for River
*/

/* Compute SL and SR for computing flux in hll solver */
static inline PetscErrorCode hll(RiverField *xL, RiverField *xR, PetscScalar *SL, PetscScalar *SR)
{
  PetscScalar hS;
  PetscReal   tolh = 1.e-6;
  PetscScalar uL, uR, cL, cR;

  PetscFunctionBegin;
  uL = xL->q / xL->h;
  uR = xR->q / xR->h;

  cL = PetscSqrtScalar(GRAV * xL->h);
  cR = PetscSqrtScalar(GRAV * xR->h);

  *SL = *SR = 1.e10;
  /* compute SL and SR dry bed condtions Toro 10.79 and 10.80 */
  if (xL->h < tolh && xR->h > tolh) {
    /* dry bed on the left side*/
    *SL = uR - 2.0 * cR;
    *SR = uR + cR; // uR + cR*qR;
  }
  if (xR->h < tolh && xL->h > tolh) {
    /* dry bed on the right side */
    *SL = uL - cL; // uL - cL*qL;
    *SR = uL + 2.0 * cL;
  }
  if (xL->h < tolh && xR->h < tolh) {
    /* dry bed on the left and  right side */
    *SL -= tolh;
    *SR += tolh;
  }

  /* use two rarefaction solution to compute hs in the star region Toro 10.18 */
  hS = PetscSqr(0.5 * (cL + cR) + 0.25 * (uL - uR)) * (1 / GRAV);

  /* compute SL and SR wet bed condtions Toro 10.22 and 10.23 */
  if (hS < xL->h) {
    *SL = uL - cL;
  } else {
    *SL = uL - cL * PetscSqrtScalar(0.5 * hS * (hS + xL->h)) / xL->h;
  }

  if (hS < xR->h) {
    *SR = uR + cR;
  } else {
    *SR = uR + cR * PetscSqrtScalar(0.5 * hS * (hS + xR->h)) / xR->h;
  }
#if defined(PETSC_USE_DEBUG)
  PetscCheck(!PetscIsNanScalar(*SL) && !PetscIsNanScalar(*SR), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SL or SR cannot be NAN, xL->h %g, xR->h %g", xL->h, xR->h);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode ComputeFlux(RiverField *x, PetscReal tol, RiverField *flux)
{
  PetscFunctionBegin;
  if (x->h <= tol) {
    x->h    = tol;
    flux->h = 0.0;
    flux->q = 0.0;
  } else {
    flux->h = x->q; /* = h*u = h*(q/h) = q */
    flux->q = x->q * x->q / x->h + 0.5 * GRAV * x->h * x->h;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute HLL flux between xL and xR  */
static inline PetscErrorCode hllRiemannSolution(PetscInt cells, RiverField *xL, RiverField *xR, RiverField *flux)
{
  PetscErrorCode ierr;
  PetscScalar    SL, SR;
  RiverField     fluxL, fluxR;

  PetscFunctionBegin;
  if (xL->h <= 0.0 || xR->h <= 0.0) {
    flux->h = 0.0;
    flux->q = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* compute wave speeds SL and SR */
  ierr = hll(xL, xR, &SL, &SR);
  CHKERRQ(ierr);

  /* compute left and right fluxes */
  ierr = ComputeFlux(xL, 0.0, &fluxL);
  CHKERRQ(ierr);
  ierr = ComputeFlux(xR, 0.0, &fluxR);
  CHKERRQ(ierr);

  /* compute flux in star region Toro 10.21 */
  if (SL > 0.0) {
    /* Right-going supersonic flow */
    flux->h = fluxL.h;
    flux->q = fluxL.q;
  } else if (SR < 0.0) {
    /* Left-going supersonic flow */
    flux->h = fluxR.h;
    flux->q = fluxR.q;
  } else if (SL < 0.0 && SR > 0.0) {
    flux->h = (SR * fluxL.h - SL * fluxR.h + SR * SL * (xR->h - xL->h)) / (SR - SL);
    flux->q = (SR * fluxL.q - SL * fluxR.q + SR * SL * (xR->q - xL->q)) / (SR - SL);
  } else PetscCheck(0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SL %g SR %g", SL, SR);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode CorrectSlope(RiverField *x, PetscReal tol)
{
  PetscFunctionBegin;
  if (PetscAbsScalar(x->q) < tol) x->q = copysign(tol, x->q);
  if (PetscAbsScalar(x->h) < tol) x->h = copysign(tol, x->h);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  (1) compute slope (Ref: Toro 2001: eqn 11.22)
  (2) compute ratio of intercell slopes. Ref: Toro 2001: eqn 11.34
  (3) compute slope limiter using superbee for a test case. Ref: Toro 2001: eqn 11.34-11.36
  (4) compute boundary extrapolated values. Toro 2001: eqn 11.23
*/
static inline PetscErrorCode Computebexv(RiverField *x, RiverField *islopeL, RiverField *islopeR, RiverField *bexvL, RiverField *bexvR)
{
  PetscErrorCode ierr;
  PetscReal      omega = 0.0;
  PetscScalar    r, delta;

  PetscFunctionBegin;
  delta = 0.5 * (1.0 + omega) * islopeL->h + 0.5 * (1.0 - omega) * islopeR->h;
  r     = islopeL->h / islopeR->h;
  ierr  = SuperbeeLimiter(r, omega, &delta);
  CHKERRQ(ierr);
  delta *= 0.5;
  bexvL->h = x->h - delta;
  bexvR->h = x->h + delta;

  delta = 0.5 * (1.0 + omega) * islopeL->q + 0.5 * (1.0 - omega) * islopeR->q;
  r     = islopeL->q / islopeR->q;
  ierr  = SuperbeeLimiter(r, omega, &delta);
  CHKERRQ(ierr);
  delta *= 0.5;
  bexvL->q = x->q - delta;
  bexvR->q = x->q + delta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Reconstruct and compute bounday extrapolated values for cell[i] - cell[i+1]
   Input:
     xl1=x[i-1], x=x[i], xr1=x[i+1], xr2=x[i+2]
   Output:
     xL, xR
 */
static inline PetscErrorCode MusclHancokScheme(RiverField *xl1, RiverField *x, RiverField *xr1, RiverField *xr2, PetscReal dx, PetscReal dt, RiverField *xL, RiverField *xR)
{
  PetscErrorCode ierr;
  PetscReal      tol = 1.e-6, r = 0.5 * dt / dx;
  RiverField     islopeL, islopeR, bexvL, bexvR, fluxL, fluxR;

  PetscFunctionBegin;
  /* xL: */
  /* compute difference of neighbouring conserved variables Toro 11.22 */
  islopeL.h = x->h - xl1->h;
  islopeL.q = x->q - xl1->q;

  islopeR.h = xr1->h - x->h;
  islopeR.q = xr1->q - x->q;

  /* correct small difference of conserved variables */
  ierr = CorrectSlope(&islopeL, tol);
  CHKERRQ(ierr);
  ierr = CorrectSlope(&islopeR, tol);
  CHKERRQ(ierr);

  ierr = Computebexv(x, &islopeL, &islopeR, &bexvL, &bexvR);
  CHKERRQ(ierr);

  ierr = ComputeFlux(&bexvL, tol, &fluxL);
  CHKERRQ(ierr);
  ierr = ComputeFlux(&bexvR, tol, &fluxR);
  CHKERRQ(ierr);

  /* evolve boundary extrapolated values Ref: Toro 2001: eqn 11.24 */
  xL->h = bexvR.h + r * (fluxL.h - fluxR.h);
  xL->q = bexvR.q + r * (fluxL.q - fluxR.q);

  /* xR: */
  /* compute difference of neighbouring conserved variables Toro 11.22 */
  islopeL.h = xr1->h - x->h;
  islopeL.q = xr1->q - x->q;

  islopeR.h = xr2->h - xr1->h;
  islopeR.q = xr2->q - xr1->q;

  /*correct small difference of conserved variables */
  ierr = CorrectSlope(&islopeL, tol);
  CHKERRQ(ierr);
  ierr = CorrectSlope(&islopeR, tol);
  CHKERRQ(ierr);

  ierr = Computebexv(xr1, &islopeL, &islopeR, &bexvL, &bexvR);
  CHKERRQ(ierr);

  ierr = ComputeFlux(&bexvL, tol, &fluxL);
  CHKERRQ(ierr);
  ierr = ComputeFlux(&bexvR, tol, &fluxR);
  CHKERRQ(ierr);

  /* evolve boundary extrapolated values Ref: Toro 2001: eqn 11.24 */
  xR->h = bexvL.h + r * (fluxL.h - fluxR.h);
  xR->q = bexvL.q + r * (fluxL.q - fluxR.q);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolutionStatsView(DM da, Vec X, PetscViewer viewer)
{
  PetscErrorCode     ierr;
  PetscReal          xmin, xmax;
  PetscScalar        sum, tvsum, tvgsum;
  const PetscScalar *x;
  PetscInt           imin, imax, Mx, i, j, xs, xm, dof;
  Vec                Xloc;
  PetscBool          iascii;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);
  CHKERRQ(ierr);
  if (iascii) {
    /* PETSc lacks a function to compute total variation norm (difficult in multiple dimensions), we do it here */
    ierr = DMGetLocalVector(da, &Xloc);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, Xloc);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, Xloc);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, Xloc, (void *)&x);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0);
    CHKERRQ(ierr);
    ierr = DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, &dof, 0, 0, 0, 0, 0);
    CHKERRQ(ierr);
    tvsum = 0;
    for (i = xs; i < xs + xm; i++) {
      for (j = 0; j < dof; j++) tvsum += PetscAbsScalar(x[i * dof + j] - x[(i - 1) * dof + j]);
    }
    ierr = MPI_Allreduce(&tvsum, &tvgsum, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)da));
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da, Xloc, (void *)&x);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Xloc);
    CHKERRQ(ierr);

    ierr = VecMin(X, &imin, &xmin);
    CHKERRQ(ierr);
    ierr = VecMax(X, &imax, &xmax);
    CHKERRQ(ierr);
    ierr = VecSum(X, &sum);
    CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Solution range [%8.5f,%8.5f] with extrema at %d and %d, mean %8.5f, ||x||_TV %8.5f\n", (double)xmin, (double)xmax, imin, imax, (double)(sum / Mx), (double)(tvgsum / Mx));
    CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, 1, "Viewer type not supported");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Get the time step dt for a river
*/
PetscErrorCode RiverGetTimeStep(River river, RiverField *x, PetscReal *dt)
{
  PetscInt  i;
  PetscReal CFL = 0.9, lambdaMax = 0, lambda = 0.0, dx = river->length / river->ncells;

  PetscFunctionBegin;
  /* CFL stability condition
     compute maximum eigenvalue for the courant condition */
  for (i = 0; i < river->ncells; i++) {
    if (x[i].h < 1.e-8) continue;
    lambda = PetscAbsReal(x[i].q / x[i].h) + PetscSqrtScalar(GRAV * x[i].h);
    if (lambda > lambdaMax) lambdaMax = lambda;
  }

  *dt = CFL * dx / lambdaMax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Set ncells based on cn = a*dt/dx = a*dt*ncells/length <= 1.0
   Input:
     river, dt - approximate dt for this river
   Output:
     ncells is for this river
*/
PetscErrorCode RiverSetNumCells(River river, PetscReal dt)
{
  PetscReal   lambdamax = 10.0;      /* a=1200.0: wave speed (to be obtained) */
  PetscScalar maxQ      = river->q0; /* initial flow velocity*/

  PetscFunctionBegin;
  if (maxQ > 0) lambdamax = GRAV / (1.5 * maxQ); /*(celerity = v0(1+1/Fr) & Fr = 2 supercritical assumed*/

  if (dt <= 0.0) dt = 0.05; /* default */
  river->ncells = ceil(river->length / (lambdamax * dt));
  if (river->ncells == 1) river->ncells = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute initial solution */
PetscErrorCode RiverSetInitialSolution(PetscInt riverCase, PetscInt riverSubCase, River river, RiverField *x, PetscReal q0, PetscReal h0)
{
  PetscInt i, ncells = river->ncells;

  PetscFunctionBeginUser;
  if (riverCase == -1) { /* intial q and h overwritten from a file */
    for (i = 0; i < ncells; i++) {
      x[i].q = q0;
      x[i].h = h0;
    }
  } else if (riverCase > -1 && riverCase < 3) { /* case 0 to case 2 */
    for (i = 0; i < ncells; i++) {
      x[i].h = 3.0;
      x[i].q = 0.0;
    }
  } else if (riverCase == 3) {
    /* See Table 7.1 page 120 in Shock-Capturing Methods for Free-Surface Shallow Flows, E.F. Toro */
    /* ncells = 500, q = hu */
    PetscReal dx = river->length / ncells;

    switch (riverSubCase) {
    case 1:
      for (i = 0; i < ncells; i++) {
        if (i * dx < 10.0) {
          x[i].h = 1.0;
          x[i].q = 2.5;
        } else {
          x[i].h = 0.1;
          x[i].q = 0.0;
        }
      }
      break;
    case 2:
      for (i = 0; i < ncells; i++) {
        if (i * dx < 25.0) {
          x[i].q = -5.0;
        } else x[i].q = 5.0;
        x[i].h = 1.0;
      }
      break;
    case 3:
      for (i = 0; i < ncells; i++) {
        if (i * dx < 20.0) {
          x[i].h = 1.0;
        } else x[i].h = 0.0;
        x[i].q = 0.0;
      }
      break;
    case 4:
      for (i = 0; i < ncells; i++) {
        if (i * dx < 30.0) {
          x[i].h = 0.0;
        } else x[i].h = 1.0;
        x[i].q = 0.0;
      }
      break;
    case 5:
      for (i = 0; i < ncells; i++) {
        if (i * dx < 25.0) {
          x[i].q = -0.3;
        } else x[i].q = 0.3;
        x[i].h = 0.1;
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "riverCase is not supported yet");
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiverSetUp(River river)
{
  PetscErrorCode ierr;
  PetscReal      dx, length = river->length;

  PetscFunctionBegin;
  /* Create a DMDA to manage the parallel grid */
  river->comm = PETSC_COMM_SELF;
  ierr        = DMDACreate1d(river->comm, DM_BOUNDARY_GHOSTED, river->ncells, 2, 1, NULL, &river->da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(river->da);
  CHKERRQ(ierr);
  ierr = DMSetUp(river->da);
  CHKERRQ(ierr);

  ierr = DMDASetFieldName(river->da, 0, "h");
  CHKERRQ(ierr);
  ierr = DMDASetFieldName(river->da, 1, "Q");
  CHKERRQ(ierr);

  /* Set coordinates of cell centers */
  dx   = length / river->ncells;
  ierr = DMDASetUniformCoordinates(river->da, 0.5 * dx, length + 0.5 * dx, 0, 0, 0, 0);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate RHSFunction at interior cell points using the MUSCL-Hancock Scheme */
PetscErrorCode RiverRHSFunctionLocal(River river, RiverField *x, RiverField *f)
{
  PetscErrorCode ierr;
  PetscInt       i, ncells = river->ncells;
  PetscReal      dx = river->length / ncells, dt = river->dt;
  RiverField     flux, xL, xR;

  PetscFunctionBegin;
  /* Calculate numerical fluxes at interface i+1/2 */
  for (i = 0; i < ncells - 1; i++) {
    /* compute xL and xR at each interface (i,i+1) -- needs 4 cells at i=i-1,i,i+1,i+2 */
    /* Reconstruct and compute bounday extrapolated values for cell[i] - cell[i+1] */
    //printf("cell %d, xL: %g %g; xR: %g %g\n",i,x[i].q,x[i].h,x[i+1].q,x[i+1].h);
    if (i == 0) {
      /* mirror: x[i-1] = x[i+1] */
      ierr = MusclHancokScheme(&x[i + 1], &x[i], &x[i + 1], &x[i + 2], dx, dt, &xL, &xR);
      CHKERRQ(ierr);
    } else if (i == ncells - 2) {
      /* mirror: x[i+2] = x[i] */
      ierr = MusclHancokScheme(&x[i - 1], &x[i], &x[i + 1], &x[i], dx, dt, &xL, &xR);
      CHKERRQ(ierr);
    } else {
      ierr = MusclHancokScheme(&x[i - 1], &x[i], &x[i + 1], &x[i + 2], dx, dt, &xL, &xR);
      CHKERRQ(ierr);
    }
    //printf("cell %d, xL: %g %g; xR: %g %g\n\n",i,xL.q,xL.h,xR.q,xR.h);

    /* compute flux using hll approximate Reimann solver */
    ierr = hllRiemannSolution(i, &xL, &xR, &flux);
    CHKERRQ(ierr);

    if (i < ncells - 2) {
      f[i + 1].q += flux.q / dx;
      f[i + 1].h += flux.h / dx;
    }

    if (i) {
      f[i].q -= flux.q / dx;
      f[i].h -= flux.h / dx;
    }

    if (i == 0) {              /* flux to be applied to the Junction, save them to the river struct */
      river->flux[0] = flux.q; /* flux at upper stream interface 1/2 */
      river->flux[1] = flux.h;
    } else if (i == ncells - 2) {
      river->flux[2] = flux.q; /* flux at downstream interface ncells-1/2 */
      river->flux[3] = flux.h;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode RiverDestroyJacobian(River river)
{
  PetscErrorCode ierr;
  Mat           *Jriver = river->jacobian;
  PetscInt       i;

  PetscFunctionBegin;
  if (Jriver) {
    for (i = 0; i < 3; i++) {
      ierr = MatDestroy(&Jriver[i]);
      CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(Jriver);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    RiverCreateJacobian - Create Jacobian matrix structures for a River.

    Collective on River

    Input Parameter:
+   river - the River object
-   Jin - array of three constructed Jacobian matrices to be reused. Set NULL if it is not available

    Output Parameter:
.   J  - array of three empty Jacobian matrices

    Level: beginner
*/
PetscErrorCode RiverCreateJacobian(River river, Mat *Jin, Mat *J[])
{
  PetscErrorCode ierr;
  Mat            J0, *Jriver;
  PetscInt       i, M, rows[2], cols[2], *nz;
  PetscScalar   *aa;

  PetscFunctionBegin;
  if (Jin) {
    *J              = Jin;
    river->jacobian = Jin;
    ierr            = PetscObjectReference((PetscObject)(Jin[0]));
    CHKERRQ(ierr);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ierr = PetscMalloc1(3, &Jriver);
  CHKERRQ(ierr);

  /* Jacobian for this river: diagonal 2x2 block matrix */
#if 0
  //ierr = DMSetMatrixStructureOnly(river->da,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(river->da,&Jriver[0]);CHKERRQ(ierr);
  //printf("Jriver[0]:\n");
  //ierr = MatView(Jriver[0],0);CHKERRQ(ierr);
  //ierr = DMSetMatrixStructureOnly(river->da,PETSC_FALSE);CHKERRQ(ierr);
#endif
  M    = 2 * river->ncells;
  ierr = PetscCalloc2(M, &nz, 4, &aa);
  CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF, &J0);
  CHKERRQ(ierr);
  ierr = MatSetSizes(J0, PETSC_DECIDE, PETSC_DECIDE, M, M);
  CHKERRQ(ierr);
  ierr = MatSetBlockSize(J0, 2);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(J0);
  CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J0, 2, NULL);
  CHKERRQ(ierr);
  for (i = 0; i < river->ncells; i++) { /* set diagonal entries */
    rows[0] = 2 * i;
    rows[1] = 2 * i + 1;
    ierr    = MatSetValues(J0, 2, rows, 2, rows, aa, INSERT_VALUES);
    CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J0, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J0, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  Jriver[0] = J0;

  /* Jacobian for upstream vertex */
  ierr = MatCreate(PETSC_COMM_SELF, &Jriver[1]);
  CHKERRQ(ierr);
  ierr = MatSetSizes(Jriver[1], PETSC_DECIDE, PETSC_DECIDE, M, 2);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jriver[1]);
  CHKERRQ(ierr);
  //ierr = MatSetOption(Jriver[1],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
  nz[0]   = 2;
  nz[1]   = 2;
  rows[0] = 0;
  rows[1] = 1;
  ierr    = MatSeqAIJSetPreallocation(Jriver[1], 0, nz);
  CHKERRQ(ierr);
  ierr = MatSetValues(Jriver[1], 2, rows, 2, rows, aa, INSERT_VALUES);
  CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jriver[1], MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jriver[1], MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  /* Jacobian for downstream vertex */
  ierr = MatCreate(PETSC_COMM_SELF, &Jriver[2]);
  CHKERRQ(ierr);
  ierr = MatSetSizes(Jriver[2], PETSC_DECIDE, PETSC_DECIDE, M, 2);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jriver[2]);
  CHKERRQ(ierr);
  //ierr = MatSetOption(Jriver[2],MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);
  nz[0]     = 0;
  nz[1]     = 0;
  nz[M - 2] = 2;
  nz[M - 1] = 2;
  rows[0]   = M - 2;
  rows[1]   = M - 1;
  cols[0]   = 0;
  cols[1]   = 1;
  ierr      = MatSeqAIJSetPreallocation(Jriver[2], 0, nz);
  CHKERRQ(ierr);
  ierr = MatSetValues(Jriver[2], 2, rows, 2, cols, aa, INSERT_VALUES);
  CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jriver[2], MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jriver[2], MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  ierr = PetscFree2(nz, aa);
  CHKERRQ(ierr);

  *J              = Jriver;
  river->jacobian = Jriver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute hydrostatic reconstruction  */
PetscErrorCode hydroRecreate(PetscScalar hL, PetscScalar hR, PetscScalar zL, PetscScalar zR, PetscScalar *zMax, PetscScalar *hLrec, PetscScalar *hRrec)
{
  PetscFunctionBegin;
  *zMax  = PetscMax(zL, zR);
  *hLrec = PetscMax(0., hL + zL - *zMax);
  *hRrec = PetscMax(0., hR + zR - *zMax);
  //printf("hL %g, hR %g, zL %g, zR %g, zMax %g, hLrec %g, hRrec %g\n",hL,hR,zL,zR,*zMax,*hLrec,*hRrec);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   RiverCleanup - Free memory spaces in River object.

   Input Parameters:
   river - .
*/
PetscErrorCode RiverCleanup(River river)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!river) PetscFunctionReturn(PETSC_SUCCESS);

  ierr = DMDestroy(&river->da);
  CHKERRQ(ierr);
  ierr = RiverDestroyJacobian(river);
  CHKERRQ(ierr);
  ierr = PetscFree2(river->z, river->zMax);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiverSetParameters(River river, PetscInt id)
{
  PetscFunctionBegin;
  river->id = id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set  bed elevation*/
PetscErrorCode RiverSetElevation(River river, PetscReal zUs, PetscReal zDs)
{
  PetscErrorCode ierr;
  PetscReal      dx, slope;
  PetscInt       i, ncells = river->ncells;

  PetscFunctionBegin;
  ierr = PetscMalloc2(ncells, &river->z, ncells + 1, &river->zMax);
  CHKERRQ(ierr); /* allocate memory */
  dx    = river->length / ncells;
  slope = (zDs - zUs) / river->length;
  for (i = 0; i < ncells; i++) {
    /* zUs and zDs upsteam and downstream elevations queried from vertexes*/
    river->z[i] = zUs + i * dx * slope; /* zUs and zDs are elevations at US and DS and are queried from vertexes*/
    //printf("cell %g, z[i] %g, zUs %g, zDs %g, slope %g\n",(i+dx)*0.5,river->z[i],zUs,zDs,slope);
  }

  for (i = 1; i < ncells; i++) river->zMax[i] = PetscMax(river->z[i - 1], river->z[i]);
  river->zMax[0]      = river->z[0];          /* assume upper stream junction has same elevation as river->z[0] */
  river->zMax[ncells] = river->z[ncells - 1]; /* assume down stream junction has same elevation as river->z[ncells-1] */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute  froud number*/
PetscErrorCode FroudNumber(PetscScalar u, PetscScalar h, PetscScalar *Fr)
{
  PetscFunctionBegin;
  *Fr = u / PetscSqrtReal(GRAV * h);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute slope limiter using superbee */
PetscErrorCode SuperbeeLimiter(PetscScalar r, PetscScalar omega, PetscScalar *delta)
{
  PetscScalar xi = 0.0, xiR;

  PetscFunctionBegin;
  /* Toro 2001: eqns 11.34 and 11.36 */
  if (r > 0.0 && r <= 0.5) {
    xi = 2.0 * r;
  } else if (r > 0.5 && r <= 1.0) {
    xi = 1.0;
  } else if (r > 1.0) {
    xiR = 2.0 / (1.0 - omega + (1.0 + omega) * r); //2.0*r/(1.0-omega+(1.0+omega)*r);
    xi  = PetscMin(r, xiR);
    xi  = PetscMin(xi, 2.0);
  } //else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"r %g is not allowed",r); r= -1.0???

  *delta = xi * (*delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute slope limiter using Van Leer*/
PetscErrorCode VanLeerLimiter(PetscScalar r, PetscScalar omega, PetscScalar *delta)
{
  PetscScalar xi, xiR;

  PetscFunctionBegin;
  /* Toro 2001: eqns 11.34 and 11.37*/
  if (r > 0.0) {
    xiR = 2.0 * r / (1.0 - omega + (1.0 + omega) * r);
    xi  = 2.0 * r / (1.0 + r);
    xi  = PetscMin(xi, xiR);
  } else {
    xi = 0.0;
  }

  *delta = xi * (*delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute slope limiter using Van Albada*/
PetscErrorCode VanAlbadaLimiter(PetscScalar r, PetscScalar omega, PetscScalar *delta)
{
  PetscScalar xi, xiR;

  PetscFunctionBegin;
  /* Toro 2001: eqns 11.34 and 11.38*/
  if (r > 0.0) {
    xiR = 2.0 * r / (1.0 - omega + (1.0 + omega) * r);
    xi  = r * (1.0 + r) / (1.0 + r * r);
    xi  = PetscMin(xi, xiR);
  } else {
    xi = 0.0;
  }

  *delta = xi * (*delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute slope limiter using Minmod*/
PetscErrorCode MinModLimiter(PetscScalar r, PetscScalar omega, PetscScalar *delta)
{
  PetscScalar xi, xiR;

  PetscFunctionBegin;
  /* Toro 2001: eqns 11.34 and 11.39*/
  if (r > 0.0 && r <= 1.0) {
    xi = 1.0;
  } else if (r > 1.0) {
    xiR = 2.0 / (1.0 - omega + (1.0 + omega) * r);
    xi  = PetscMin(1.0, xiR);
  } else {
    xi = 0.0;
  }

  *delta = xi * (*delta);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiverView(Vec X, DM networkdm, Wash wash)
{
  PetscErrorCode ierr;
  River          river;
  PetscInt       key, Start, End;
  PetscMPIInt    rank;
  PetscInt       nx, nnodes, nidx, *idx1, *idx2, *idx1_h, *idx2_h, idx_start, i, k, k1, xstart, j1;
  Vec            Xq, Xh, localX;
  IS             is1_q, is2_q, is1_h, is2_h;
  VecScatter     ctx_q, ctx_h;
  PetscScalar    Q, H, U;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);

  /* get num of local and global total nnodes */
  nidx = wash->nnodes_loc;
  ierr = MPIU_Allreduce(&nidx, &nx, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Xq);
  CHKERRQ(ierr);
  if (rank == 0) { /* all entries of Xq are in proc[0] */
    ierr = VecSetSizes(Xq, nx, PETSC_DECIDE);
    CHKERRQ(ierr);
  } else {
    ierr = VecSetSizes(Xq, 0, PETSC_DECIDE);
    CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(Xq);
  CHKERRQ(ierr);
  ierr = VecSet(Xq, 0.0);
  CHKERRQ(ierr);
  ierr = VecDuplicate(Xq, &Xh);
  CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm, &localX);
  CHKERRQ(ierr);

  /* set idx1 and idx2 */
  ierr = PetscCalloc4(nidx, &idx1, nidx, &idx2, nidx, &idx1_h, nidx, &idx2_h);
  CHKERRQ(ierr);

  ierr = DMNetworkGetEdgeRange(networkdm, &Start, &End);
  CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(X, &xstart, NULL);
  CHKERRQ(ierr);
  k1 = 0;
  j1 = 0;
  for (i = Start; i < End; i++) {
    ierr = DMNetworkGetComponent(networkdm, i, 0, &key, (void **)&river, NULL);
    CHKERRQ(ierr);
    nnodes    = river->ncells;
    idx_start = river->id * nnodes;
    for (k = 0; k < nnodes; k++) {
      idx1[k1] = xstart + j1 * 2 * nnodes + 2 * k;
      idx2[k1] = idx_start + k;

      idx1_h[k1] = xstart + j1 * 2 * nnodes + 2 * k + 1;
      idx2_h[k1] = idx_start + k;
      k1++;
    }
    j1++;
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF, nidx, idx1, PETSC_COPY_VALUES, &is1_q);
  CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nidx, idx2, PETSC_COPY_VALUES, &is2_q);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(X, is1_q, Xq, is2_q, &ctx_q);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx_q, X, Xq, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx_q, X, Xq, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, nidx, idx1_h, PETSC_COPY_VALUES, &is1_h);
  CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nidx, idx2_h, PETSC_COPY_VALUES, &is2_h);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(X, is1_h, Xh, is2_h, &ctx_h);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx_h, X, Xh, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx_h, X, Xh, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
#if 0
  if (!rank) printf("Xq: \n");
  ierr = VecView(Xq,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (!rank) printf("Xh: \n");
  ierr = VecView(Xh,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  if (!rank) {
    PetscScalar *q_arr, *h_arr;
    printf("    i       Q           H        U \n");
    ierr = VecGetArray(Xq, &q_arr);
    CHKERRQ(ierr);
    ierr = VecGetArray(Xh, &h_arr);
    CHKERRQ(ierr);
    for (i = 0; i < nx; i++) {
      Q = q_arr[i];
      H = h_arr[i];
      if (H == 0.0) {
        U = 0.0;
      } else {
        U = q_arr[i] / h_arr[i];
      }
      printf("  %d    %g    %g   %g\n", i, Q, H, U);
      //printf("  %g    %g   %g\n",q_arr[i],h_arr[i],q_arr[i]/h_arr[i]);
    }
    ierr = VecRestoreArray(Xq, &q_arr);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(Xh, &h_arr);
    CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&ctx_q);
  CHKERRQ(ierr);
  ierr = PetscFree4(idx1, idx2, idx1_h, idx2_h);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is1_q);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is2_q);
  CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ctx_h);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is1_h);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is2_h);
  CHKERRQ(ierr);

  ierr = VecDestroy(&Xq);
  CHKERRQ(ierr);
  ierr = VecDestroy(&Xh);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm, &localX);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}
