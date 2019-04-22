static char help[] = "Vlasov example of particles orbiting around a central massive point.\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscts.h>
#include <petscmath.h>
typedef struct {
  PetscInt       dim;                              /* The topological mesh dimension */
  PetscInt       nts;                              /* print the energy at each nts time steps */
  PetscBool      simplex;                          /* Flag for simplices or tensor cells */
  PetscBool      monitor;                          /* Flag for use of the TS monitor */
  PetscBool      uniform;
  char           meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt       faces;                            /* Number of faces per edge if unit square/cube generated */
  PetscReal      domain_lo[3], domain_hi[3];       /* Lower left and upper right mesh corners */
  PetscReal omega;                                 /* Oscillation value omega */
  DMBoundaryType boundary[3];                      /* The domain boundary type, e.g. periodic */
  PetscInt       particlesPerCell;                 /* The number of partices per cell */
  PetscReal      particleRelDx;                    /* Relative particle position perturbation compared to average cell diameter h */
  PetscReal      meshRelDx;                        /* Relative vertex position perturbation compared to average cell diameter h */
  PetscInt       k;                                /* Mode number for test function */
  PetscReal      momentTol;                        /* Tolerance for checking moment conservation */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  /* Poisson solve */
  SNES           snes;
  PetscInt       steps;                            /* steps to take in ts */
} AppCtx;

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d]/(ctx->domain_hi[d] - ctx->domain_lo[d]);
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 1;
  for (d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d])*PetscSqr(ctx->domain_hi[d]) - PetscPowRealInt(x[d], 4);
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx *) a_ctx;

  u[0] = sin(2*PETSC_PI*ctx->k*x[0]/(ctx->domain_hi[0] - ctx->domain_lo[0]));
  return 0;
}



static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       ii, bd;
  char           fstring[PETSC_MAX_PATH_LEN] = "linear";
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->monitor          = PETSC_TRUE;
  options->faces            = 1;
  options->domain_lo[0]     = 0.0;
  options->domain_lo[1]     = -1.0;
  options->domain_lo[2]     = -1.0;
  options->domain_hi[0]     = 2*PETSC_PI;
  options->domain_hi[1]     = 1.0;
  options->domain_hi[2]     = 1.0;
  options->boundary[0]      = DM_BOUNDARY_PERIODIC; /* PERIODIC (plotting does not work in parallel, moments not conserved) */
  options->boundary[1]      = DM_BOUNDARY_NONE;
  options->boundary[2]      = DM_BOUNDARY_NONE;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->meshRelDx        = 0.0;
  options->momentTol        = 100.*PETSC_MACHINE_EPSILON;
  options->omega            = 64.;
  options->nts              = 100;
  options->uniform          = PETSC_FALSE;
  options->steps            = 1;

  ierr = PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");CHKERRQ(ierr);
  
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-next_output","time steps for next output point","<100>",options->nts,&options->nts,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex5.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps", "TS steps to take", "ex7.c", options->steps, &options->steps, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-monitor", "To use the TS monitor or not", "ex5.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex5.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uniform", "Uniform particle spacing", "ex5.c", options->uniform, &options->uniform, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex5.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-faces", "Number of faces per edge if unit square/cube generated", "ex5.c", options->faces, &options->faces, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex5.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex5.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-omega","parameter","<64>",options->omega,&options->omega,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex5.c", options->particleRelDx, &options->particleRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex5.c", options->meshRelDx, &options->meshRelDx, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex5.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex5.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  bd = options->boundary[0];
  ierr = PetscOptionsEList("-x_boundary", "The x-boundary", "ex5.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[0]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[0] = (DMBoundaryType) bd;
  bd = options->boundary[1];
  ierr = PetscOptionsEList("-y_boundary", "The y-boundary", "ex5.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[1]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[1] = (DMBoundaryType) bd;
  bd = options->boundary[2];
  ierr = PetscOptionsEList("-z_boundary", "The z-boundary", "ex5.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[2]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[2] = (DMBoundaryType) bd;
  ierr = PetscOptionsString("-function", "Name of test function", "ex5.c", fstring, fstring, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscStrcmp(fstring, "linear", &flag);CHKERRQ(ierr);
  if (flag) {
    options->func = linear;
  } else {
    ierr = PetscStrcmp(fstring, "sin", &flag);CHKERRQ(ierr);
    if (flag) {
      options->func = sinx;
    } else {
      ierr = PetscStrcmp(fstring, "x2_x4", &flag);CHKERRQ(ierr);
      options->func = x2_x4;
      if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown function %s",fstring);
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode PerturbVertices(DM dm, AppCtx *user)
{
  PetscRandom    rnd;
  PetscReal      interval = user->meshRelDx;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscReal      hh[3];
  PetscInt       d, cdim, N, p, bs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (d = 0; d < user->dim; ++d) hh[d] = (user->domain_hi[d] - user->domain_lo[d])/user->faces;
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  if (bs != cdim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Coordinate vector has wrong block size %D != %D", bs, cdim);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (p = 0; p < N; p += cdim) {
    PetscScalar *coord = &coords[p], value;

    for (d = 0; d < cdim; ++d) {
      ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
      coord[d] = PetscMax(user->domain_lo[d], PetscMin(user->domain_hi[d], coord[d] + value*hh[d]));
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshFilename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInt faces[3];

    faces[0] = user->faces; faces[1] = 1; faces[2] = 1;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, user->domain_lo, user->domain_hi, user->boundary, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshFilename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  //ierr = PerturbVertices(*dm, user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void laplacian_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {f1[d] = u_x[d];}
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static void laplacian(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {g3[d*dim+d] = 1.0;}
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscDS        prob;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, NULL, laplacian_f1);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, laplacian);CHKERRQ(ierr);

  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscRandom    rnd, rndp;
  PetscReal      interval = user->particleRelDx;
  PetscScalar    value, *vals;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ, *initialConditions;
  PetscInt      *cellid, cStart;
  PetscInt       Ncell, Np = user->particlesPerCell, p, c, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);

  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, 0.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rndp, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rndp);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &Ncell);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);

  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  /* simplices would need different handling, but two stream would not work without tensor cells */
  for (c = cStart; c < Ncell; c++) {
    /* handle this differently in the future */
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      cellid[c] = c;
      /* Place particles at centroid of each cell, which will be in a contiguous block linearly along x
      then have them moved evenly across the x axis using the cell width */
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3], spacing;
        
        cellid[n] = c;
        
        if(user->uniform){
          spacing = 2./Np;
          
          for(d=0; d<dim; ++d) refcoords[d] = d == 0 ? -1. + spacing/2. + p*spacing : 0.;

        }
        else{
          for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr); refcoords[d] = d == 0 ? PetscRealPart(value) : 0. ;}
        }
        vals[n] = 0.0;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
        /* constant particle weights */
        for (d = 0; d < dim; ++d) vals[n] = 1.;
      }
    }
  }
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    for (p = 0; p < Np; ++p) {
      if(c < Ncell/2){
        for (d = 0; d < dim; ++d) initialConditions[(c*Np + p)*dim + d] = d == 0 ? 1. : 0.;
      } 
      else {
        for (d = 0; d < dim; ++d) initialConditions[(c*Np + p)*dim + d] = d == 0 ? -1. : 0.;
      }
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rndp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec V,Vec Posres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *posres;
  PetscInt          Np, p, dim, d;
  DM                dm;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(Posres, &Np);CHKERRQ(ierr);
  ierr = VecGetArray(Posres,&posres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Np  /= dim;

  for (p = 0; p < Np; ++p) {
     for(d = 0; d < dim; ++d){
       posres[p*dim+d] = v[p*dim+d];
     }
  }

  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Posres,&posres);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec X,Vec Vres,void *ctx)
{
  AppCtx            *user = (AppCtx *) ctx;
  DM                 dm, plex;
  PetscDS            prob;
  PetscFE            fe;
  Mat                M_p;
  Vec                phi, locPhi, rho, f;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscReal         *coords, *rhoArr, rhoSum, rhoAvg;
  PetscInt           dim, d, cStart, cEnd, cell, cdim, rhoSize;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  PetscObjectSetName((PetscObject) X, "rhsf2 position");
  VecViewFromOptions(X, NULL, "-rhsf2_x_view");
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Vres,&vres);CHKERRQ(ierr);

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = SNESGetDM(user->snes, &plex);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(plex, &cdim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(plex, &phi);CHKERRQ(ierr);
  ierr = DMGetLocalVector(plex, &locPhi);CHKERRQ(ierr);
  /* Get charge vector */
  ierr = DMCreateMassMatrix(dm, plex, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-mp_view");
  ierr = DMGetGlobalVector(plex, &rho);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(dm, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) f, "weights vector");
  ierr = VecViewFromOptions(f, NULL, "-weights_view");
  ierr = MatMultTranspose(M_p, f, rho);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dm, "w_q", &f);CHKERRQ(ierr);
  /* Solve Poisson */
  PetscObjectSetName((PetscObject) rho, "rho");
  ierr = VecViewFromOptions(rho, NULL, "-poisson_rho_view");
  
  ierr = VecGetLocalSize(rho, &rhoSize);CHKERRQ(ierr);
  ierr = VecGetArray(rho, &rhoArr);CHKERRQ(ierr);
  
  /* subtract average for SVD */
  rhoSum = 0;
  for(int i = 0; i < rhoSize; ++i){
    rhoSum += rhoArr[i];
  }

  rhoAvg = rhoSum/rhoSize;

  for(int i = 0; i < rhoSize; ++i){
    rhoArr[i] = rhoArr[i] - rhoAvg;
  }
  
  ierr = VecRestoreArray(rho, &rhoArr);CHKERRQ(ierr);
  
  ierr = VecSet(phi, 0.0);CHKERRQ(ierr);
  ierr = SNESSolve(user->snes, rho, phi);CHKERRQ(ierr);
  ierr = VecViewFromOptions(phi, NULL, "-phi_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(plex, &rho);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(plex, phi, INSERT_VALUES, locPhi);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(plex, phi, INSERT_VALUES, locPhi);CHKERRQ(ierr);
  /* Add in electrostatic forces */
  ierr = DMSwarmSortGetAccess(dm);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscReal    v[3], J[9], invJ[9], detJ;
    PetscScalar *ph       = PETSC_NULL;
    PetscReal   *D        = PETSC_NULL;
    PetscReal   *pcoord   = PETSC_NULL;
    PetscReal   *refcoord = PETSC_NULL;
    PetscInt    *points   = PETSC_NULL, Ncp, cp;
    PetscScalar  gradPhi[3];

    // Get geometry of cell
    //   TODO Put in correct quadrature handling
    ierr = DMPlexComputeCellGeometryFEM(plex, cell, NULL, v, J, invJ, &detJ);CHKERRQ(ierr);
    // Determine particles in cell from swarm
    ierr = DMSwarmSortGetPointsPerCell(dm, cell, &Ncp, &points);CHKERRQ(ierr);
    // Get particle coordinates
    ierr = DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord);CHKERRQ(ierr);
    for (cp = 0; cp < Ncp; ++cp) {
      for (d = 0; d < cdim; ++d) {
        pcoord[cp*cdim+d] = coords[points[cp]*cdim+d];
      }
    }
    // Tabulate basis at particle coordinates
    // Need to tabulate basis for particle coordinates in the reference cell
    ierr = DMPlexCoordinatesToReference(plex, cell, Ncp, pcoord, refcoord);CHKERRQ(ierr);
    ierr = PetscFEGetTabulation(fe, Ncp, refcoord, NULL, &D, NULL);CHKERRQ(ierr);
    // Get coefficients from phi for closure of cell
    ierr = DMPlexVecGetClosure(plex, NULL, locPhi, cell, NULL, &ph);CHKERRQ(ierr);
    // Interpolate gradient
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];
      gradPhi[0] = 0.0;
      gradPhi[1] = 0.0;
      gradPhi[2] = 0.0;
      ierr = PetscFEFreeInterpolateGradient_Static(fe, D, ph, cdim, invJ, NULL, cp, gradPhi);CHKERRQ(ierr);
      // Compute particle residual
      for (d = 0; d < cdim; ++d) {
        // TODO put in electrostatic force using gradPhi[p*cdim]
        vres[p*cdim+d] = d == 0 ? gradPhi[d] : 0.;
        PetscPrintf(MPI_COMM_WORLD, "gradphi indexing: %i\n", d);
        PetscPrintf(MPI_COMM_WORLD, "gradphi[%i]: %f\n", d, gradPhi[d]);
        PetscPrintf(MPI_COMM_WORLD, "cell: %i\n", cell);

        ierr = PetscPrintf(PETSC_COMM_SELF, "vres for particle %d, dim %d: %g\n", cp, d, vres[p*dim+d]);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexVecRestoreClosure(plex, NULL, locPhi, cell, NULL, &ph);CHKERRQ(ierr);
    ierr = PetscFERestoreTabulation(fe, Ncp, pcoord, NULL, &D, NULL);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord);CHKERRQ(ierr);
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(dm);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(plex, &locPhi);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(plex, &phi);CHKERRQ(ierr);

  ierr = VecRestoreArray(Vres,&vres);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Vres, NULL, "-vel_res_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts,PetscReal t,Vec U,Vec R,void *ctx)
{
  const PetscScalar *u;
  PetscScalar       *r, rsqr; 
  PetscInt          Np, p, dim, d;
  DM                dm;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Np /= 2*dim;
  
  for( p = 0; p < Np; ++p){
    rsqr = 0;
    for(d=0; d < dim; ++d) rsqr += PetscSqr(u[p*2*dim+d]);
    for(d=0; d < dim; ++d){
        r[p*2*dim+d] = u[p*2*dim+d+2];
        r[p*2*dim+d+2] = (-1.*1000./(1.+p))*u[p*2*dim+d]/rsqr;
    }
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt          i, par;
  TSConvergedReason reason;
  const PetscScalar *endVals;
  PetscReal         vx, vy;
  PetscInt          locSize, p, d, dim, Np, steps, step, *idx1, *idx2;
  TS                ts;
  DM                dm, sw;
  AppCtx            user;
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  Vec               coorVec, kinVec, probVec, solution, position, momentum;
  const PetscScalar *coorArr, *kinArr;
  PetscReal         ftime   = 10., *probArr, *probVecArr;
  IS                is1,is2;
  PetscReal         *coor, *kin, *pos, *mom;
  PetscScalar       *weights;

  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  /* Create dm and particles */
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateFEM(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  

  ierr = SNESCreate(comm, &user.snes);CHKERRQ(ierr);
  ierr = SNESSetDM(user.snes, dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user.snes);CHKERRQ(ierr);
  
  /* Place TSSolve in a loop to handle resetting the TS at every manual call of TSStep() */
  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,100000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  for(step = 0; step < user.steps ; ++step){
  
    ierr = DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &kinVec);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(kinVec, NULL, "-ic_vec_view");
    ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
    ierr = VecGetLocalSize(kinVec, &locSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(locSize, &idx1);CHKERRQ(ierr);
    ierr = PetscMalloc1(locSize, &idx2);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*locSize, &probArr);CHKERRQ(ierr);
    Np = locSize/dim;

    ierr = VecGetArrayRead(kinVec, &kinArr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coorVec, &coorArr);CHKERRQ(ierr);
    for (p=0; p<Np; ++p){
        for(d=0; d<dim;++d){
            probArr[p*2*dim + d] = coorArr[p*dim+d];
            probArr[(p*2+1)*dim + d] = kinArr[p*dim+d];
        }
    }
    ierr = VecRestoreArrayRead(kinVec, &kinArr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coorVec, &coorArr);CHKERRQ(ierr);

    /* Allocate for IS Strides that will contain x, y and vx, vy */
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        idx1[p*dim+d] = (p*2+0)*dim + d;
        idx2[p*dim+d] = (p*2+1)*dim + d;
      }
    }

    ierr = ISCreateGeneral(comm, locSize, idx1, PETSC_OWN_POINTER, &is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, locSize, idx2, PETSC_OWN_POINTER, &is2);CHKERRQ(ierr);

    /* DM needs to be set before splits so it propogates to sub TSs */
    
    ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSBASICSYMPLECTIC);CHKERRQ(ierr);
    
    ierr = TSRHSSplitSetIS(ts,"position",is1);CHKERRQ(ierr);
    ierr = TSRHSSplitSetIS(ts,"momentum",is2);CHKERRQ(ierr);

    ierr = TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user);CHKERRQ(ierr);
    ierr = TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user);CHKERRQ(ierr);

    ierr = TSSetRHSFunction(ts,NULL,RHSFunctionParticles,&user);CHKERRQ(ierr);

    ierr = TSSetTime(ts, step*.1);CHKERRQ(ierr);
    if (step == 0){
      ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    }
    /* Compose vector from array for TS solve with all kinematic variables */
    ierr = VecCreate(comm,&probVec);CHKERRQ(ierr);
    ierr = VecSetBlockSize(probVec,1);CHKERRQ(ierr);
    ierr = VecSetSizes(probVec,PETSC_DECIDE,2*locSize);CHKERRQ(ierr);
    ierr = VecSetUp(probVec);CHKERRQ(ierr);

    ierr = VecGetArray(probVec,&probVecArr);
    for (i=0; i < 2*locSize; ++i) {

      probVecArr[i] = probArr[i];

    }
    ierr = VecRestoreArray(probVec,&probVecArr);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, probVec);CHKERRQ(ierr);
    ierr = PetscFree(probArr);CHKERRQ(ierr);
    ierr = VecViewFromOptions(kinVec, NULL, "-ic_view");
    ierr = DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &kinVec);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec);CHKERRQ(ierr);

    ierr = TSMonitor(ts, step, ts->ptime, ts->vec_sol);CHKERRQ(ierr);
    if (!ts->steprollback) {
      ierr = TSPreStep(ts);CHKERRQ(ierr);
    }
    ierr = TSStep(ts);CHKERRQ(ierr);
    if (ts->steprollback) {
         ierr = TSPostEvaluate(ts);CHKERRQ(ierr);
    }
    if (!ts->steprollback) {
      TSPostStep(ts);

      ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor);CHKERRQ(ierr);
      ierr = DMSwarmGetField(sw, "kinematics", NULL, NULL, (void **) &kin);CHKERRQ(ierr);
      ierr = TSGetSolution(ts, &solution);CHKERRQ(ierr);
      
      ierr = VecGetSubVector(solution,is1,&position);CHKERRQ(ierr);
      ierr = VecGetSubVector(solution,is2,&momentum);CHKERRQ(ierr);
      
      ierr = VecGetArray(position, &pos);CHKERRQ(ierr);
      ierr = VecGetArray(momentum, &mom);CHKERRQ(ierr);

      for(par = 0; par < Np; ++par){
        for(d=0; d<dim; ++d){
          if( pos[par*dim+d] < 0.){
            coor[par*dim+d] = pos[par*dim+d] + 2.*PETSC_PI;
          }
          else if(pos[par*dim+d] > 2.*PETSC_PI){
            coor[par*dim+d] = pos[par*dim+d] - 2.*PETSC_PI;
          }
          else{
            coor[par*dim+d] = pos[par*dim+d];
          }
          
          kin[par*dim+d] = mom[par*dim+d];
        }
      }
      
      ierr = VecRestoreArray(position, &pos);CHKERRQ(ierr);
      ierr = VecRestoreArray(momentum, &mom);CHKERRQ(ierr);
      
      ierr = VecRestoreSubVector(solution,is1,&position);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(solution,is2,&momentum);CHKERRQ(ierr);
      
      ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sw, "kinematics", NULL, NULL, (void **) &kin);CHKERRQ(ierr);

    }
    ierr = DMSwarmMigrate(sw, PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSReset(ts);CHKERRQ(ierr);
    ierr = PetscFree(idx1);CHKERRQ(ierr);
    ierr = PetscFree(idx2);CHKERRQ(ierr);
    ierr = PetscFree(probArr);CHKERRQ(ierr);
  }
  ierr = TSGetConvergedReason(ts, &reason);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = SNESDestroy(&user.snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* move back to space degree 2 */
/*TEST

   build:
     requires: triangle !single !complex
   test:
     suffix: bsi1
     args: -dim 2 -faces 4 -simplex 0 -particlesPerCell 6 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 2 -ts_basicsymplectic_type 1 -snes_monitor -pc_type svd
   test:
     suffix: bsi2
     args: -dim 2 -faces 4 -simplex 0 -particlesPerCell 6 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 2 -ts_basicsymplectic_type 2 -snes_monitor -pc_type svd
   test:
     suffix: bsi1q2
     args: -dim 2 -faces 32 -simplex 0 -particlesPerCell 6 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order 2 -ts_basicsymplectic_type 1 -snes_monitor -pc_type svd
   test:
     suffix: bsi2q2
     args: -dim 2 -faces 4 -simplex 0 -particlesPerCell 6 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order 2 -ts_basicsymplectic_type 2 -snes_monitor -pc_type svd

TEST*/
