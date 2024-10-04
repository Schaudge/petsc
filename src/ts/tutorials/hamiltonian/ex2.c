static char help[] = "Landau Damping/Two Stream instability test using Vlasov-Poisson equations\n";

/*
  To run the code with particles sinusoidally perturbed in x space use the test "pp_poisson_bsi_1d_4" or "pp_poisson_bsi_2d_4"
  According to Lukas, good damping results come at ~16k particles per cell

  To visualize the efield use

    -monitor_efield

  To visualize the swarm distribution use

    -ts_monitor_hg_swarm

  To visualize the particles, we can use

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500

*/
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petsc/private/dmpleximpl.h>  /* For norm and dot */
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include "petscdm.h"
#include "petscdmlabel.h"

PETSC_EXTERN PetscErrorCode stream(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode line(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

const char *EMTypes[] = {"primal", "mixed", "coulomb", "none", "EMType", "EM_", NULL};
typedef enum {
  EM_PRIMAL,
  EM_MIXED,
  EM_COULOMB,
  EM_NONE
} EMType;

typedef enum {
  V0,
  X0,
  T0,
  M0,
  Q0,
  PHI0,
  POISSON,
  VLASOV,
  SIGMA,
  M_N0,
  NUM_CONSTANTS
} ConstantType;
typedef struct {
  PetscScalar v0; /* Velocity scale, often the thermal velocity */
  PetscScalar t0; /* Time scale */
  PetscScalar x0; /* Space scale */
  PetscScalar m0; /* Mass scale */
  PetscScalar q0; /* Charge scale */
  PetscScalar kb;
  PetscScalar epsi0;
  PetscScalar phi0;          /* Potential scale */
  PetscScalar poissonNumber; /* Non-Dimensional Poisson Number */
  PetscScalar vlasovNumber;  /* Non-Dimensional Vlasov Number */
  PetscReal   sigma;         /* Nondimensional charge per length in x */
} Parameter;

typedef struct {
  PetscBag    bag;            /* Problem parameters */
  PetscBool   error;          /* Flag for printing the error */
  PetscBool   efield_monitor; /* Flag to show electric field monitor */
  PetscBool   initial_monitor;
  PetscBool   fake_1D;           /* Run simulation in 2D but zeroing second dimension */
  PetscBool   perturbed_weights; /* Uniformly sample x,v space with gaussian weights */
  PetscBool   poisson_monitor;
  PetscInt    ostep;                  /* print the energy at each ostep time steps */
  PetscReal   timeScale;              /* Nondimensionalizing time scale */
  PetscReal   charges[2];             /* The charges of each species */
  PetscReal   masses[2];              /* The masses of each species */
  PetscReal   thermal_energy[2];      /* Thermal Energy (used to get other constants)*/
  PetscReal   cosine_coefficients[2]; /*(alpha, k)*/
  PetscReal   totalWeight;
  PetscReal   stepSize;
  PetscInt    steps;
  PetscReal   initVel;
  EMType      em; /* Type of electrostatic model */
  SNES        snes;
  PetscDraw   drawef;
  PetscDrawLG drawlg_ef;
  PetscDraw   drawic_x;
  PetscDraw   drawic_v;
  PetscDraw   drawic_w;
  PetscDrawHG drawhgic_x;
  PetscDrawHG drawhgic_v;
  PetscDrawHG drawhgic_w;
  PetscDraw   EDraw;
  PetscDraw   RhoDraw;
  PetscDraw   PotDraw;
  PetscDrawSP EDrawSP;
  PetscDrawSP RhoDrawSP;
  PetscDrawSP PotDrawSP;
  PetscBool   monitor_positions; /* Flag to show particle positins at each time step */
  PetscDraw   positionDraw;
  PetscDrawSP positionDrawSP;
  DM          swarm;
  PetscRandom random;
  PetscBool   twostream;
  PetscBool   checkweights;
  PetscInt    checkVRes; /* Flag to check/output velocity residuals for nightly tests */
  PetscReal   m_n0;      // for entropy: m0 * mass / global_weight
  PetscInt    resample_period;
  Mat         M_p0;      // Particles to resample with
  DM          vdm;
  PetscReal  *part_crd_x;
  PetscReal  *velocity;
  PetscBool   uniform_velocity; // not AMR in v space
  PetscReal   v_amr_grading_factor;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                      = 2;
  PetscInt maxSpecies             = 2;
  options->error                  = PETSC_FALSE;
  options->efield_monitor         = PETSC_FALSE;
  options->initial_monitor        = PETSC_FALSE;
  options->fake_1D                = PETSC_FALSE;
  options->perturbed_weights      = PETSC_FALSE;
  options->poisson_monitor        = PETSC_FALSE;
  options->ostep                  = 100;
  options->timeScale              = 2.0e-14;
  options->charges[0]             = -1.0;
  options->charges[1]             = 1.0;
  options->masses[0]              = 1.0;
  options->masses[1]              = 1000.0;
  options->thermal_energy[0]      = 1.0;
  options->thermal_energy[1]      = 1.0;
  options->cosine_coefficients[0] = 0.01;
  options->cosine_coefficients[1] = 0.5;
  options->initVel                = 1;
  options->totalWeight            = 1.0;
  options->drawef                 = NULL;
  options->drawlg_ef              = NULL;
  options->drawic_x               = NULL;
  options->drawic_v               = NULL;
  options->drawic_w               = NULL;
  options->drawhgic_x             = NULL;
  options->drawhgic_v             = NULL;
  options->drawhgic_w             = NULL;
  options->EDraw                  = NULL;
  options->RhoDraw                = NULL;
  options->PotDraw                = NULL;
  options->EDrawSP                = NULL;
  options->RhoDrawSP              = NULL;
  options->PotDrawSP              = NULL;
  options->em                     = EM_COULOMB;
  options->monitor_positions      = PETSC_FALSE;
  options->positionDraw           = NULL;
  options->positionDrawSP         = NULL;
  options->twostream              = PETSC_FALSE;
  options->checkweights           = PETSC_FALSE;
  options->checkVRes              = 0;
  options->m_n0                   = 0;
  options->resample_period        = 0;
  options->M_p0                   = NULL;
  options->vdm = NULL;
  options->part_crd_x = NULL;
  options->velocity = NULL;
  options->uniform_velocity      = PETSC_TRUE;
  options->v_amr_grading_factor      = 0.05;

  PetscOptionsBegin(comm, "", "Landau Damping and Two Stream options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex2.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsBool("-monitor_efield", "Flag to show efield plot", "ex2.c", options->efield_monitor, &options->efield_monitor, NULL));
  PetscCall(PetscOptionsBool("-monitor_ics", "Flag to show initial condition histograms", "ex2.c", options->initial_monitor, &options->initial_monitor, NULL));
  PetscCall(PetscOptionsBool("-monitor_positions", "The flag to show particle positions", "ex2.c", options->monitor_positions, &options->monitor_positions, NULL));
  PetscCall(PetscOptionsBool("-monitor_poisson", "The flag to show charges, Efield and potential solve", "ex2.c", options->poisson_monitor, &options->poisson_monitor, NULL));
  PetscCall(PetscOptionsBool("-fake_1D", "Flag to run a 1D simulation (but really in 2D)", "ex2.c", options->fake_1D, &options->fake_1D, NULL));
  PetscCall(PetscOptionsBool("-twostream", "Run two stream instability", "ex2.c", options->twostream, &options->twostream, NULL));
  PetscCall(PetscOptionsBool("-perturbed_weights", "Flag to run uniform sampling with perturbed weights", "ex2.c", options->perturbed_weights, &options->perturbed_weights, NULL));
  PetscCall(PetscOptionsBool("-check_weights", "Ensure all particle weights are positive", "ex2.c", options->checkweights, &options->checkweights, NULL));
  PetscCall(PetscOptionsInt("-check_vel_res", "Check particle velocity residuals for nightly tests", "ex2.c", options->checkVRes, &options->checkVRes, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex2.c", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", "ex2.c", options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", "ex2.c", options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", "ex2.c", options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", "ex2.c", options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", "ex2.c", options->charges, &maxSpecies, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", "ex2.c", EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
  PetscCall(PetscOptionsInt("-resample_period", "Number of time steps between resampling", "ex2.c", options->resample_period, &options->resample_period, NULL));
  PetscCall(PetscOptionsBool("-uniform_velocity", "Uniform velocity grid, otherwise a graded mesh", "ex2.c", options->uniform_velocity, &options->uniform_velocity, NULL));
  PetscCall(PetscOptionsReal("-v_amr_grading_factor", "Mysterious parameter for velocity space grading", "ex2.c", options->v_amr_grading_factor, &options->v_amr_grading_factor, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, DM sw, AppCtx *user)
{
  PetscFunctionBeginUser;
  if (user->efield_monitor) {
    /* PetscDrawAxis axis_ef; */
    /* PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "monitor_efield", 0, 300, 400, 300, &user->drawef)); */
    /* PetscCall(PetscDrawSetSave(user->drawef, "ex9_Efield.png")); */
    /* PetscCall(PetscDrawSetFromOptions(user->drawef)); */
    /* PetscCall(PetscDrawLGCreate(user->drawef, 1, &user->drawlg_ef)); */
    /* PetscCall(PetscDrawLGGetAxis(user->drawlg_ef, &axis_ef)); */
    /* PetscCall(PetscDrawAxisSetLabels(axis_ef, "Electron Electric Field", "time", "E_max")); */
    /* PetscCall(PetscDrawLGSetLimits(user->drawlg_ef, 0., user->steps * user->stepSize, -10., 0.)); */
    /* PetscCall(PetscDrawAxisSetLimits(axis_ef, 0., user->steps * user->stepSize, -10., 0.)); */
  }
  if (user->initial_monitor) {
    PetscDrawAxis axis1, axis2, axis3;
    PetscReal     dmboxlower[2], dmboxupper[2];
    PetscInt      dim, cStart, cEnd;
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMGetBoundingBox(dm, dmboxlower, dmboxupper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "monitor_initial_conditions_x", 0, 300, 400, 300, &user->drawic_x));
    PetscCall(PetscDrawSetSave(user->drawic_x, "ex9_ic_x.png"));
    PetscCall(PetscDrawSetFromOptions(user->drawic_x));
    PetscCall(PetscDrawHGCreate(user->drawic_x, (int)dim, &user->drawhgic_x));
    PetscCall(PetscDrawHGGetAxis(user->drawhgic_x, &axis1));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgic_x, (int)(cEnd - cStart)));
    PetscCall(PetscDrawAxisSetLabels(axis1, "Initial X Distribution", "X", "counts"));
    PetscCall(PetscDrawAxisSetLimits(axis1, dmboxlower[0], dmboxupper[0], 0, 1500));

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "monitor_initial_conditions_v", 400, 300, 400, 300, &user->drawic_v));
    PetscCall(PetscDrawSetSave(user->drawic_v, "ex9_ic_v.png"));
    PetscCall(PetscDrawSetFromOptions(user->drawic_v));
    PetscCall(PetscDrawHGCreate(user->drawic_v, (int)dim, &user->drawhgic_v));
    PetscCall(PetscDrawHGGetAxis(user->drawhgic_v, &axis2));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgic_v, 1000));
    PetscCall(PetscDrawAxisSetLabels(axis2, "Initial V_x Distribution", "V", "counts"));
    PetscCall(PetscDrawAxisSetLimits(axis2, -1, 1, 0, 1500));

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "monitor_initial_conditions_w", 800, 300, 400, 300, &user->drawic_w));
    PetscCall(PetscDrawSetSave(user->drawic_w, "ex9_ic_w.png"));
    PetscCall(PetscDrawSetFromOptions(user->drawic_w));
    PetscCall(PetscDrawHGCreate(user->drawic_w, (int)dim, &user->drawhgic_w));
    PetscCall(PetscDrawHGGetAxis(user->drawhgic_w, &axis3));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgic_w, 10));
    PetscCall(PetscDrawAxisSetLabels(axis3, "Initial W Distribution", "weight", "counts"));
    PetscCall(PetscDrawAxisSetLimits(axis3, 0, 0.01, 0, 5000));
  }
  if (user->monitor_positions) {
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "position_monitor_species1", 0, 0, 400, 300, &user->positionDraw));
    PetscCall(PetscDrawSetFromOptions(user->positionDraw));
    PetscCall(PetscDrawSPCreate(user->positionDraw, 10, &user->positionDrawSP));
    PetscCall(PetscDrawSPSetDimension(user->positionDrawSP, 1));
    PetscCall(PetscDrawSPGetAxis(user->positionDrawSP, &axis));
    PetscCall(PetscDrawSPReset(user->positionDrawSP));
    PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "v"));
    PetscCall(PetscDrawSetSave(user->positionDraw, "ex9_pos.png"));
  }
  if (user->poisson_monitor) {
    PetscDrawAxis axis_E, axis_Rho, axis_Pot;

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "Efield_monitor", 0, 0, 400, 300, &user->EDraw));
    PetscCall(PetscDrawSetFromOptions(user->EDraw));
    PetscCall(PetscDrawSPCreate(user->EDraw, 10, &user->EDrawSP));
    PetscCall(PetscDrawSPSetDimension(user->EDrawSP, 1));
    PetscCall(PetscDrawSPGetAxis(user->EDrawSP, &axis_E));
    PetscCall(PetscDrawSPReset(user->EDrawSP));
    PetscCall(PetscDrawAxisSetLabels(axis_E, "Particles", "x", "E"));
    PetscCall(PetscDrawSetSave(user->EDraw, "ex9_E_spatial.png"));

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "rho_monitor", 0, 0, 400, 300, &user->RhoDraw));
    PetscCall(PetscDrawSetFromOptions(user->RhoDraw));
    PetscCall(PetscDrawSPCreate(user->RhoDraw, 10, &user->RhoDrawSP));
    PetscCall(PetscDrawSPSetDimension(user->RhoDrawSP, 1));
    PetscCall(PetscDrawSPGetAxis(user->RhoDrawSP, &axis_Rho));
    PetscCall(PetscDrawSPReset(user->RhoDrawSP));
    PetscCall(PetscDrawAxisSetLabels(axis_Rho, "Particles", "x", "rho"));
    PetscCall(PetscDrawSetSave(user->RhoDraw, "ex9_rho_spatial.png"));

    PetscCall(PetscDrawCreate(PETSC_COMM_WORLD, NULL, "potential_monitor", 0, 0, 400, 300, &user->PotDraw));
    PetscCall(PetscDrawSetFromOptions(user->PotDraw));
    PetscCall(PetscDrawSPCreate(user->PotDraw, 10, &user->PotDrawSP));
    PetscCall(PetscDrawSPSetDimension(user->PotDrawSP, 1));
    PetscCall(PetscDrawSPGetAxis(user->PotDrawSP, &axis_Pot));
    PetscCall(PetscDrawSPReset(user->PotDrawSP));
    PetscCall(PetscDrawAxisSetLabels(axis_Pot, "Particles", "x", "potential"));
    PetscCall(PetscDrawSetSave(user->PotDraw, "ex9_phi_spatial.png"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(PetscDrawLGDestroy(&user->drawlg_ef));
  PetscCall(PetscDrawDestroy(&user->drawef));
  PetscCall(PetscDrawHGDestroy(&user->drawhgic_x));
  PetscCall(PetscDrawDestroy(&user->drawic_x));
  PetscCall(PetscDrawHGDestroy(&user->drawhgic_v));
  PetscCall(PetscDrawDestroy(&user->drawic_v));
  PetscCall(PetscDrawHGDestroy(&user->drawhgic_w));
  PetscCall(PetscDrawDestroy(&user->drawic_w));
  PetscCall(PetscDrawSPDestroy(&user->positionDrawSP));
  PetscCall(PetscDrawDestroy(&user->positionDraw));

  PetscCall(PetscDrawSPDestroy(&user->EDrawSP));
  PetscCall(PetscDrawDestroy(&user->EDraw));
  PetscCall(PetscDrawSPDestroy(&user->RhoDrawSP));
  PetscCall(PetscDrawDestroy(&user->RhoDraw));
  PetscCall(PetscDrawSPDestroy(&user->PotDrawSP));
  PetscCall(PetscDrawDestroy(&user->PotDraw));

  if (user->part_crd_x) {
    PetscCall(PetscFree2(user->part_crd_x, user->velocity));
    PetscCall(MatDestroy(&user->M_p0));
  }
  if (user->vdm) PetscCall(DMDestroy(&user->vdm));

  PetscCall(PetscBagDestroy(&user->bag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckNonNegativeWeights(DM sw, AppCtx *user)
{
  const PetscScalar *w;
  PetscInt           Np;

  PetscFunctionBeginUser;
  if (!user->checkweights) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  for (PetscInt p = 0; p < Np; ++p) PetscCheck(w[p] >= 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " has negative weight %g", p, w[p]);
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computeParticleMoments(DM sw, PetscReal moments[5], AppCtx *user)
{
  DM                 dm;
  const PetscReal   *coords;
  const PetscScalar *w;
  PetscReal          mom[] = {0.0, 0.0, 0.0, 0.0, 0.0};
  PetscInt           cell, cStart, cEnd, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pidx;
    PetscInt  Np, p, d;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, cell, &Np, &pidx));
    for (p = 0; p < Np; ++p) {
      const PetscInt   idx = pidx[p];
      const PetscReal *c   = &coords[idx * dim];
      mom[0] += PetscRealPart(w[idx]);
      mom[1] += PetscRealPart(w[idx]) * c[0];
      for (d = 0; d < dim; ++d) mom[2] += PetscRealPart(w[idx]) * c[d] * c[d];
      if (w[idx] * user->m_n0 > PETSC_REAL_MIN) {
        PetscReal ww = w[idx] * user->m_n0;
        mom[3] += ww * PetscLogReal(ww);
      } else if (w[idx] * user->m_n0 < 0) {
        PetscReal ww = -w[idx] * user->m_n0;
        mom[4] += ww * PetscLogReal(ww);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "warning, negative weight %" PetscInt_FMT " %e\n", idx, w[idx]));
      }
    }
    PetscCall(PetscFree(pidx));
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCallMPI(MPIU_Allreduce(mom, moments, 5, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)sw)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

static void f0_x(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = x[0] * u[0];
}

static void f0_r2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) f0[0] += PetscSqr(x[d]) * u[0];
}

static PetscReal s_neg_w;

static void f0_f_logf(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal m_n0 = PetscRealPart(constants[M_N0]), ww = PetscRealPart(u[0]) * m_n0;

  if (ww > 0) f0[0] = ww * PetscLogReal(ww);
  else s_neg_w += ww;
}

static PetscErrorCode computeFEMMoments(DM dm, Vec u, PetscReal moments[5], AppCtx *user)
{
  PetscDS     prob;
  PetscScalar mom;
  PetscInt    field = 0;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSSetObjective(prob, field, &f0_1));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[0] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(prob, field, &f0_x));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[1] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(prob, field, &f0_r2));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[2] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(prob, field, &f0_f_logf));
  s_neg_w = 0;
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[4] = s_neg_w; // not a real integral, just != 0
  moments[3] = PetscRealPart(mom);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorEField(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx    *user = (AppCtx *)ctx;
  DM         dm, sw;
  PetscReal *E;
  PetscReal  Enorm = 0., lgEnorm, lgEmax, sum = 0., Emax = 0., temp = 0., *weight, chargesum = 0.;
  PetscReal *x, *v;
  PetscInt  *species, dim, p, d, Np, cStart, cEnd;
  PetscReal  pmoments[5]; /* \int f, \int x f, \int r^2 f, \int - f ln (f) */
  PetscReal  fmoments[5]; /* \int \hat f, \int x \hat f, \int r^2 \hat f , \int - \hatf ln (\hat f)*/
  Vec        rho;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  else if (step == 0)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time            Sum E           |E|              log(|E|)       E_max            log(E_max)      sum(q)   part: moment-0         moment-1       moment-2        entropy            FEM: moment-0        "
                                            "moment-1        moment-2        entropy.    negative w: particle ent.    FEM points\n"));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  for (p = 0; p < Np; ++p) {
    for (d = 0; d < 1; ++d) {
      temp = PetscAbsReal(E[p * dim + d]);
      if (temp > Emax) Emax = temp;
    }
    Enorm += PetscSqrtReal(E[p * dim] * E[p * dim]);
    sum += E[p * dim];
    chargesum += user->charges[0] * weight[p];
  }
  lgEnorm = Enorm != 0 ? PetscLog10Real(Enorm) : -16.;
  lgEmax  = Emax != 0 ? PetscLog10Real(Emax) : -16.;

  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));

  Parameter *param;
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "charges", &rho));
  if (user->em == EM_PRIMAL) {
    PetscCall(computeParticleMoments(sw, pmoments, user));
    PetscCall(computeFEMMoments(dm, rho, fmoments, user));
  } else if (user->em == EM_MIXED) {
    DM       potential_dm;
    IS       potential_IS;
    PetscInt fields = 1;
    PetscCall(DMCreateSubDM(dm, 1, &fields, &potential_IS, &potential_dm));

    PetscCall(computeParticleMoments(sw, pmoments, user));
    PetscCall(computeFEMMoments(potential_dm, rho, fmoments, user));
    PetscCall(DMDestroy(&potential_dm));
    PetscCall(ISDestroy(&potential_IS));
  }
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "charges", &rho));

  pmoments[3] = -pmoments[3];
  fmoments[3] = -fmoments[3];
  PetscCall(DMSwarmGetSize(sw, &Np));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "E: %f\t%+e\t%e\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%20.14e\t%f\t%f\t%f\t%20.14e\t%e\t%f. %" PetscInt_FMT " particles\n", (double)t, (double)sum, (double)Enorm, (double)lgEnorm, (double)Emax, (double)lgEmax, (double)chargesum, (double)pmoments[0], (double)pmoments[1], (double)pmoments[2], (double)pmoments[3], (double)fmoments[0], (double)fmoments[1], (double)fmoments[2], (double)fmoments[3], (double)pmoments[4], (double)fmoments[4], Np));

  if (user->fake_1D && user->ostep && step % user->ostep == 0) { // print time series of 1 + 1V
    PetscInt np;
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmGetLocalSize(sw, &np));
    for (int p = 0; p < np; ++p) {
      for (int d = 0; d < dim; ++d) {
        if (d > 0) {
          x[p * dim + d] = v[p * dim]; // put V[0] into x[1] for viz
        }
      }
    }
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
    char line[128];
    if (user->fake_1D && !user->uniform_velocity) PetscCall(PetscSNPrintf(line, 128, "e_phase_amr_%04d.xmf", (int)step));
    else PetscCall(PetscSNPrintf(line, 128, "e_phase_%04d.xmf", (int)step));
    PetscCall(DMSwarmViewXDMF(sw, line));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
    for (int p = 0; p < np; ++p) {
      for (d = 1; d < dim; ++d) x[p * dim + d] = 0;
    }
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  }
  /* PetscCall(PetscDrawLGAddPoint(user->drawlg_ef, &t, &lgEmax)); */
  /* PetscCall(PetscDrawLGDraw(user->drawlg_ef)); */
  /* PetscCall(PetscDrawSave(user->drawef)); */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorInitialConditions(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 dm, sw;
  const PetscScalar *u;
  PetscReal         *weight, *pos, *vel;
  PetscInt           dim, p, Np, cStart, cEnd;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS); /* -1 indicates interpolated solution */
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  if (step == 0) {
    PetscCall(PetscDrawHGReset(user->drawhgic_x));
    PetscCall(PetscDrawHGGetDraw(user->drawhgic_x, &user->drawic_x));
    PetscCall(PetscDrawClear(user->drawic_x));
    PetscCall(PetscDrawFlush(user->drawic_x));

    PetscCall(PetscDrawHGReset(user->drawhgic_v));
    PetscCall(PetscDrawHGGetDraw(user->drawhgic_v, &user->drawic_v));
    PetscCall(PetscDrawClear(user->drawic_v));
    PetscCall(PetscDrawFlush(user->drawic_v));

    PetscCall(PetscDrawHGReset(user->drawhgic_w));
    PetscCall(PetscDrawHGGetDraw(user->drawhgic_w, &user->drawic_w));
    PetscCall(PetscDrawClear(user->drawic_w));
    PetscCall(PetscDrawFlush(user->drawic_w));

    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));

    PetscCall(VecGetLocalSize(U, &Np));
    Np /= dim * 2;
    for (p = 0; p < Np; ++p) {
      PetscCall(PetscDrawHGAddValue(user->drawhgic_x, pos[p * dim]));
      PetscCall(PetscDrawHGAddValue(user->drawhgic_v, vel[p * dim]));
      PetscCall(PetscDrawHGAddValue(user->drawhgic_w, weight[p]));
    }

    PetscCall(VecRestoreArrayRead(U, &u));
    PetscCall(PetscDrawHGDraw(user->drawhgic_x));
    PetscCall(PetscDrawHGSave(user->drawhgic_x));

    PetscCall(PetscDrawHGDraw(user->drawhgic_v));
    PetscCall(PetscDrawHGSave(user->drawhgic_v));

    PetscCall(PetscDrawHGDraw(user->drawhgic_w));
    PetscCall(PetscDrawHGSave(user->drawhgic_w));

    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPositions_2D(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx         *user = (AppCtx *)ctx;
  DM              dm, sw;
  PetscScalar    *x, *v, *weight;
  PetscReal       lower[3], upper[3], speed;
  const PetscInt *s;
  PetscInt        dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  if (step > 0 && user->ostep && step % user->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetBoundingBox(dm, lower, upper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&s));
    PetscCall(DMSwarmSortGetAccess(sw));
    PetscCall(PetscDrawSPReset(user->positionDrawSP));
    PetscCall(PetscDrawSPSetLimits(user->positionDrawSP, lower[0], upper[0], lower[1], upper[1]));
    PetscCall(PetscDrawSPSetLimits(user->positionDrawSP, lower[0], upper[0], -12, 12));
    for (c = 0; c < cEnd - cStart; ++c) {
      PetscInt *pidx, Npc, q;
      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
      for (q = 0; q < Npc; ++q) {
        const PetscInt p = pidx[q];
        if (s[p] == 0) {
          speed = PetscSqrtReal(PetscSqr(v[p * dim]) + PetscSqr(v[p * dim + 1]));
          if (dim == 1 || user->fake_1D) {
            PetscCall(PetscDrawSPAddPointColorized(user->positionDrawSP, &x[p * dim], &v[p * dim], &speed));
          } else {
            PetscCall(PetscDrawSPAddPointColorized(user->positionDrawSP, &x[p * dim], &x[p * dim + 1], &speed));
          }
        } else if (s[p] == 1) {
          PetscCall(PetscDrawSPAddPoint(user->positionDrawSP, &x[p * dim], &v[p * dim]));
        }
      }
      PetscCall(PetscFree(pidx));
    }
    PetscCall(PetscDrawSPDraw(user->positionDrawSP, PETSC_TRUE));
    PetscCall(PetscDrawSave(user->positionDraw));
    PetscCall(DMSwarmSortRestoreAccess(sw));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&s));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPoisson(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx      *user = (AppCtx *)ctx;
  DM           dm, sw;
  PetscScalar *x, *E, *weight, *pot, *charges;
  PetscReal    lower[3], upper[3], xval;
  PetscInt     dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  if (step > 0 && user->ostep && step % user->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetBoundingBox(dm, lower, upper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    PetscCall(PetscDrawSPReset(user->RhoDrawSP));
    PetscCall(PetscDrawSPReset(user->EDrawSP));
    PetscCall(PetscDrawSPReset(user->PotDrawSP));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
    PetscCall(DMSwarmGetField(sw, "potential", NULL, NULL, (void **)&pot));
    PetscCall(DMSwarmGetField(sw, "charges", NULL, NULL, (void **)&charges));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));

    PetscCall(DMSwarmSortGetAccess(sw));
    for (c = 0; c < cEnd - cStart; ++c) {
      PetscReal Esum = 0.0;
      PetscInt *pidx, Npc, q;
      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
      for (q = 0; q < Npc; ++q) {
        const PetscInt p = pidx[q];
        Esum += E[p * dim];
      }
      xval = (c + 0.5) * ((upper[0] - lower[0]) / (cEnd - cStart));
      PetscCall(PetscDrawSPAddPoint(user->EDrawSP, &xval, &Esum));
      PetscCall(PetscFree(pidx));
    }
    for (c = 0; c < (cEnd - cStart); ++c) {
      xval = (c + 0.5) * ((upper[0] - lower[0]) / (cEnd - cStart));
      PetscCall(PetscDrawSPAddPoint(user->RhoDrawSP, &xval, &charges[c]));
      PetscCall(PetscDrawSPAddPoint(user->PotDrawSP, &xval, &pot[c]));
    }
    PetscCall(PetscDrawSPDraw(user->RhoDrawSP, PETSC_TRUE));
    PetscCall(PetscDrawSave(user->RhoDraw));
    PetscCall(PetscDrawSPDraw(user->EDrawSP, PETSC_TRUE));
    PetscCall(PetscDrawSave(user->EDraw));
    PetscCall(PetscDrawSPDraw(user->PotDrawSP, PETSC_TRUE));
    PetscCall(PetscDrawSave(user->PotDraw));
    PetscCall(DMSwarmSortRestoreAccess(sw));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "potential", NULL, NULL, (void **)&pot));
    PetscCall(DMSwarmRestoreField(sw, "charges", NULL, NULL, (void **)&charges));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, (void **)&p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Vlasov-Poisson Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "v0", "Velocity scale, m/s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->t0, 1.0, "t0", "Time scale, s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->x0, 1.0, "x0", "Space scale, m"));
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "phi0", "Potential scale, kg*m^2/A*s^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->q0, 1.0, "q0", "Charge Scale, A*s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->m0, 1.0, "m0", "Mass Scale, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->epsi0, 1.0, "epsi0", "Permittivity of Free Space, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->kb, 1.0, "kb", "Boltzmann Constant, m^2 kg/s^2 K^1"));

  PetscCall(PetscBagRegisterScalar(bag, &p->sigma, 1.0, "sigma", "Charge per unit area, C/m^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->poissonNumber, 1.0, "poissonNumber", "Non-Dimensional Poisson Number"));
  PetscCall(PetscBagRegisterScalar(bag, &p->vlasovNumber, 1.0, "vlasovNumber", "Non-Dimensional Vlasov Number"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsCreateViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ion_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -constants[SIGMA];
}

static void laplacian_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void laplacian_g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return PETSC_SUCCESS;
}

/*
   /  I   -grad\ / q \ = /0\
   \-div    0  / \phi/   \f/
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[d] += u[uOff[0] + d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] = u[uOff[1]];
}

static void f0_phi_backgroundCharge(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += constants[SIGMA];
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

/* Boundary residual. Dirichlet boundary for u means u_bdy=p*n */
static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; ++d) g0[d * dim + d] = 1.0;
}

static void g2_qphi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] = 1.0;
}

static void g1_phiq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE   fephi, feq;
  PetscDS   ds;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (user->em == EM_MIXED) {
    DMLabel        label;
    const PetscInt id = 1;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field_", PETSC_DETERMINE, &feq));
    PetscCall(PetscObjectSetName((PetscObject)feq, "field"));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "potential_", PETSC_DETERMINE, &fephi));
    PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
    PetscCall(PetscFECopyQuadrature(feq, fephi));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feq));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fephi));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fephi));
    PetscCall(PetscFEDestroy(&feq));

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMGetDS(dm, &ds));

    PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
    PetscCall(PetscDSSetResidual(ds, 1, f0_phi_backgroundCharge, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
    PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));

  } else if (user->em == EM_PRIMAL) {
    MatNullSpace nullsp;
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &fephi));
    PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fephi));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetResidual(ds, 0, ion_f0, laplacian_f1));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian_g3));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullsp));
    PetscCall(PetscObjectCompose((PetscObject)fephi, "nullspace", (PetscObject)nullsp));
    PetscCall(MatNullSpaceDestroy(&nullsp));
    PetscCall(PetscFEDestroy(&fephi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreatePoisson(DM dm, AppCtx *user)
{
  SNES         snes;
  Mat          J;
  MatNullSpace nullSpace;

  PetscFunctionBeginUser;
  PetscCall(CreateFEM(dm, user));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "em_"));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(J, nullSpace));
  PetscCall(MatNullSpaceDestroy(&nullSpace));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));
  user->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPDFPertubedConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1 + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  p[1] = (1 + 0.01 * PetscCosReal(0.5 * x[1])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}
PetscErrorCode PetscPDFPertubedConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.0;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine2D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.;
  const PetscReal k     = scale ? scale[0] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * (x[0] + x[1])));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine1D_TwoStream(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.0;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1. + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

static PetscErrorCode InitializeParticles_PerturbedWeights(DM sw, AppCtx *user)
{
  DM           dm;
  PetscScalar *weight;
  PetscReal   *x, *v, vmin[3], vmax[3], gmin[3], gmax[3], xi0[3];
  PetscInt     Ns, swdim, *cellid, *species, Np, cStart, cEnd, Npc;
  PetscInt     Np_global, p, q, s, c, d, cv, vStart, vEnd;
  PetscBool    flg;
  PetscMPIInt  size, rank;
  Parameter   *param;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sw), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sw), &rank));
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscOptionsEnd();

  PetscCall(DMGetDimension(sw, &swdim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  //PetscCheck((N/cdim)%size == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "[%d]Number of cells %" PetscInt_FMT " mode np %d != 0", rank, N/cdim, size);
  if (0 && !user->vdm) { // create Landau
  } else if (!user->vdm) {
    PetscCall(DMCreate(PETSC_COMM_SELF, &user->vdm));
    PetscCall(DMSetType(user->vdm, DMPLEX));
    PetscCall(DMPlexSetOptionsPrefix(user->vdm, "v"));
    PetscCall(DMSetFromOptions(user->vdm));
    PetscCall(PetscObjectSetName((PetscObject)user->vdm, "Velocity space"));
    PetscCall(DMViewFromOptions(user->vdm, NULL, "-dm_view"));
  }
  PetscCall(DMPlexGetHeightStratum(user->vdm, 0, &vStart, &vEnd));
  PetscCall(DMGetBoundingBox(user->vdm, vmin, vmax)); // works with all Landau (eg, file)

  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  Np = (cEnd - cStart) * (vEnd - vStart);
  PetscCallMPI(MPIU_Allreduce(&Np, &Np_global, 1, MPIU_INT, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Global Np = %" PetscInt_FMT "\n", Np_global));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  Npc = vEnd - vStart;
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = 0, p = 0; c < cEnd - cStart; ++c) {
    for (s = 0; s < Ns; ++s) {
      for (q = 0; q < Npc; ++q, ++p) cellid[p] = c;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));

  PetscCall(DMSwarmSortGetAccess(sw));
  // simple velocity space adaptivity for an approximate of Maxwellian
  PetscReal max_v = 0;
  if (user->fake_1D && !user->uniform_velocity) {
    if (0) { // Landau, version of setting v coordinates
    } else {
      Vec          coordinates;
      PetscScalar *coords;
      PetscInt     vdim, N, bs, i;
      // create 1D AMR particles (not Landau)
      PetscCall(DMGetCoordinateDim(user->vdm, &vdim)); // dim == 1
      PetscCall(DMGetCoordinates(user->vdm, &coordinates));
      PetscCall(VecGetLocalSize(coordinates, &N));
      PetscCall(VecGetBlockSize(coordinates, &bs));
      PetscCall(VecGetArray(coordinates, &coords));
      for (i = 0; i < N; i += vdim) {
        PetscScalar v = coords[i];
        if (PetscAbsReal(v) > vmax[0]/2) coords[i] *= 1 + PetscSqr(PetscAbsReal(v) - vmax[0]/2) * user->v_amr_grading_factor;
        if (coords[i] > max_v) max_v = coords[i];
      }
      // scale down & set
      PetscReal fact = vmax[0] / max_v;
      for (i = 0; i < N; i += vdim) coords[i] *= fact;
      PetscCall(VecRestoreArray(coordinates, &coords));
      PetscCall(DMSetCoordinates(user->vdm, coordinates));
    }
    PetscCall(DMGetCoordinatesLocalSetUp(user->vdm));
  }
  // put velocity and configuration space coordinate into Swarm
  for (c = 0; c < cEnd - cStart; ++c) {
    const PetscInt cell = c + cStart;
    PetscInt      *pidx, Npc;
    PetscReal      centroid[3], volume;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, cell, &volume, centroid, NULL));
    if (0) { // Landau
    } else {
      for (cv = 0; cv < vEnd - vStart; ++cv) {
        const PetscInt p = pidx[cv];
        PetscInt           Nc;
        const PetscScalar *array_v;
        PetscScalar       *coords_v = NULL;
        PetscBool          isDG;
        PetscCall(DMPlexGetCellCoordinates(user->vdm, cv, &isDG, &Nc, &array_v, &coords_v));
        for (d = 0; d < swdim; ++d) {
          x[p * swdim + d] = centroid[d];
          if (user->fake_1D && d > 0) v[p * swdim + d] = 0;
          else v[p * swdim + d] = (coords_v[1] + coords_v[0])/2;
        }
        //printf ("\t\t\t%d) (x,v) = %e, %e\n", p, x[p * swdim], v[p * swdim]);
        PetscCall(DMPlexRestoreCellCoordinates(user->vdm, cv, &isDG, &Nc, &array_v, &coords_v));
      }
    }
    PetscCall(PetscFree(pidx));
  }
  /* Setup Quadrature for spatial and velocity weight calculations*/
  PetscQuadrature  quad_x;
  PetscInt         Nq_x;
  const PetscReal *wq_x, *xq_x;
  PetscReal       *xq_x_extended;
  PetscReal        weightsum = 0., totalcellweight = 0., *weight_x, *weight_v;
  PetscReal        scale[2] = {user->cosine_coefficients[0], user->cosine_coefficients[1]};

  PetscCall(PetscCalloc2(cEnd - cStart, &weight_x, Np, &weight_v));
  if (user->fake_1D) PetscCall(PetscDTGaussTensorQuadrature(1, 1, 5, -1.0, 1.0, &quad_x));
  else PetscCall(PetscDTGaussTensorQuadrature(swdim, 1, 5, -1.0, 1.0, &quad_x));
  PetscCall(PetscQuadratureGetData(quad_x, NULL, NULL, &Nq_x, &xq_x, &wq_x));
  if (user->fake_1D) {
    PetscCall(PetscCalloc1(Nq_x * swdim, &xq_x_extended));
    for (PetscInt i = 0; i < Nq_x; ++i) xq_x_extended[i * swdim] = xq_x[i];
  }
  /* Integrate the density function to get the weights of particles in each cell */
  for (d = 0; d < swdim; ++d) xi0[d] = -1.0;
  for (c = cStart; c < cEnd; ++c) {
    PetscReal          v0_x[3], J_x[9], invJ_x[9], detJ_x, xr_x[3], den_x;
    PetscInt          *pidx, Npc, q;
    PetscInt           Ncx;
    const PetscScalar *array_x;
    PetscScalar       *coords_x = NULL;
    PetscBool          isDGx;
    weight_x[c] = 0.;

    PetscCall(DMPlexGetCellCoordinates(dm, c, &isDGx, &Ncx, &array_x, &coords_x));
    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0_x, J_x, invJ_x, &detJ_x));
    for (q = 0; q < Nq_x; ++q) {
      /*Transform quadrature points from ref space to real space (0,12.5664)*/
      if (user->fake_1D) CoordinatesRefToReal(swdim, swdim, xi0, v0_x, J_x, &xq_x_extended[q * swdim], xr_x);
      else CoordinatesRefToReal(swdim, swdim, xi0, v0_x, J_x, &xq_x[q * swdim], xr_x);

      /*Transform quadrature points from real space to ideal real space (0, 2PI/k)*/
      if (user->fake_1D) {
        if (user->twostream) PetscCall(PetscPDFCosine1D_TwoStream(xr_x, scale, &den_x));
        else PetscCall(PetscPDFCosine1D(xr_x, scale, &den_x));
        detJ_x = J_x[0];
      } else PetscCall(PetscPDFCosine2D(xr_x, scale, &den_x));
      /*We have to transform the quadrature weights as well*/
      weight_x[c] += den_x * (wq_x[q] * detJ_x);
    }
    // Get the cell numbering for consistent output between sequential and distributed tests
    IS              globalOrdering;
    const PetscInt *ordering;
    PetscCall(DMPlexGetCellNumbering(dm, &globalOrdering));
    PetscCall(ISGetIndices(globalOrdering, &ordering));
    //PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "c:%" PetscInt_FMT " [x_a,x_b] = %1.15f,%1.15f -> cell weight = %1.15f\n", ordering[c], (double)PetscRealPart(coords_x[0]), (double)PetscRealPart(coords_x[2]), (double)weight_x[c]));
    PetscCall(ISRestoreIndices(globalOrdering, &ordering));
    totalcellweight += weight_x[c];
    // Confirm the number of particles per spatial cell conforms to the size of the velocity grid
    PetscCheck(Npc == vEnd - vStart, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of particles %" PetscInt_FMT " in cell (rank %d/%d) != %" PetscInt_FMT " number of velocity vertices", Npc, rank, size, vEnd - vStart);

    /* Set weights to be gaussian in velocity cells (using exact solution) */
    for (cv = 0; cv < vEnd - vStart; ++cv) {
      PetscInt           Nc;
      const PetscScalar *array_v;
      PetscScalar       *coords_v = NULL;
      PetscBool          isDG;
      PetscCall(DMPlexGetCellCoordinates(user->vdm, cv, &isDG, &Nc, &array_v, &coords_v));
      const PetscInt p = pidx[cv];
      // Two stream function from 1/2pi v^2 e^(-v^2/2)
      if (user->twostream)
        weight_v[p] = 1. / (PetscSqrtReal(2 * PETSC_PI)) * ((coords_v[0] * PetscExpReal(-PetscSqr(coords_v[0]) / 2.)) - (coords_v[1] * PetscExpReal(-PetscSqr(coords_v[1]) / 2.))) - 0.5 * PetscErfReal(coords_v[0] / PetscSqrtReal(2.)) + 0.5 * (PetscErfReal(coords_v[1] / PetscSqrtReal(2.)));
      else weight_v[p] = 0.5 * (PetscErfReal(coords_v[1] / PetscSqrtReal(2.)) - PetscErfReal(coords_v[0] / PetscSqrtReal(2.)));

      weight[p] = user->totalWeight * weight_v[p] * weight_x[c];
      if (weight[p] > 1.) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "weights: %g, %g, %g\n", user->totalWeight, weight_v[p], weight_x[c]));
      //PetscPrintf(PETSC_COMM_WORLD, "particle %"PetscInt_FMT": %g, weight_v: %g weight_x: %g\n", p, weight[p], weight_v[p], weight_x[p]);
      weightsum += weight[p];

      PetscCall(DMPlexRestoreCellCoordinates(user->vdm, cv, &isDG, &Nc, &array_v, &coords_v));
    }
    PetscCall(DMPlexRestoreCellCoordinates(dm, c, &isDGx, &Ncx, &array_x, &coords_x));
    PetscCall(PetscFree(pidx));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscReal global_cellweight, global_weightsum;
  PetscCallMPI(MPIU_Allreduce(&totalcellweight, &global_cellweight, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(&weightsum, &global_weightsum, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "particle weight sum = %1.10f cell weight sum = %1.10f\n", (double)global_cellweight, (double)global_weightsum));
  if (user->fake_1D) PetscCall(PetscFree(xq_x_extended));
  PetscCall(PetscFree2(weight_x, weight_v));
  PetscCall(PetscQuadratureDestroy(&quad_x));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeConstants(DM sw, AppCtx *user)
{
  DM         dm;
  PetscInt  *species;
  PetscReal *weight, totalCharge = 0., totalWeight = 0., gmin[3], gmax[3], global_charge, global_weight;
  PetscInt   Np, dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    totalWeight += weight[p];
    totalCharge += user->charges[species[p]] * weight[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  {
    Parameter *param;
    PetscReal  Area;

    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    switch (dim) {
    case 1:
      Area = (gmax[0] - gmin[0]);
      break;
    case 2:
      if (user->fake_1D) {
        Area = (gmax[0] - gmin[0]);
      } else {
        Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]);
      }
      break;
    case 3:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]) * (gmax[2] - gmin[2]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", dim);
    }
    PetscCallMPI(MPIU_Allreduce(&totalWeight, &global_weight, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&totalCharge, &global_charge, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim = %" PetscInt_FMT "\ttotalWeight = %f, user->charges[species[0]] = %f\ttotalCharge = %f, Total Area = %f\n", dim, (double)global_weight, (double)user->charges[0], (double)global_charge, (double)Area));
    param->sigma = PetscAbsReal(global_charge / (Area));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma: %g\n", (double)param->sigma));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(x0,v0,t0,m0,q0,phi0): (%e, %e, %e, %e, %e, %e) - (P, V) = (%e, %e)\n", (double)param->x0, (double)param->v0, (double)param->t0, (double)param->m0, (double)param->q0, (double)param->phi0, (double)param->poissonNumber,
                          (double)param->vlasovNumber));
    user->m_n0 = param->m0 * user->masses[0] / global_weight; // place where we have everything : m0, masses, weight sum
  }
  /* Setup Constants */
  {
    PetscDS    ds;
    Parameter *param;
    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    PetscScalar constants[NUM_CONSTANTS];
    constants[SIGMA]   = param->sigma;
    constants[V0]      = param->v0;
    constants[T0]      = param->t0;
    constants[X0]      = param->x0;
    constants[M0]      = param->m0;
    constants[Q0]      = param->q0;
    constants[PHI0]    = param->phi0;
    constants[POISSON] = param->poissonNumber;
    constants[VLASOV]  = param->vlasovNumber;
    constants[M_N0]    = user->m_n0;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeVelocities_Fake1D(DM sw, AppCtx *user)
{
  DM         dm;
  PetscReal *v;
  PetscInt  *species, cStart, cEnd;
  PetscInt   dim, Np;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscRandom rnd;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, 0, 1.));
  PetscCall(PetscRandomSetFromOptions(rnd));

  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal a[3] = {0., 0., 0.}, vel[3] = {0., 0., 0.};

    PetscCall(PetscRandomGetValueReal(rnd, &a[0]));
    if (user->perturbed_weights) {
      PetscCall(PetscPDFSampleConstant1D(a, NULL, vel));
    } else {
      PetscCall(PetscPDFSampleGaussian1D(a, NULL, vel));
    }
    v[p * dim] = vel[0];
  }
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscReal v0[2] = {1., 0.};
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initCoordinates", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initVelocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "E_field", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "potential", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "charges", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(DMSetFromOptions(*sw));
  user->swarm = *sw;
  if (user->perturbed_weights) {
    PetscCall(InitializeParticles_PerturbedWeights(*sw, user));
  } else {
    PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
    PetscCall(DMSwarmInitializeCoordinates(*sw));
    if (user->fake_1D) {
      PetscCall(InitializeVelocities_Fake1D(*sw, user));
    } else {
      PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
    }
  }
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  {
    Vec gc, gc0, gv, gv0;

    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(VecCopy(gc, gc0));
    PetscCall(VecViewFromOptions(gc, NULL, "-ic_x_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(VecViewFromOptions(gv, NULL, "-velocity_vec_view"));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initVelocity", &gv0));
    PetscCall(VecCopy(gv, gv0));
    PetscCall(VecViewFromOptions(gv, NULL, "-ic_v_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initVelocity", &gv0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Coulomb(SNES snes, DM sw, PetscReal E[])
{
  AppCtx     *user;
  PetscReal  *coords;
  PetscInt   *species, dim, Np, Ns;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)snes), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Coulomb code only works in serial");
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetApplicationContext(sw, (void *)&user));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal *pcoord = &coords[p * dim];
    PetscReal  pE[3]  = {0., 0., 0.};

    /* Calculate field at particle p due to particle q */
    for (PetscInt q = 0; q < Np; ++q) {
      PetscReal *qcoord = &coords[q * dim];
      PetscReal  rpq[3], r, r3, q_q;

      if (p == q) continue;
      q_q = user->charges[species[q]] * 1.;
      for (PetscInt d = 0; d < dim; ++d) rpq[d] = pcoord[d] - qcoord[d];
      r = DMPlex_NormD_Internal(dim, rpq);
      if (r < PETSC_SQRT_MACHINE_EPSILON) continue;
      r3 = PetscPowRealInt(r, 3);
      for (PetscInt d = 0; d < dim; ++d) pE[d] += q_q * rpq[d] / r3;
    }
    for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = pE[d];
  }
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Primal(SNES snes, DM sw, PetscReal E[])
{
  DM              dm;
  AppCtx         *user;
  PetscDS         ds;
  PetscFE         fe;
  Mat             M_p, M;
  Vec             phi, locPhi, rho, f;
  PetscReal      *coords;
  PetscInt        dim, cStart, cEnd, Np;
  PetscQuadrature q;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, (void *)&user));

  KSP         ksp;
  Vec         rho0;
  char        oldField[PETSC_MAX_PATH_LEN];
  const char *tmp;

  /* Create the charges rho */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSwarmVectorGetField(sw, &tmp));
  PetscCall(PetscStrncpy(oldField, tmp, PETSC_MAX_PATH_LEN));
  PetscCall(DMSwarmVectorDefineField(sw, "w_q"));
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMSwarmVectorDefineField(sw, oldField));

  PetscCall(DMCreateMassMatrix(dm, dm, &M));
  PetscCall(DMGetGlobalVector(dm, &rho0));
  PetscCall(PetscObjectSetName((PetscObject)rho0, "Charge density (rho0) from Primal Compute"));
  PetscCall(DMGetGlobalVector(dm, &rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));

  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));
  PetscCall(MatMultTranspose(M_p, f, rho));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(M, NULL, "-m_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj_"));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rho, rho0));
  PetscCall(VecViewFromOptions(rho0, NULL, "-rho0_view"));

  PetscInt           rhosize;
  PetscReal         *charges;
  const PetscScalar *rho_vals;
  PetscCall(DMSwarmGetField(sw, "charges", NULL, NULL, (void **)&charges));
  PetscCall(VecGetLocalSize(rho0, &rhosize));
  PetscCall(VecGetArrayRead(rho0, &rho_vals));
  for (PetscInt c = 0; c < rhosize; ++c) charges[c] = rho_vals[c];
  PetscCall(VecRestoreArrayRead(rho0, &rho_vals));
  PetscCall(DMSwarmRestoreField(sw, "charges", NULL, NULL, (void **)&charges));

  PetscCall(VecScale(rho, -1.0));

  PetscCall(VecViewFromOptions(rho0, NULL, "-rho0_view"));
  PetscCall(VecViewFromOptions(rho, NULL, "-rho_view"));
  PetscCall(DMRestoreGlobalVector(dm, &rho0));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M));
  if (user->resample_period && !user->M_p0) {
    PetscCall(PetscInfo(sw, "Save M_p\n"));
    user->M_p0 = M_p; // this now has the correct size
  } else {
    PetscCall(MatDestroy(&M_p));
  }
  PetscCall(DMGetGlobalVector(dm, &phi));
  PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, rho, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rho));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscInt           phisize;
  PetscReal         *pot;
  const PetscScalar *phi_vals;
  PetscCall(DMSwarmGetField(sw, "potential", NULL, NULL, (void **)&pot));
  PetscCall(VecGetLocalSize(phi, &phisize));
  PetscCall(VecGetArrayRead(phi, &phi_vals));
  for (PetscInt c = 0; c < phisize; ++c) pot[c] = phi_vals[c];
  PetscCall(VecRestoreArrayRead(phi, &phi_vals));
  PetscCall(DMSwarmRestoreField(sw, "potential", NULL, NULL, (void **)&pot));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phi));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscTabulation tab;
    PetscScalar    *clPhi = NULL;
    PetscReal      *pcoord, *refcoord;
    PetscReal       v[3], J[9], invJ[9], detJ;
    PetscInt       *points;
    PetscInt        Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    for (PetscInt cp = 0; cp < Ncp; ++cp)
      for (PetscInt d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    PetscCall(DMPlexCoordinatesToReference(dm, c, Ncp, pcoord, refcoord));
    PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v, J, invJ, &detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscReal *basisDer = tab->T[1];
      const PetscInt   p        = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEGetQuadrature(fe, &q));
      PetscCall(PetscFEFreeInterpolateGradient_Static(fe, basisDer, clPhi, dim, invJ, NULL, cp, &E[p * dim]));
      for (PetscInt d = 0; d < dim; ++d) {
        E[p * dim + d] *= -1.0;
        if (user->fake_1D && d > 0) E[p * dim + d] = 0;
      }
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(PetscFree(points));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Mixed(SNES snes, DM sw, PetscReal E[])
{
  AppCtx         *user;
  DM              dm, potential_dm;
  KSP             ksp;
  IS              potential_IS;
  PetscDS         ds;
  PetscFE         fe;
  PetscFEGeom     feGeometry;
  Mat             M_p, M;
  Vec             phi, locPhi, rho, f, temp_rho, rho0;
  PetscQuadrature q;
  PetscReal      *coords, *pot;
  PetscInt        dim, cStart, cEnd, Np, fields = 1;
  char            oldField[PETSC_MAX_PATH_LEN];
  const char     *tmp;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &user));

  /* Create the charges rho */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(dm, &rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));

  PetscCall(DMCreateSubDM(dm, 1, &fields, &potential_IS, &potential_dm));

  PetscCall(DMSwarmVectorGetField(sw, &tmp));
  PetscCall(PetscStrncpy(oldField, tmp, PETSC_MAX_PATH_LEN));
  PetscCall(DMSwarmVectorDefineField(sw, "w_q"));
  PetscCall(DMCreateMassMatrix(sw, potential_dm, &M_p));
  PetscCall(DMSwarmVectorDefineField(sw, oldField));

  PetscCall(DMCreateMassMatrix(potential_dm, potential_dm, &M));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(M, NULL, "-m_view"));
  PetscCall(DMGetGlobalVector(potential_dm, &temp_rho));
  PetscCall(PetscObjectSetName((PetscObject)temp_rho, "Mf"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(MatMultTranspose(M_p, f, temp_rho));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(DMGetGlobalVector(potential_dm, &rho0));
  PetscCall(PetscObjectSetName((PetscObject)rho0, "Charge density (rho0) from Mixed Compute"));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj"));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, temp_rho, rho0));
  PetscCall(VecViewFromOptions(rho0, NULL, "-rho0_view"));

  PetscInt           rhosize;
  PetscReal         *charges;
  const PetscScalar *rho_vals;
  Parameter         *param;
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  PetscCall(DMSwarmGetField(sw, "charges", NULL, NULL, (void **)&charges));
  PetscCall(VecGetLocalSize(rho0, &rhosize));

  /* Integral over reference element is size 1.  Reference element area is 4.  Scale rho0 by 1/4 because the basis function is 1/4 */
  PetscCall(VecScale(rho0, 0.25));
  PetscCall(VecGetArrayRead(rho0, &rho_vals));
  for (PetscInt c = 0; c < rhosize; ++c) charges[c] = rho_vals[c];
  PetscCall(VecRestoreArrayRead(rho0, &rho_vals));
  PetscCall(DMSwarmRestoreField(sw, "charges", NULL, NULL, (void **)&charges));

  PetscCall(VecISCopy(rho, potential_IS, SCATTER_FORWARD, temp_rho));
  PetscCall(VecScale(rho, 0.25));
  PetscCall(VecViewFromOptions(rho0, NULL, "-rho0_view"));
  PetscCall(VecViewFromOptions(temp_rho, NULL, "-temprho_view"));
  PetscCall(VecViewFromOptions(rho, NULL, "-rho_view"));
  PetscCall(DMRestoreGlobalVector(potential_dm, &temp_rho));
  PetscCall(DMRestoreGlobalVector(potential_dm, &rho0));

  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&M));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&potential_dm));
  PetscCall(ISDestroy(&potential_IS));

  PetscCall(DMGetGlobalVector(dm, &phi));
  PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, rho, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rho));

  PetscInt           phisize;
  const PetscScalar *phi_vals;
  PetscCall(DMSwarmGetField(sw, "potential", NULL, NULL, (void **)&pot));
  PetscCall(VecGetLocalSize(phi, &phisize));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));
  PetscCall(VecGetArrayRead(phi, &phi_vals));
  for (PetscInt c = 0; c < phisize; ++c) pot[c] = phi_vals[c];
  PetscCall(VecRestoreArrayRead(phi, &phi_vals));
  PetscCall(DMSwarmRestoreField(sw, "potential", NULL, NULL, (void **)&pot));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phi));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(PetscFECreateCellGeometry(fe, q, &feGeometry));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscTabulation tab;
    PetscScalar    *clPhi = NULL;
    PetscReal      *pcoord, *refcoord;
    PetscInt       *points;
    PetscInt        Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    for (PetscInt cp = 0; cp < Ncp; ++cp)
      for (PetscInt d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    PetscCall(DMPlexCoordinatesToReference(dm, c, Ncp, pcoord, refcoord));
    PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, q, feGeometry.v, feGeometry.J, feGeometry.invJ, feGeometry.detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));

    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clPhi, &feGeometry, cp, &E[p * dim]));
      PetscCall(PetscFEPushforward(fe, &feGeometry, 1, &E[p * dim]));
      for (PetscInt d = 0; d < dim; ++d) {
        E[p * dim + d] *= -2.0;
        if (user->fake_1D && d > 0) E[p * dim + d] = 0;
      }
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(PetscFree(points));
  }
  PetscCall(PetscFEDestroyCellGeometry(fe, &feGeometry));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles(SNES snes, DM sw, PetscReal E[])
{
  AppCtx  *ctx;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(PetscInfo(sw, "E_field, charges and potential\n"));
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscAssertPointer(E, 3);
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(PetscArrayzero(E, Np * dim));

  switch (ctx->em) {
  case EM_PRIMAL:
    PetscCall(ComputeFieldAtParticles_Primal(snes, sw, E));
    break;
  case EM_COULOMB:
    PetscCall(ComputeFieldAtParticles_Coulomb(snes, sw, E));
    break;
  case EM_MIXED:
    PetscCall(ComputeFieldAtParticles_Mixed(snes, sw, E));
    break;
  case EM_NONE:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No solver for electrostatic model %s", EMTypes[ctx->em]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscReal   *coords, *vel;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal         *E, m_p = 1., q_p = -1.;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));

  PetscCall(ComputeFieldAtParticles(snes, sw, E));

  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = q_p * E[p * dim + d] / m_p;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
   TODO Now there is another term with w^2 from the electric field. I think we will need to invert the operator.
        Perhaps we can approximate the Jacobian using only the cellwise P-P gradient from Coulomb
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  DM               sw;
  const PetscReal *coords, *vel;
  PetscInt         dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0      = coords[p * dim + 0];
    const PetscReal vy0     = vel[p * dim + 1];
    const PetscReal omega   = vy0 / x0;
    PetscScalar     vals[4] = {0., 1., -PetscSqr(omega), 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 sw;
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p, d, dim;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(Xres, &Np));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      xres[p * dim + d] = v[p * dim + d];
      if (user->fake_1D && d > 0) xres[p * dim + d] = 0;
    }
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  DM                 sw;
  AppCtx            *user = (AppCtx *)ctx;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *x;
  const PetscReal   *coords, *vel;
  PetscReal         *E, m_p, q_p;
  PetscScalar       *vres;
  PetscInt           Np, p, dim, d;
  Parameter         *param;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  m_p = user->masses[0] * param->m0;
  q_p = user->charges[0] * param->q0;
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension must be 2");
  PetscCall(ComputeFieldAtParticles(snes, sw, E));

  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      vres[p * dim + d] = q_p * E[p * dim + d] / m_p;
      if (user->fake_1D && d > 0) vres[p * dim + d] = 0.;
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  /*
    Synchronized, ordered output for parallel/sequential test cases.
    In the 1D (on the 2D mesh) case, every y component should be zero.
  */
  if (user->checkVRes) {
    PetscBool pr = user->checkVRes > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt  step;

    PetscCall(TSGetStepNumber(ts, &step));
    if (pr) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step: %" PetscInt_FMT "\n", step));
    for (PetscInt p = 0; p < Np; ++p) {
      if (pr) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Residual: %.12g %.12g\n", (double)PetscRealPart(vres[p * dim + 0]), (double)PetscRealPart(vres[p * dim + 1])));
      PetscCheck(PetscAbsScalar(vres[p * dim + 1]) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Y velocity should be 0., not %g", (double)PetscRealPart(vres[p * dim + 1]));
    }
    if (pr) PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetProblem(TS ts)
{
  AppCtx *user;
  DM      sw;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  // Define unified system for (X, V)
  {
    Mat      J;
    PetscInt dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, 2 * Np * dim, 2 * Np * dim, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetBlockSize(J, 2 * dim));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, user));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, user));
    PetscCall(MatDestroy(&J));
  }
  /* Define split system for X and V */
  {
    Vec             u;
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np, rstart;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(VecGetOwnershipRange(u, &rstart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, user));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(CreateSolution(ts));
  PetscCall(SetProblem(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode line(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  DM        sw, cdm;
  PetscInt  Np;
  PetscReal low[2], high[2];
  AppCtx   *user = (AppCtx *)ctx;

  sw = user->swarm;
  PetscCall(DMSwarmGetCellDM(sw, &cdm));
  // Get the bounding box so we can equally space the particles
  PetscCall(DMGetLocalBoundingBox(cdm, low, high));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  // shift it by h/2 so nothing is initialized directly on a boundary
  x[0] = ((high[0] - low[0]) / Np) * (p + 0.5);
  x[1] = 0.;
  return PETSC_SUCCESS;
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameters:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM       sw;
  Vec      u, gc, gv, gc0, gv0;
  IS       isx, isv;
  PetscInt dim;
  AppCtx  *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  if (useInitial) {
    PetscReal v0[2] = {1., 0.}; /* this does not seem needed */
    if (user->perturbed_weights) {
      PetscCall(InitializeParticles_PerturbedWeights(sw, user));
    } else {
      PetscCall(DMSwarmComputeLocalSizeFromOptions(sw));
      PetscCall(DMSwarmInitializeCoordinates(sw));
      if (user->fake_1D) {
        PetscCall(InitializeVelocities_Fake1D(sw, user));
      } else {
        PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
      }
    }
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initVelocity", &gv0));
  if (useInitial) {
    PetscCall(VecCopy(gc, gc0));
    PetscCall(VecCopy(gv, gv0));
  }
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initVelocity", &gv0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sw;
  AppCtx            *user;
  const PetscScalar *u;
  const PetscReal   *coords, *vel;
  PetscScalar       *e;
  PetscReal          t;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    /* TODO generalize initial conditions and project into plane instead of assuming x-y */
    const PetscReal    r0    = DMPlex_NormD_Internal(dim, &coords[p * dim]);
    const PetscReal    th0   = PetscAtan2Real(coords[p * dim + 1], coords[p * dim + 0]);
    const PetscReal    v0    = DMPlex_NormD_Internal(dim, &vel[p * dim]);
    const PetscReal    omega = v0 / r0;
    const PetscReal    ct    = PetscCosReal(omega * t + th0);
    const PetscReal    st    = PetscSinReal(omega * t + th0);
    const PetscScalar *x     = &u[(p * 2 + 0) * dim];
    const PetscScalar *v     = &u[(p * 2 + 1) * dim];
    const PetscReal    xe[3] = {r0 * ct, r0 * st, 0.0};
    const PetscReal    ve[3] = {-v0 * st, v0 * ct, 0.0};
    PetscInt           d;

    for (d = 0; d < dim; ++d) {
      e[(p * 2 + 0) * dim + d] = x[d] - xe[d];
      e[(p * 2 + 1) * dim + d] = v[d] - ve[d];
    }
    if (user->error) {
      const PetscReal en   = 0.5 * DMPlex_DotRealD_Internal(dim, v, v);
      const PetscReal exen = 0.5 * PetscSqr(v0);
      PetscCall(PetscPrintf(comm, "t %.4g: p%" PetscInt_FMT " error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g (%.10lf%%)\n", (double)t, p, (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 0) * dim]), (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 1) * dim]), (double)x[0], (double)x[1], (double)v[0], (double)v[1], (double)xe[0], (double)xe[1], (double)ve[0], (double)ve[1], (double)en, (double)exen, (double)(PetscAbsReal(exen - en) * 100. / exen)));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// post step
static PetscErrorCode MigrateParticles(TS ts)
{
  DM               sw, cdm;
  const PetscReal *L;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec        u, gc, gv, position, momentum;
    IS         isx, isv;
    PetscReal *pos, *mom;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(VecGetSubVector(u, isx, &position));
    PetscCall(VecGetSubVector(u, isv, &momentum));
    PetscCall(VecGetArray(position, &pos));
    PetscCall(VecGetArray(momentum, &mom));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));

    PetscCall(DMSwarmGetCellDM(sw, &cdm));
    PetscCall(DMGetPeriodicity(cdm, NULL, NULL, &L));
    if ((L[0] || L[1]) >= 0.) {
      PetscReal *x, *v, upper[3], lower[3];
      PetscInt   Np, dim;

      PetscCall(DMSwarmGetLocalSize(sw, &Np));
      PetscCall(DMGetDimension(cdm, &dim));
      PetscCall(DMGetBoundingBox(cdm, lower, upper));
      PetscCall(VecGetArray(gc, &x));
      PetscCall(VecGetArray(gv, &v));
      for (PetscInt p = 0; p < Np; ++p) {
        for (PetscInt d = 0; d < dim; ++d) {
          if (pos[p * dim + d] < lower[d]) {
            x[p * dim + d] = pos[p * dim + d] + (upper[d] - lower[d]);
          } else if (pos[p * dim + d] > upper[d]) {
            x[p * dim + d] = pos[p * dim + d] - (upper[d] - lower[d]);
          } else {
            x[p * dim + d] = pos[p * dim + d];
          }
          PetscCheck(x[p * dim + d] >= lower[d] && x[p * dim + d] <= upper[d], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "p: %" PetscInt_FMT "x[%" PetscInt_FMT "] %g", p, d, (double)x[p * dim + d]);
          v[p * dim + d] = mom[p * dim + d];
        }
      }
      PetscCall(VecRestoreArray(gc, &x));
      PetscCall(VecRestoreArray(gv, &v));
    }
    PetscCall(VecRestoreArray(position, &pos));
    PetscCall(VecRestoreArray(momentum, &mom));
    PetscCall(VecRestoreSubVector(u, isx, &position));
    PetscCall(VecRestoreSubVector(u, isv, &momentum));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
  resampling
  */
typedef struct {
  Mat MpTrans;
  Mat Mp;
  Vec ff;
  Vec uu;
} MatShellCtx;

PetscErrorCode MatMultMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMult(matshellctx->MpTrans, matshellctx->ff, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy, Vec zz)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// pre solve
static PetscErrorCode Resample(TS ts)
{
  PetscInt     step;
  AppCtx      *user;
  DM           sw, sw_dm;
  PetscFunctionBeginUser;

  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetCellDM(sw, &sw_dm));
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(DMGetApplicationContext(sw, &user));
  if (user->resample_period && step % user->resample_period == 0) {
    PetscInt vdim, ph_dim, swdim;
    Vec u;
    PetscCall(TSGetSolution(ts, &u));
    PetscCall(VecViewFromOptions(u, NULL, "-resample_u_vec_view"));
    PetscCall(DMGetCoordinateDim(user->vdm, &vdim)); // 1 [or 2]
    PetscCall(DMGetCoordinateDim(sw, &swdim));
    ph_dim = 1 + vdim; // 1X + 1 [or 2]V
    if ( !user->M_p0 ) { // step = 0
      DM sw_phase, resample_plex;
      PetscInt np, vStart, vEnd, cStart, cEnd, Np, Np_cell, *cellid, Ns;
      const PetscReal *src_x, *src_v;
      PetscReal *phase_coords, vmin[3], vmax[3], xmin_re[3], xmax_re[3], xmin[3], xmax[3];
      PetscFE fe;
      // get vdm data
      PetscCall(DMGetBoundingBox(user->vdm, vmin, vmax));
      PetscCall(DMPlexGetHeightStratum(user->vdm, 0, &vStart, &vEnd));
      // phase space swarm - copy swarm with a (1) configuration coordinate and velocity coordinate (x,v)
      PetscCall(DMCreate(PetscObjectComm((PetscObject)sw), &sw_phase));
      PetscCall(DMSetType(sw_phase, DMSWARM));
      PetscCall(DMSetDimension(sw_phase, ph_dim));
      PetscCall(DMSwarmSetType(sw_phase, DMSWARM_PIC));
      PetscCall(DMSwarmFinalizeFieldRegister(sw_phase));
      PetscCall(DMSetApplicationContext(sw_phase, user));
      PetscCall(PetscObjectSetName((PetscObject)sw_phase, "Initial phase Swarm for resampling"));
      PetscCall(DMPlexGetHeightStratum(sw_dm, 0, &cStart, &cEnd)); // low order cells local, 1D cell array
      PetscCheck((cEnd - cStart)%2 == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "#cells mod 2 %d != 0", (int)(cEnd - cStart)%2);
      Np = (cEnd - cStart) * (vEnd - vStart);
      PetscCall(DMSwarmSetLocalSizes(sw_phase, Np, 0));
      Np_cell = vEnd - vStart;
      PetscCheck(Np_cell%2 == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "#part/cell mod 2 %d != 0", (int)Np_cell%2);
      PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
      PetscCall(DMSwarmSetNumSpecies(sw_phase, Ns));
      // resample plex
      PetscCall(DMCreate(PetscObjectComm((PetscObject)sw), &resample_plex));
      PetscCall(DMSetType(resample_plex, DMPLEX));
      PetscCall(DMPlexSetOptionsPrefix(resample_plex, "resample_"));
      PetscCall(DMSetFromOptions(resample_plex));
      PetscCall(DMSetDimension(resample_plex, ph_dim));
      // hardwire order to 2 - not working (Matt)
      PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, ph_dim, 1, PETSC_FALSE, NULL, 2, &fe));
      PetscCall(DMSetField(resample_plex, 0, NULL, (PetscObject)fe));
      PetscCall(PetscFEDestroy(&fe));
      PetscCall(DMCreateDS(resample_plex));
      PetscCall(PetscObjectSetName((PetscObject)resample_plex, "resample plex"));
      PetscCall(DMViewFromOptions(resample_plex, NULL, "-dm_view"));
      // check size, with vdm (could add 2V Landau)
      PetscCall(DMGetBoundingBox(resample_plex, xmin_re, xmax_re));
      PetscCall(DMGetBoundingBox(sw_dm, xmin, xmax));
      PetscCheck(xmin[0] == xmin_re[0] && vmin[0] == xmin_re[1], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "(min) Bounding box inconsistant %e == %e & %e == %e", (double)xmin[0], (double)xmin_re[0], (double)vmin[0], (double)xmin_re[1]);
      PetscCheck(xmax[0] == xmax_re[0] && vmax[0] == xmax_re[1], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "(max) Bounding box inconsistant %e == %e & %e == %e", (double)xmax[0], (double)xmax_re[0], (double)vmax[0], (double)xmax_re[1]);
      // continue with sw_phase, put in phase space coords
      PetscCall(DMSwarmSetCellDM(sw_phase, resample_plex));
      PetscCall(DMSwarmGetLocalSize(sw_phase, &np)); printf ("\t\t************ NP = %d\n", np);
      // duplicate swarm with velocity coordinate added & and cache x and v
      PetscCall(PetscMalloc2(np, &user->part_crd_x, np, &user->velocity));
      PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&src_x));
      PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&src_v));
      PetscCall(DMSwarmGetField(sw_phase, DMSwarmPICField_coor, NULL, NULL, (void **)&phase_coords));
      for (int p = 0; p < np; ++p) {
        user->part_crd_x[p]        = phase_coords[p * ph_dim + 0] = src_x[p * swdim + 0]; // x
        user->velocity[p*vdim + 0] = phase_coords[p * ph_dim + 1] = src_v[p * swdim + 0]; // v
        if (ph_dim == 3) user->velocity[p*vdim + 1] = phase_coords[p * ph_dim + 2] = 0; // todo
        printf ("\t\t%d) Add coord to phase space grid = %e, %e, swdim = %d\n", p, src_x[p * swdim + 0], src_v[p * swdim + 0], swdim);
      }
      PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&src_x));
      PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&src_v));
      PetscCall(DMSwarmRestoreField(sw_phase, DMSwarmPICField_coor, NULL, NULL, (void **)&phase_coords));
      // move vertices in velocity space for AMR: if (!user->uniform_velocity) {
      {
        Vec          phcoordinates,vcoordinates;
        PetscScalar *phcoords;
        const PetscScalar *vcoords;
        PetscInt     cdim, N, Nv, bs, i, v, k, nCells_x = (cEnd - cStart) / 2;
        // create 1D AMR particles (not Landau)
        PetscCall(DMGetCoordinateDim(resample_plex, &cdim)); // 2 or 3 = ph_dim
        PetscCall(DMGetCoordinates(resample_plex, &phcoordinates));
        PetscCall(VecGetLocalSize(phcoordinates, &N));
        PetscCall(VecGetBlockSize(phcoordinates, &bs));
        PetscCall(VecGetArray(phcoordinates, &phcoords));
        PetscCall(DMGetCoordinateDim(user->vdm, &vdim)); // 1
        PetscCall(DMGetCoordinates(user->vdm, &vcoordinates));
        PetscCall(VecGetLocalSize(vcoordinates, &Nv));
        PetscCall(VecGetBlockSize(vcoordinates, &bs));
        PetscCall(VecGetArrayRead(vcoordinates, &vcoords));
        for (i = v = 0; v < Nv ; v += 2*vdim) {    // Nv = 7: v = 0, 2, 4, 6
          for (k = 0 ; k < nCells_x ; k++, i += cdim) {  // periodic: Nc = 2: c = 0, 1
            PetscReal y = phcoords[i + 1];
            phcoords[i + 1] = vcoords[v];
            printf ("\t%d.%d.%d) x = %e, v = %e --> %e ; N = %d, Nv = %d, nCells_x = %d\n", i, k, v, phcoords[i + 0], y, phcoords[i + 1], N, Nv, nCells_x);
          }
        }
        PetscCall(VecRestoreArray(phcoordinates, &phcoords));
        PetscCall(VecRestoreArrayRead(vcoordinates, &vcoords));
        PetscCall(DMSetCoordinates(resample_plex, phcoordinates));
        PetscCall(DMGetCoordinatesLocalSetUp(resample_plex));
      }
      PetscCall(DMSwarmGetField(sw_phase, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
      for (int c = 0, p = 0; c < cEnd - cStart; ++c) {
        for (int s = 0; s < Ns; ++s) {
          for (int q = 0; q < Np_cell; ++q, ++p) cellid[p] = c; // wrong!!!!!!!!!
        }
      }
      PetscCall(DMSwarmRestoreField(sw_phase, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
      // PetscCall(DMSetFromOptions(sw_phase)); // call this??? only num species use on command like
PetscCall(DMSwarmGetLocalSize(sw_phase, &np)); printf ("\t\t************ NP = %d\n", np);
      PetscCall(DMSwarmMigrate(sw_phase, PETSC_TRUE));
PetscCall(DMSwarmGetLocalSize(sw_phase, &np)); printf ("\t\t************ NP = %d\n", np);
      PetscCall(DMSwarmGetLocalSize(sw_phase, &np));
      PetscCall(DMSwarmGetField(sw_phase, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
      for (int c = 0, p = 0; c < cEnd - cStart; ++c) {
        for (int s = 0; s < Ns; ++s) {
          for (int q = 0; q < Np_cell; ++q, ++p) printf ("\tcellid[%d/%d] = %d\n",p,np,cellid[p]);
        }
      }
      PetscCall(DMSwarmRestoreField(sw_phase, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));

      PetscCall(PetscObjectSetName((PetscObject)sw_phase, "Resample Particles (copy)"));
      PetscCall(DMViewFromOptions(sw_phase, NULL, "-sw_view"));
      // make M_p0
      PetscCall(DMCreateMassMatrix(sw_phase, resample_plex, &user->M_p0));
      PetscCall(DMSwarmViewXDMF(sw_phase, "sw_phase.xmf"));
      PetscCall(MatViewFromOptions(user->M_p0, NULL, "-resample_mat_view"));
      PetscInt M,N;
      PetscCall(MatGetSize(user->M_p0, &M, &N));
      PetscCall(DMDestroy(&sw_phase));
      PetscCall(DMDestroy(&resample_plex));
      PetscCall(PetscInfo(sw, "Setup resampling swarm, save coords, step %d, M_p[ %" PetscInt_FMT ", %" PetscInt_FMT "]\n", (int)step, M, N));
    } else { // resample - w_bar = M_p_bar (M_p_bar' M_p_bar)^-1 M_p' w
      Vec          ff, rho;
      KSP          ksp;
      Mat          MtM, D = NULL, M_p_bar = user->M_p0, M_p_;
      PetscInt     N, M, nzl;
      MatShellCtx *matshellctx = NULL;
      PC           pc;
      PetscCall(PetscInfo(sw, "Resampled to original particle grid\n"));
      // 1) particles to grid: rho = M_p' w
      PetscCall(DMGetGlobalVector(sw_dm, &rho));
      PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff));
      PetscCall(DMCreateMassMatrix(sw, sw_dm, &M_p_));
      PetscCall(MatMultTranspose(M_p_, ff, rho)); // rho <-- M'_p w
      PetscCall(MatDestroy(&M_p_));
PetscCall(PetscObjectSetName((PetscObject)ff, "pre resampling W"));
PetscCall(VecViewFromOptions(ff, NULL, "-resample_vec_view"));
      PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));
      // 2) pseudo-inverse, first part: (Mp' Mp)^-1
      PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
      PetscCall(KSPSetFromOptions(ksp));
      // create solver matrix
      PetscCall(MatGetLocalSize(M_p_bar, &M, &N));
      PetscCheck(N < M, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "need to N_p > N_v Moore-Penrose pseudo-inverse");
      PetscCall(PetscNew(&matshellctx));
      PetscCall(MatCreateVecs(M_p_bar, &matshellctx->uu, &matshellctx->ff));
      PetscCall(MatCreateShell(PetscObjectComm((PetscObject)sw_dm), N, N, PETSC_DECIDE, PETSC_DECIDE, matshellctx, &MtM));
      PetscCall(MatTranspose(M_p_bar, MAT_INITIAL_MATRIX, &matshellctx->MpTrans));
      matshellctx->Mp = M_p_bar;
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ));
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 1, NULL, &D));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_MpT_mat_view"));
      for (PetscInt i = 0; i < N; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar        dot = 0;
        PetscCall(MatGetRow(matshellctx->MpTrans, i, &nzl, &cols, &vals));
        for (PetscInt ii = 0; ii < nzl; ii++) dot += PetscSqr(vals[ii]);
        PetscCheck(dot > PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "empty row in pseudo-inverse %d", (int)i);
        PetscCall(MatSetValue(D, i, i, dot, INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscInfo(sw, "createMtMKSP Have %" PetscInt_FMT " eqs, nzl = %" PetscInt_FMT "\n", N, nzl));
      PetscCall(KSPSetOperators(ksp, MtM, D));
      PetscCall(MatViewFromOptions(D, NULL, "-ftop2_D_mat_view"));
      PetscCall(MatViewFromOptions(M_p_bar, NULL, "-ftop2_Mp_mat_view"));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_MpTranspose_mat_view"));
      // 3) pseudo-inverse solve
      PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access
      PetscCall(KSPSolve(ksp, rho, matshellctx->uu));
      PetscCall(DMRestoreGlobalVector(sw_dm, &rho));
      // 4) map back to particles: M_p_bar c'
      PetscCall(MatMult(M_p_bar, matshellctx->uu, ff));
      PetscCall(PetscObjectSetName((PetscObject)ff, "resampled W"));
      PetscCall(VecViewFromOptions(ff, NULL, "-resample_vec_view"));
      PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));
      // cleanup
      PetscCall(MatDestroy(&D));
      PetscCall(MatDestroy(&MtM));
      PetscCall(MatDestroy(&matshellctx->MpTrans));
      PetscCall(VecDestroy(&matshellctx->ff));
      PetscCall(VecDestroy(&matshellctx->uu));
      PetscCall(PetscFree(matshellctx));
      PetscCall(KSPDestroy(&ksp));
      // reset swarms coordinates, reset swarm size in parallel (todo)
      {
        PetscReal *x, *v;
        PetscInt np;
        Vec      gc, gv, gc0, gv0;
        IS       isx, isv;
        PetscCall(DMSwarmGetLocalSize(sw, &np));
        PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
        PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
        for (int p = 0; p < np; ++p) {
          x[p * swdim + 0] = user->part_crd_x[p];
          v[p * swdim + 0] = user->velocity[p];
printf("resampled particle %d: x = %e, v = %e\n", p, x[p * swdim], v[p * swdim]);
        }
        PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
        PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
        // like InitializeSolveAndSwarm
        PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
        PetscCall(DMSwarmTSRedistribute(ts));
        // reset u and other fields. Ignore 'species' (not used), and 'E_field' and 'potential' and 'charges' (done in ComputeFieldAtParticles)
        PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
        PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
        PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
        PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
        PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
        PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initVelocity", &gv0));
        PetscCall(VecCopy(gc, gc0));
        PetscCall(VecCopy(gv, gv0));
        PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
        PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
        PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
        PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
        PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
        PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initVelocity", &gv0));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sw;
  TS        ts;
  Vec       u;
  PetscReal dt;
  PetscInt  maxn;
  AppCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreatePoisson(dm, &user));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
  PetscCall(InitializeConstants(sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  if (user.efield_monitor) PetscCall(TSMonitorSet(ts, MonitorEField, &user, NULL));
  if (user.initial_monitor) PetscCall(TSMonitorSet(ts, MonitorInitialConditions, &user, NULL));
  if (user.monitor_positions) PetscCall(TSMonitorSet(ts, MonitorPositions_2D, &user, NULL));
  if (user.poisson_monitor) PetscCall(TSMonitorSet(ts, MonitorPoisson, &user, NULL));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxSteps(ts, &maxn));
  user.steps    = maxn;
  user.stepSize = dt;
  PetscCall(SetupContext(dm, sw, &user));
  PetscCall(DMSwarmVectorDefineField(sw, "velocity"));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetComputeExactError(ts, ComputeError));
  PetscCall(TSSetPostStep(ts, MigrateParticles));
  PetscCall(TSSetPreStep(ts, Resample));
  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(CheckNonNegativeWeights(sw, &user));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(SNESDestroy(&user.snes)); // move to destructor ?
  if (user.M_p0) PetscCall(MatDestroy(&user.M_p0));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(DestroyContext(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
    requires: !complex double

  # This tests that we can put particles in a box and compute the Coulomb force
  # Recommend -draw_size 500,500
   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -fake_1D -dm_plex_simplex 0 -dm_plex_box_faces 20,1 \
             -dm_plex_box_lower 0,-1 -dm_plex_box_upper 12.5664,1 \
             -dm_plex_box_bd periodic,none \
           -dm_swarm_coordinate_density constant -dm_swarm_num_particles 100 \
           -sigma 1.0e-8 -timeScale 2.0e-14 \
           -ts_type basicsymplectic -ts_basicsymplectic_type 1 \
             -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 0 \
           -output_step 50 -check_vel_res
     test:
       suffix: none_1d
       args: -em_type none -error
     test:
       suffix: coulomb_1d
       args: -em_type coulomb

   # for viewers
   #-ts_monitor_sp_swarm_phase -ts_monitor_sp_swarm -em_snes_monitor -ts_monitor_sp_swarm_multi_species 0 -ts_monitor_sp_swarm_retain 0
   testset:
     nsize: {{1 2}}
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -fake_1D -dm_plex_simplex 0 -dm_plex_box_faces 36,1 \
             -dm_plex_box_lower 0.,-0.5 -dm_plex_box_upper 12.5664,0.5 \
             -dm_plex_box_bd periodic,none \
           -vdm_plex_dim 1 -vdm_plex_simplex 0 -vdm_plex_box_faces 10 \
             -vdm_plex_box_lower -3 -vdm_plex_box_upper 3 \
           -dm_swarm_num_species 1 -twostream -charges -1.,1. -sigma 1.0e-8 \
             -cosine_coefficients 0.01,0.5 -perturbed_weights -total_weight 1. \
           -ts_type basicsymplectic -ts_basicsymplectic_type 2 \
             -ts_dt 0.01 -ts_max_time 5 -ts_max_steps 10 \
           -em_snes_atol 1.e-15 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged \
           -output_step 1 -check_vel_res
     test:
       suffix: two_stream_c0
       args: -em_type primal -petscfe_default_quadrature_order 2 -petscspace_degree 2 -em_pc_type svd
     test:
       suffix: two_stream_rt
       requires: superlu_dist
       args: -em_type mixed \
               -potential_petscspace_degree 0 \
               -potential_petscdualspace_lagrange_use_moments \
               -potential_petscdualspace_lagrange_moment_order 2 \
               -field_petscspace_degree 2 -field_petscfe_default_quadrature_order 1 \
               -field_petscspace_type sum \
                 -field_petscspace_variables 2 \
                 -field_petscspace_components 2 \
                 -field_petscspace_sum_spaces 2 \
                 -field_petscspace_sum_concatenate true \
                 -field_sumcomp_0_petscspace_variables 2 \
                 -field_sumcomp_0_petscspace_type tensor \
                 -field_sumcomp_0_petscspace_tensor_spaces 2 \
                 -field_sumcomp_0_petscspace_tensor_uniform false \
                 -field_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
                 -field_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
                 -field_sumcomp_1_petscspace_variables 2 \
                 -field_sumcomp_1_petscspace_type tensor \
                 -field_sumcomp_1_petscspace_tensor_spaces 2 \
                 -field_sumcomp_1_petscspace_tensor_uniform false \
                 -field_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
                 -field_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
               -field_petscdualspace_form_degree -1 \
               -field_petscdualspace_order 1 \
               -field_petscdualspace_lagrange_trimmed true \
             -em_snes_error_if_not_converged \
             -em_ksp_type preonly -em_ksp_error_if_not_converged \
             -em_pc_type fieldsplit -em_pc_fieldsplit_type schur \
               -em_pc_fieldsplit_schur_fact_type full -em_pc_fieldsplit_schur_precondition full \
               -em_fieldsplit_field_pc_type lu \
                 -em_fieldsplit_field_pc_factor_mat_solver_type superlu_dist \
               -em_fieldsplit_potential_pc_type svd

   # For verification, we use
   # -dm_plex_box_faces 100,1 -vdm_plex_box_faces 8000 -dm_swarm_num_particles 800000
   # -ts_monitor_sp_swarm_multi_species 0 -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500
   testset:
     nsize: {{1 2}}
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -fake_1D -dm_plex_simplex 0 -dm_plex_box_faces 10,1 \
             -dm_plex_box_lower 0.,-0.5 -dm_plex_box_upper 12.5664,0.5 \
             -dm_plex_box_bd periodic,none \
           -vdm_plex_dim 1 -vdm_plex_simplex 0 -vdm_plex_box_faces 10 \
             -vdm_plex_box_lower -10 -vdm_plex_box_upper 10 \
           -dm_swarm_num_species 1 -charges -1.,1. \
             -cosine_coefficients 0.01,0.5 -perturbed_weights -total_weight 1. \
           -ts_type basicsymplectic -ts_basicsymplectic_type 1 \
             -ts_dt 0.03 -ts_max_time 500 -ts_max_steps 1 \
           -em_snes_atol 1.e-12 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged \
           -output_step 1 -check_vel_res

     test:
       suffix: uniform_equilibrium_1d
       args: -cosine_coefficients 0.0,0.5 -em_type primal -petscspace_degree 1 -em_pc_type svd
     test:
       suffix: uniform_primal_1d
       args: -em_type primal -petscspace_degree 1 -em_pc_type svd
     test:
       requires: superlu_dist
       suffix: uniform_mixed_1d
       args: -em_type mixed \
               -potential_petscspace_degree 0 \
               -potential_petscdualspace_lagrange_use_moments \
               -potential_petscdualspace_lagrange_moment_order 2 \
               -field_petscspace_degree 2 -field_petscfe_default_quadrature_order 1 \
               -field_petscspace_type sum \
                 -field_petscspace_variables 2 \
                 -field_petscspace_components 2 \
                 -field_petscspace_sum_spaces 2 \
                 -field_petscspace_sum_concatenate true \
                 -field_sumcomp_0_petscspace_variables 2 \
                 -field_sumcomp_0_petscspace_type tensor \
                 -field_sumcomp_0_petscspace_tensor_spaces 2 \
                 -field_sumcomp_0_petscspace_tensor_uniform false \
                 -field_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
                 -field_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
                 -field_sumcomp_1_petscspace_variables 2 \
                 -field_sumcomp_1_petscspace_type tensor \
                 -field_sumcomp_1_petscspace_tensor_spaces 2 \
                 -field_sumcomp_1_petscspace_tensor_uniform false \
                 -field_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
                 -field_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
               -field_petscdualspace_form_degree -1 \
               -field_petscdualspace_order 1 \
               -field_petscdualspace_lagrange_trimmed true \
             -em_snes_error_if_not_converged \
             -em_ksp_type preonly -em_ksp_error_if_not_converged \
             -em_pc_type fieldsplit -em_pc_fieldsplit_type schur \
               -em_pc_fieldsplit_schur_fact_type full -em_pc_fieldsplit_schur_precondition full \
               -em_fieldsplit_field_pc_type lu \
                 -em_fieldsplit_field_pc_factor_mat_solver_type superlu_dist \
               -em_fieldsplit_potential_pc_type svd

TEST*/
