#include "petscmat.h"
#include "petscmath.h"
#include "petscoptions.h"
#include "petscriemannsolver.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petsctao.h"
#include "petscvec.h"
#include <petsc/private/localnetrpimpl.h> /*I "petscnetrp.h"  I*/

/*
   Implementation of basic Network Riemann Solver for the LWR traffic equation including support 
   for the more in roads that out roads. 
*/

typedef struct {
  Mat       TrafficDistribution;   /*  Negative of Standard Traffic Distribution Matrix */
  Vec       GammaMax, FluxOut, UB; /* Maximum Flux that can be obtained on a road. Used in the A * FluxStar <= GammaMax term */
  Vec       priority;              /* normalized priority vector*/
  PetscReal priority_norm;         /* norm of the original vector, if the orginal need to be viewed */
  PetscReal priority_weight;
} TrafficSolverCtx;

typedef struct {
  PetscBool                fluxmaximumcomputed;
  PetscReal                sigma, fluxsigma; /* point of the maximum of the flux, assumes flux is concave */
  NetRPTrafficDistribution traffic_distribution;
  NetRPTrafficPriorityVec  traffic_priority;
  PetscReal                priority_weight;
} TrafficCtx;

/* 
 Evaluate: E(\mathbf{\gamma}) = \sum_{e into v} -\gamma_e  + \epsilon \|\gamma_e - proj(\gamma_e,P) \|_2^2
 where: 
   proj is the orthogonal projection, 
   \epsilon is the priority weight
*/

static PetscErrorCode FormObjective(Tao tao, Vec X, PetscReal *f, void *ctx)
{
  const PetscScalar *x, *p;
  PetscInt           i, n;
  TrafficSolverCtx  *solver_ctx = (TrafficSolverCtx *)ctx;
  PetscScalar        innerprod;
  PetscReal          w = solver_ctx->priority_weight;

  PetscFunctionBegin;
  PetscCall(VecDot(X, solver_ctx->priority, &innerprod));
  PetscCall(VecGetLocalSize(X, &n));
  /* do the objective inplace manually as Petsc Has no function 
     to compute the norm of a projection in-place (I think) */
  PetscCall(VecGetArrayRead(solver_ctx->priority, &p));
  PetscCall(VecGetArrayRead(X, &x));
  *f = 0;
  for (i = 0; i < n; i++) { *f += -x[i] + w * PetscSqr(x[i] - innerprod * p[i]); }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(solver_ctx->priority, &p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  Evaluate \nabla E  = -\mathbf{1} + 2*\epsilon (\gamma_e -  proj(\gamma_e,P))
   where: 
    proj is the orthogonal projection, 
    \epsilon is the priority weight
*/
static PetscErrorCode FormObjectiveGradient(Tao tao, Vec X, Vec G, void *ctx)
{
  TrafficSolverCtx *solver_ctx = (TrafficSolverCtx *)ctx;
  PetscScalar       innerprod;
  PetscReal         w = solver_ctx->priority_weight;

  PetscFunctionBegin;
  PetscCall(VecDot(X, solver_ctx->priority, &innerprod));
  PetscCall(VecAXPBYPCZ(G, 2 * w, -2 * w * innerprod, 0.0, X, solver_ctx->priority));
  PetscCall(VecShift(G, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  Evaluate:
   E(\mathbf{\gamma}) = \sum_{e into v} -\gamma_e  + \epsilon \|\gamma_e - proj(\gamma_e,P) \|_2^2,
   \nabla E  = -\mathbf{1} + 2*\epsilon (\gamma_e -  proj(\gamma_e,P))
   where: 
    proj is the orthogonal projection, 
    \epsilon is the priority weight
*/
static PetscErrorCode FormmObjectiveAndGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x, *p;
  PetscInt           i, n;
  TrafficSolverCtx  *solver_ctx = (TrafficSolverCtx *)ctx;
  PetscScalar        innerprod;
  PetscReal          w = solver_ctx->priority_weight;

  PetscFunctionBegin;

  PetscCall(VecDot(X, solver_ctx->priority, &innerprod));
  PetscCall(VecGetLocalSize(X, &n));
  /* do the objective inplace manually as Petsc Has no function 
     to compute the norm of a projection in-place (I think) */
  PetscCall(VecGetArrayRead(solver_ctx->priority, &p));
  PetscCall(VecGetArrayRead(X, &x));
  *f = 0;
  for (i = 0; i < n; i++) { *f += -x[i] + w * PetscSqr(x[i] - innerprod * p[i]); }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(solver_ctx->priority, &p));

  PetscCall(VecAXPBYPCZ(G, 2 * w, -2 * w * innerprod, 0.0, X, solver_ctx->priority));
  PetscCall(VecShift(G, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate
  h(x) >= 0 where 
  h(x)  = GammaMax + A * x 

  where A is the negative of the usual Traffic Distribution Matrix as shown below. 

  In traffic network papers this is the condition that 
  A * \mathbf{\gamma} \in \Omega_{n+1} \times \hdots \times \Omega_{n+m}, i.e 

 0 \geq A * \mathbf{\gamma} \leq [\gamma_j^{max}(\rho_{j,0}) : j = n+1, \hdots , n+m] = GammaMax. 

 The lower bound is automatically satisified by the constraints on \gamma. 
*/
static PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  TrafficSolverCtx *solver_ctx = (TrafficSolverCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatMultAdd(solver_ctx->TrafficDistribution, X, solver_ctx->GammaMax, CI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  grad h = A, which is already formed 
*/
static PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre, void *ctx)
{
  TrafficSolverCtx *solver_ctx = (TrafficSolverCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatCopy(solver_ctx->TrafficDistribution, JI, SAME_NONZERO_PATTERN));
  if (JI != JIpre) { PetscCall(MatCopy(solver_ctx->TrafficDistribution, JIpre, SAME_NONZERO_PATTERN)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  variable bound routine 
*/

static PetscErrorCode VariableBound(Tao tao, Vec UL, Vec UB, void *ctx)
{
  TrafficSolverCtx *solver_ctx = (TrafficSolverCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(UL));
  PetscCall(VecCopy(solver_ctx->UB, UB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* traffic_distribution is garuanteed to exist by setup check */
static PetscErrorCode NetRPTraffic_ComputeDistribution(NetRP rp, PetscInt indeg, PetscInt outdeg, Mat trafficdistribution)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidHeaderSpecific(trafficdistribution, MAT_CLASSID, 4);
  PetscCallBack("NetRP Traffic callback traffic distribution", (*traffic->traffic_distribution)(rp, indeg, outdeg, trafficdistribution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTraffic_ComputePriority(NetRP rp, PetscInt indeg, PetscInt outdeg, Vec priority)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 1);
  PetscValidHeaderSpecific(priority, VEC_CLASSID, 4);
  PetscCallBack("NetRP Traffic callback priority vector", (*traffic->traffic_priority)(rp, indeg, outdeg, priority));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetUpTao_Traffic(NetRP rp, PetscInt indeg, PetscInt outdeg, void *ctx, Tao tao)
{
  PetscInt          dof        = 1;
  TrafficSolverCtx *solver_ctx = (TrafficSolverCtx *)ctx;
  Vec               CI;
  Mat               JacIneq;

  PetscFunctionBeginUser;
  PetscCall(TaoSetObjective(tao, FormObjective, ctx));
  PetscCall(TaoSetGradient(tao, NULL, FormObjectiveGradient, ctx));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormmObjectiveAndGradient, ctx));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, outdeg * dof, &CI));
  PetscCall(TaoSetInequalityConstraintsRoutine(tao, CI, FormInequalityConstraints, ctx));
  PetscCall(VecDestroy(&CI));

  PetscCall(MatDuplicate(solver_ctx->TrafficDistribution, MAT_COPY_VALUES, &JacIneq));
  PetscCall(TaoSetJacobianInequalityRoutine(tao, JacIneq, JacIneq, FormInequalityJacobian, ctx));
  PetscCall(MatDestroy(&JacIneq));

  PetscCall(TaoSetVariableBoundsRoutine(tao, VariableBound, ctx));

  PetscCall(TaoSetType(tao, TAOALMM));
  PetscCall(TaoSetTolerances(tao, 1e-3, 0.0, 0.0)); // this seems to be as low as I can go with default tao parameters. 
  PetscCall(TaoSetConstraintTolerances(tao, 1e-4, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetSolverCtx_Traffic(NetRP rp, PetscInt indeg, PetscInt outdeg, void **solver_ctx)
{
  TrafficCtx       *ctx = (TrafficCtx *)rp->data;
  TrafficSolverCtx *traffic_ctx;


  PetscFunctionBegin;
  PetscCall(PetscNew(&traffic_ctx));
  PetscCall(MatCreateDense(PETSC_COMM_SELF, outdeg, indeg, outdeg, indeg, NULL, &traffic_ctx->TrafficDistribution));
  PetscCall(NetRPTraffic_ComputeDistribution(rp, indeg, outdeg, traffic_ctx->TrafficDistribution));
  PetscCall(MatScale(traffic_ctx->TrafficDistribution, -1.));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, outdeg, &traffic_ctx->GammaMax));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, indeg, &traffic_ctx->UB));
  PetscCall(VecDuplicate(traffic_ctx->GammaMax, &traffic_ctx->FluxOut));

  PetscCall(VecDuplicate(traffic_ctx->UB, &traffic_ctx->priority));
  PetscCall(NetRPTraffic_ComputePriority(rp, indeg, outdeg, traffic_ctx->priority));
  PetscCall(VecNormalize(traffic_ctx->priority, &traffic_ctx->priority_norm));
  traffic_ctx->priority_weight = ctx->priority_weight;
  *solver_ctx                  = (void *)traffic_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPDestroySolverCtx_Traffic(NetRP rp, PetscInt indeg, PetscInt outdeg, void *solver_ctx)
{
  TrafficSolverCtx *traffic_ctx = (TrafficSolverCtx *)solver_ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&traffic_ctx->TrafficDistribution));
  PetscCall(VecDestroy(&traffic_ctx->GammaMax));
  PetscCall(VecDestroy(&traffic_ctx->FluxOut));
  PetscCall(VecDestroy(&traffic_ctx->UB));
  PetscCall(VecDestroy(&traffic_ctx->priority));
  PetscCall(PetscFree(traffic_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPPreSolve_Traffic(NetRP rp, PetscInt indeg, PetscInt outdeg, PetscBool *edgein, Vec U, void *solver_ctx)
{
  TrafficCtx        *ctx         = (TrafficCtx *)rp->data;
  TrafficSolverCtx  *traffic_ctx = (TrafficSolverCtx *)solver_ctx;
  const PetscScalar *u;
  PetscScalar       *gammamax, *ub;
  PetscInt           e, e_in, e_out, vdeg = indeg + outdeg;
  PetscReal         *flux;
  RiemannSolver      rs;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(traffic_ctx->UB, &ub));
  PetscCall(VecGetArray(traffic_ctx->GammaMax, &gammamax));

  PetscCall(NetRPGetFlux(rp, &rs));

  e_in  = 0;
  e_out = 0;
  for (e = 0; e < vdeg; e++) {
    if (edgein[e]) {
      if (u[e] <= ctx->sigma) {
        PetscCall(RiemannSolverEvaluateFlux(rs, &u[e], &flux));
        ub[e_in] = flux[0];
      } else {
        ub[e_in] = ctx->fluxsigma;
      }
      e_in++;
    } else {
      if (u[e] > ctx->sigma) {
        PetscCall(RiemannSolverEvaluateFlux(rs, &u[e], &flux));
        gammamax[e_out] = flux[0];
      } else {
        gammamax[e_out] = ctx->fluxsigma;
      }
      e_out++;
    }
  }
  PetscCall(VecRestoreArray(traffic_ctx->GammaMax, &gammamax));
  PetscCall(VecRestoreArray(traffic_ctx->UB, &ub));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPPostSolve_Traffic(NetRP rp, PetscInt indeg, PetscInt outdeg, PetscBool *edgein, Vec PostSolve, Vec Out, void *solver_ctx)
{
  TrafficSolverCtx  *traffic_ctx = (TrafficSolverCtx *)solver_ctx;
  PetscInt           e, vdeg = indeg + outdeg, e_in, e_out;
  const PetscScalar *fluxin, *fluxout;
  PetscScalar       *flux;

  PetscFunctionBegin;
  PetscCall(MatMult(traffic_ctx->TrafficDistribution, PostSolve, traffic_ctx->FluxOut));
  PetscCall(VecScale(traffic_ctx->FluxOut, -1));
  PetscCall(VecGetArrayRead(PostSolve, &fluxin));
  PetscCall(VecGetArrayRead(traffic_ctx->FluxOut, &fluxout));
  PetscCall(VecGetArray(Out, &flux));
  e_in  = 0;
  e_out = 0;
  for (e = 0; e < vdeg; e++) {
    if (edgein[e]) {
      flux[e] = fluxin[e_in++];
    } else {
      flux[e] = fluxout[e_out++];
    }
  }
  PetscCall(VecRestoreArray(Out, &flux));
  PetscCall(VecRestoreArrayRead(PostSolve, &fluxin));
  PetscCall(VecRestoreArrayRead(traffic_ctx->FluxOut, &fluxin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPView_Traffic(NetRP rp, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPDestroy_Traffic(NetRP rp)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(rp->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetDistribution_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetFluxMaximumPoint_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficGetFluxMaximumPoint_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetPriority_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetPriorityWeight_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficGetPriorityWeight_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTrafficSetDistribution_LWR(NetRP rp, NetRPTrafficDistribution traffic_distribution)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  traffic->traffic_distribution = traffic_distribution;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTrafficSetPriorityVec_LWR(NetRP rp, NetRPTrafficPriorityVec traffic_priority)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  traffic->traffic_priority = traffic_priority;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetPriorityWeight_LWR(NetRP rp, PetscReal weight)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  traffic->priority_weight = weight;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPGetPriorityWeight_LWR(NetRP rp, PetscReal *weight)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  *weight = traffic->priority_weight;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTrafficSetFluxMaximumPoint_LWR(NetRP rp, PetscReal sigma)
{
  TrafficCtx   *traffic = (TrafficCtx *)rp->data;
  RiemannSolver rs;
  PetscReal    *flux;

  PetscFunctionBegin;
  traffic->sigma = sigma;
  PetscCall(NetRPGetFlux(rp, &rs));
  PetscCall(RiemannSolverEvaluateFlux(rs, &sigma, &flux));
  traffic->fluxsigma           = flux[0];
  traffic->fluxmaximumcomputed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Flux_FormObjectiveAndGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  RiemannSolver      rs = (RiemannSolver)ctx;
  const PetscScalar *x;
  PetscScalar       *g, dflux;
  PetscReal         *flux;
  Mat                Dflux;

  PetscInt i, j;
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(RiemannSolverEvaluateFlux(rs, x, &flux));
  *f = -flux[0];
  PetscCall(RiemannSolverComputeJacobian(rs, x, &Dflux));
  PetscCall(VecRestoreArrayRead(X, &x));
  i = 0;
  j = 0;
  PetscCall(MatGetValues(Dflux, 1, &i, 1, &j, &dflux));
  PetscCall(VecGetArray(G, &g));
  g[0] = -dflux;
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTrafficComputeFluxMaximumPoint_LWR(NetRP rp)
{
  TrafficCtx        *traffic = (TrafficCtx *)rp->data;
  RiemannSolver      rs;
  Tao                tao;
  Vec                Sigma;
  const PetscScalar *sigma;
  PetscReal         *flux;

  PetscFunctionBegin;
  if (traffic->fluxmaximumcomputed) { PetscFunctionReturn(PETSC_SUCCESS); }

  /* Use Tao to maximize the flux, assumes concavity  of flux */
  PetscCall(NetRPGetFlux(rp, &rs));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &Sigma));
  PetscCall(VecZeroEntries(Sigma));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoAppendOptionsPrefix(tao, "netrp_traffic_flux_"));
  PetscCall(TaoSetSolution(tao, Sigma));
  PetscCall(VecDestroy(&Sigma));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, Flux_FormObjectiveAndGradient, (void *)rs));
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolution(tao, &Sigma));

  PetscCall(VecGetArrayRead(Sigma, &sigma));
  traffic->sigma = PetscRealPart(sigma[0]);
  PetscCall(VecRestoreArrayRead(Sigma, &sigma));

  PetscCall(RiemannSolverEvaluateFlux(rs, &traffic->sigma, &flux));
  traffic->fluxsigma           = flux[0];
  traffic->fluxmaximumcomputed = PETSC_TRUE;

  PetscCall(TaoDestroy(&tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPTrafficGetFluxMaximumPoint_LWR(NetRP rp, PetscReal *maxpoint)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  PetscCall(NetRPTrafficComputeFluxMaximumPoint_LWR(rp));
  *maxpoint = traffic->sigma;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetup_Traffic(NetRP rp)
{
  TrafficCtx *traffic = (TrafficCtx *)rp->data;

  PetscFunctionBegin;
  PetscCall(NetRPTrafficComputeFluxMaximumPoint_LWR(rp));
  PetscCheck(traffic->traffic_distribution, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Traffic Distribution Function not set. Set with NetRPTrafficSetDistribution() before SetUp()");
  PetscCheck(traffic->traffic_priority, PetscObjectComm((PetscObject)rp), PETSC_ERR_ARG_WRONGSTATE, "Traffic Priority Function not set. Set with NetRPTrafficSetDistribution() before SetUp()");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRPSetFromOptions_Traffic(PetscOptionItems *PetscOptionsObject, NetRP rp)
{
  PetscBool   flg;
  TrafficCtx *traffic = (TrafficCtx *)rp->data;
  PetscReal   w       = traffic->priority_weight;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "NetRP LWR Priority Traffic Options");
  PetscCall(PetscOptionsReal("-netrp_traffic_priority_weight", "Weight of the traffic priority in the objective", "NetRP", w, &w, &flg));
  if (flg) PetscCall(NetRPTrafficSetPriorityWeight(rp, w));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode NetRPCreate_TrafficLWR_Priority(NetRP rp)
{
  TrafficCtx *traffic;

  PetscFunctionBegin;
  PetscCall(PetscNew(&traffic));
  rp->data                  = (void *)traffic;
  rp->ops->setfromoptions   = NetRPSetFromOptions_Traffic;
  rp->ops->view             = NetRPView_Traffic;
  rp->ops->setuptao         = NetRPSetUpTao_Traffic;
  rp->ops->setsolverctx     = NetRPSetSolverCtx_Traffic;
  rp->ops->destroysolverctx = NetRPDestroySolverCtx_Traffic;
  rp->ops->PostSolve        = NetRPPostSolve_Traffic;
  rp->ops->PreSolve         = NetRPPreSolve_Traffic;
  rp->ops->destroy          = NetRPDestroy_Traffic;
  rp->ops->setup            = NetRPSetup_Traffic;
  rp->solvetype             = Optimization;
  rp->cacheU                = Yes_Manual;
  rp->cachetype             = DirectedVDeg;
  rp->physicsgenerality     = Generic; /* needs a rework in how these work as it is generic on single valued fluxes (that asssumes concavity) */

  traffic->priority_weight = 1e-2; /* Small weight large enough to be above the tao gatol */
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetDistribution_C", NetRPTrafficSetDistribution_LWR));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetFluxMaximumPoint_C", NetRPTrafficSetFluxMaximumPoint_LWR));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficGetFluxMaximumPoint_C", NetRPTrafficGetFluxMaximumPoint_LWR));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetPriority_C", NetRPTrafficSetPriorityVec_LWR));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficSetPriorityWeight_C", NetRPSetPriorityWeight_LWR));
  PetscCall(PetscObjectComposeFunction((PetscObject)rp, "NetRPTrafficGetPriorityWeight_C", NetRPGetPriorityWeight_LWR));

  PetscFunctionReturn(PETSC_SUCCESS);
}