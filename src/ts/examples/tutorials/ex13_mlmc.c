
static char help[] = "Uses MLMC on a 2d heat equation problem\n\n";

/*
 
 This code is originally a copy of the MATLAB code developed by Mohammad Motamed
 and presented at an Argonne National Laboratory tutorial on uncertainty quantification
 May 20 through May 24th, organized by Oana Marin who suggested coding this example.
 */

/*
 Include "petscmat.h" so that we can use matrices.
 automatically includes:
 petscsys.h    - base PETSc routines   petscvec.h    - vectors
 petscmat.h    - matrices
 petscis.h     - index sets            petscviewer.h - viewers
 */
#include <time.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscmatlab.h>

/*
 Data structure for MLMC algorithm
 */
#define MLMC_MAX_LEVELS 10
typedef struct {
    PetscInt  q;                   /* order of accuracy of deterministic solver */
    PetscInt  q1;                  /* convergence rate of weak error */
    PetscInt  q2;                  /* convergence rate of strong error */
    PetscInt  beta;                /* mesh refinement parameter */
    PetscReal gamma;               /* it appears in the work formula Wm = h^(-gamma) */
    PetscReal theta;               /* splitting parameter */
    PetscReal C_alpha;             /* constant in statistical error, given a failure probability */
    PetscInt  M0;                  /* initial number of grid realizations for the first three levels */
    PetscInt  Nl[MLMC_MAX_LEVELS]; /* number of realizations run on each level */
    PetscInt  L;                   /* total number of levels */
} MLMC;

/*
 Data structure for the solution of a single heat equation problem on a given grid with give parameters
 */
typedef struct {
    DM          da;
    PetscReal   hx,hy,dt,T;
    PetscInt    nx,ny;
    PetscInt    nt;
    Vec         x,y;      /* coordinates of local part of tensor product mesh */
    PetscReal   xQ;       /* location of QoI  */
//    PetscRandom rand;
    PetscScalar**  U;
    PetscScalar*   S;
    PetscInt       Lx, Ly;
    PetscReal      mu, sigma; /* sigma here stands for noise strength */
    PetscReal      lc;        /* corelation length for exponential or Gaussian covariance function*/
    PetscReal      lx,ly;     /* corelation length for separable exponential covariance function*/
    PetscScalar    b, c, rad;
    Mat            A;
} HeatSimulation;

//PetscBool useMatlabRand = PETSC_TRUE;

PetscErrorCode heat_solver(HeatSimulation*,PetscReal*);
PetscErrorCode mlmc_heat(HeatSimulation **,PetscInt,PetscInt,PetscReal[]);
PetscErrorCode fmlmc(MLMC *,HeatSimulation **,PetscReal,PetscReal*);
PetscErrorCode HeatSimulationCreate(MPI_Comm,PetscReal,PetscInt,PetscInt,HeatSimulation**);
PetscErrorCode HeatSimulationRefine(HeatSimulation*,HeatSimulation**);
PetscErrorCode HeatSimulationDestroy(HeatSimulation**);
PetscErrorCode KLSetup(HeatSimulation*);
PetscErrorCode FormInitialSolution(HeatSimulation*,Vec);
PetscErrorCode BuildA_CN(HeatSimulation*);
PetscErrorCode FormRHS_CN(HeatSimulation*,Vec,Vec);
PetscErrorCode BuildR(HeatSimulation*,Vec);
PetscErrorCode svd(PetscScalar**,PetscScalar**,PetscScalar**,PetscScalar*,PetscInt);

/*
 Options:
 
 -eps <tol> - error tolerance to use for MLMC
 -h0 <h0> - mesh discretization parameter on coarsest level
 -nx nx - number of grid points on the coarsest level  (use just one of this option and the option above)
 
 -w <w> - don't run MLMC, just compute the QoI for a given value of w on the coarsest level
 -Nw <Nw> -lw <lw> - don't run MLMC instead sample for QoI on a given level lw Nw times
 -use_matlab_rand - use the Matlab engine to create random numbers
 
 The -w, -use_matlab_rand and -Nw and -lw are mostly for testing against the Matlab version
 */
/* ----------------------------------------------------------------------------------------------------------------------- */
int main(int argc,char **args)
{
    PetscErrorCode  ierr;
    PetscReal       QoI,h0 = .125;  /* initial step size */
    PetscInt        nx,ny,i;
    HeatSimulation  *hs[MLMC_MAX_LEVELS];
    MLMC            mlmc;
//    PetscBool       flgw,flgNw,flglw;
//    PetscReal       w;
    
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
//    ierr = PetscOptionsGetBool(NULL,NULL,"-use_matlab_rand",&useMatlabRand,NULL);CHKERRQ(ierr);
//    if (useMatlabRand) {
//        ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),"rng('default')");CHKERRQ(ierr);
//    }
    
    mlmc.q       = 2;
    mlmc.q1      = mlmc.q;
    mlmc.q2      = 2*mlmc.q;
    mlmc.beta    = 2;
    mlmc.gamma   = 3;
    mlmc.theta   = 0.5;
    mlmc.C_alpha = 4;
    mlmc.M0      = 100;
    mlmc.L       = 3;
    
    ierr = PetscOptionsGetReal(NULL,NULL,"-h0",&h0,NULL);CHKERRQ(ierr);
    nx = ny = 1 + (PetscInt)PetscRoundReal(1.0/h0);
    ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&nx,NULL);
    ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&ny,NULL);
    
    ierr = PetscMemzero(hs,sizeof(hs));CHKERRQ(ierr);
    ierr = HeatSimulationCreate(PETSC_COMM_WORLD,.1,nx,ny,&hs[0]);CHKERRQ(ierr);
    
//    ierr = PetscOptionsGetReal(NULL,NULL,"-w",&w,&flgw);CHKERRQ(ierr);
//    ierr = PetscOptionsGetInt(NULL,NULL,"-Nw",&Nw,&flgNw);CHKERRQ(ierr);
//    ierr = PetscOptionsGetInt(NULL,NULL,"-lw",&lw,&flglw);CHKERRQ(ierr);
    
//    if (flgw) {
//        PetscReal QoI;
//        ierr = heat_solver(hs[0],w,&QoI);CHKERRQ(ierr);
//        ierr = PetscPrintf(PETSC_COMM_WORLD,"QoI for single heat solve %g using w %g\n",(double)QoI,(double)w);CHKERRQ(ierr);
//    } else if (flgNw || flglw) {
//        PetscReal sum[2];
//        for (i=1; i<lw; i++) {
//            ierr = HeatSimulationRefine(hs[i-1],&hs[i]);CHKERRQ(ierr);
//        }
//        ierr = mlmc_heat(hs,lw,Nw,sum);CHKERRQ(ierr);
//        ierr = PetscPrintf(PETSC_COMM_WORLD,"QoI %g for Nw random heat solves %D on level %D\n",(double)sum[0],Nw,lw);CHKERRQ(ierr);
//        for (i=1; i<lw; i++) {
//            ierr = HeatSimulationDestroy(&hs[i]);CHKERRQ(ierr);
//        }
//    } else {
#define MAX_EPS 10
        PetscReal eps[MAX_EPS];
        PetscInt  meps = MAX_EPS;
        
        eps[0] = .1;
        ierr = PetscOptionsGetRealArray(NULL,NULL,"-eps",eps,&meps,NULL);CHKERRQ(ierr);
        ierr = HeatSimulationRefine(hs[0],&hs[1]);CHKERRQ(ierr);
        ierr = HeatSimulationRefine(hs[1],&hs[2]);CHKERRQ(ierr);
        for (i=0; i<meps; i++) {
            ierr = fmlmc(&mlmc,hs,eps[i],&QoI);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"QoI for complete solve %g with %g EPS tolerance\n",(double)QoI,(double)eps[i]);CHKERRQ(ierr);
            
            ierr = PetscMemzero(mlmc.Nl,sizeof(mlmc.Nl));CHKERRQ(ierr);   /* Reset counters for MLMC */
            mlmc.L = 3;
//            if (useMatlabRand) {
//                ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),"rng('default')");CHKERRQ(ierr);
//            }
        }
        for (i=1; i<MLMC_MAX_LEVELS; i++) {
            ierr = HeatSimulationDestroy(&hs[i]);CHKERRQ(ierr);
        }
//    }
    ierr = HeatSimulationDestroy(&hs[0]);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
/* ----------------------------------------------------------------------------------------------------------------------- */
static PetscErrorCode HeatSimulationSetup(HeatSimulation *hs,PetscReal T)
{
    PetscErrorCode ierr;
    PetscInt       nx,ny,xs,ys,xn,yn,i;
    PetscReal      *x,*y;
    
    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(hs->da,NULL,&nx,&ny,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    hs->hx = 1.0/(nx-1);
    hs->hy = 1.0/(ny-1);
    hs->nx = nx;
    hs->ny = ny;
    hs->dt = 0.5*hs->hx;
    hs->T  = T;
    hs->nt = 1 + (PetscInt) PetscFloorReal(hs->T/hs->dt);
    hs->dt = hs->T/hs->nt;
    ierr = DMDAGetCorners(hs->da,&xs,&ys,NULL,&xn,&yn,NULL);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,xn,&hs->x);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,yn,&hs->y);CHKERRQ(ierr);
    ierr = VecGetArray(hs->x,&x);CHKERRQ(ierr);
    for (i=0; i<xn; i++) x[i] = hs->hx*(i+xs);
    ierr = VecRestoreArray(hs->x,&x);CHKERRQ(ierr);
    ierr = VecGetArray(hs->y,&y);CHKERRQ(ierr);
    for (i=0; i<yn; i++) y[i] = hs->hy*(i+ys);
    ierr = VecRestoreArray(hs->y,&y);CHKERRQ(ierr);
    hs->xQ = .5;
    /* domain geometry */
    hs->Lx       = 1;
    hs->Ly       = 1;
    /* physical parameters */
    hs->b        = 5.0;
    hs->c        = 30.0;
    hs->rad      = 0.5;
    hs->mu       = 0.0;
    hs->sigma    = 1.5;   /* sigma here stands for noise strength */
    hs->lc       = 2.0;   /* corelation length for exponential or Gaussian covariance function*/
    hs->lx       = 0.1;   /* corelation length for separable exponential covariance function*/
    hs->ly       = 0.1;   /* corelation length for separable exponential covariance function*/
    ierr = PetscOptionsGetReal(NULL,NULL,"-b",&(hs->b),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-c",&(hs->c),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-rad",&(hs->rad),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&(hs->mu),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-sigma",&(hs->sigma),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-lc",&(hs->lc),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-lx",&(hs->lx),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-ly",&(hs->ly),NULL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode HeatSimulationDestroy(HeatSimulation **hs)
{
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    if (!*hs) PetscFunctionReturn(0);
    ierr = DMDestroy(&(*hs)->da);CHKERRQ(ierr);
    ierr = VecDestroy(&(*hs)->x);CHKERRQ(ierr);
    ierr = VecDestroy(&(*hs)->y);CHKERRQ(ierr);
//    ierr = PetscRandomDestroy(&(*hs)->rand);CHKERRQ(ierr);
    ierr = PetscFree(*hs);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode HeatSimulationCreate(MPI_Comm comm,PetscReal T,PetscInt nx,PetscInt ny,HeatSimulation **hs)
{
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = PetscNew(hs);CHKERRQ(ierr);
    ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(*hs)->da);CHKERRQ(ierr);
    ierr = DMSetUp((*hs)->da);CHKERRQ(ierr);
//    if (!useMatlabRand) {
//        ierr = PetscRandomCreate(comm,&(*hs)->rand);CHKERRQ(ierr);
//    }
    ierr = HeatSimulationSetup(*hs,T);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates((*hs)->da,0.0,(*hs)->Lx,0.0,(*hs)->Ly,0.0,0.0);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup KL expansion
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KLSetup(*hs);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode HeatSimulationRefine(HeatSimulation *hs,HeatSimulation **hsf)
{
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = PetscNew(hsf);CHKERRQ(ierr);
    ierr = DMRefine(hs->da,PetscObjectComm((PetscObject)hs->da),&(*hsf)->da);CHKERRQ(ierr);
    ierr = HeatSimulationSetup(*hsf,hs->T);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates((*hsf)->da,0.0,hs->Lx,0.0,hs->Ly,0.0,0.0);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup KL expansion
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = KLSetup(*hsf);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode KLSetup(HeatSimulation *hs)
{
    DM             cda;
    DMDACoor2d   **coors;
    DMDALocalInfo  info;
    Vec            global;
    PetscInt       xs, xm, ys, ym, N2, Nx, Ny;
    PetscInt       i, j, i0, j0, i1, j1;
    PetscScalar  **Cov;
    PetscScalar  **V;                /* SVD */
    PetscScalar   *W;                /* Weights for quadrature (trapezoidal) */
    PetscScalar    x1, y1, x0, y0, rr;
    PetscInt       Lx     = hs->Lx;
    PetscInt       Ly     = hs->Ly;
    PetscReal      lc     = hs->lc;
//    PetscReal      lx     = hs->lx;
//    PetscReal      ly     = hs->ly;
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    
    /*------------------------------------------------------------------------
     Access coordinate field
     ---------------------------------------------------------------------*/
    ierr = DMDAGetLocalInfo(hs->da,&info);CHKERRQ(ierr);
    Nx   = info.mx;
    Ny   = info.my;
    N2   = Nx * Ny;
    
    /* Get local grid boundaries */
    ierr = DMDAGetCorners(hs->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(hs->da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinates(hs->da,&global);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
    
    /* allocate covariance matrix, its SVD associates */
    ierr = PetscMalloc1(N2,&Cov);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&Cov[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) Cov[i] = Cov[i-1]+N2;
    ierr = PetscMalloc1(N2,&(hs->U));CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&(hs->U)[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) (hs->U)[i] = (hs->U)[i-1]+N2;
    ierr = PetscMalloc1(N2,&V);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&V[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) V[i] = V[i-1]+N2;
    ierr = PetscMalloc1(N2,&(hs->S));CHKERRQ(ierr);
    
    /* Compute covariance function over the locally owned part of the grid */
    for (j0=ys; j0<ys+ym; j0++)
    {
        for (i0=xs; i0<xs+xm; i0++)
        {
            x0=coors[j0][i0].x;
            y0=coors[j0][i0].y;
            for (j1=ys; j1<ys+ym; j1++)
            {
                for (i1=xs; i1<xs+xm; i1++)
                {
                    x1=coors[j1][i1].x;
                    y1=coors[j1][i1].y;
//                    rr = PetscAbsReal(x1-x0) / lx + PetscAbsReal(y1-y0) / ly; //Seperable Exp
//                    rr = PetscSqrtReal(PetscPowReal(x1-x0,2) + PetscPowReal(y1-y0,2)) / lc; //Exp
                    rr = (PetscPowReal(x1-x0,2) + PetscPowReal(y1-y0,2)) / (2 * lc * lc); //Gaussian
                    Cov[j0*xm+i0][j1*xm+i1] = PetscExpReal(-rr);
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);

    /* Approximate the covariance integral operator via collocation and vertex-based quadrature */
    
    /* allocate quadrature weights W along the diagonal */
    ierr = PetscMalloc1(N2,&W);CHKERRQ(ierr);
    
    /* fill the weights (trapezoidal rule in 2d uniform mesh) */
    /* fill the first and the last*/
    W[0]=1; W[Nx-1]=1; W[N2-Nx]=1; W[N2-1]=1;
    for (i=1; i<Nx-1; i++) {W[i] = 2; W[N2-Nx+i]=2;}
    /* fill in between */
    for (i=0; i<Nx; i++)
    {
        for (j=1; j<Ny-1; j++) W[j*Nx+i] = 2.0 * W[i];
    }
    
    /* Scale W*/
    for (i = 0; i < N2; i++) W[i] = W[i] * ((Lx)*(Ly))/(4*(Nx-1)*(Ny-1));
    
    /* Combine W with covariance matrix Cov to form covariance operator K
     K = sqrt(W) * Cov * sqrt(W) (modifed to be symmetric)             */
    for (i=0; i<N2; i++)
    {
        for (j=0; j<N2; j++) Cov[i][j] = Cov[i][j] * PetscSqrtReal(W[i]) * PetscSqrtReal(W[j]);
    }
    
    /* Use SVD to decompose the PSD matrix K to get its eigen decomposition */
    svd(Cov,hs->U,V,hs->S,N2);
    
    /* Recover eigenvectors by divding sqrt(W) */
    for (i = 0; i < N2; i++)
    {
        for (j = 0; j < N2; j++) (hs->U)[i][j] = (hs->U)[i][j] / PetscSqrtReal(W[i]);
    }
    
    ierr = PetscFree(Cov);CHKERRQ(ierr);
//    ierr = PetscFree(U);CHKERRQ(ierr);
    ierr = PetscFree(V);CHKERRQ(ierr);
//    ierr = PetscFree(S);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode svd(PetscScalar **A_input, PetscScalar **U, PetscScalar **V, PetscScalar *S, PetscInt n)
/* svd.c: Perform a singular value decomposition A = USV' of square matrix.
 *
 * Input: The A_input matrix must has n rows and n columns.
 * Output: The product are U, S and V(not V').
 The S vector returns the singular values. */
{
    PetscInt  i, j, k, EstColRank = n, RotCount = n, SweepCount = 0, slimit = (n<120) ? 30 : n/4;
    PetscScalar eps = 1e-15, e2 = 10.0*n*eps*eps, tol = 0.1*eps, vt, p, x0, y0, q, r, c0, s0, d1, d2;
    PetscScalar *S2;
    PetscScalar **A;
    
    PetscFunctionBeginUser;
    
    PetscMalloc1(n,&S2);
    PetscMalloc1(2*n,&A);
    PetscMalloc1(2*n*n,&A[0]);
    for (i=1; i<n; i++) A[i] = A[i-1]+n;
    
    for (i=0; i<n; i++)
    {
        A[i] = malloc(n * sizeof(PetscScalar));
        A[n+i] = malloc(n * sizeof(PetscScalar));
        for (j=0; j<n; j++)
        {
            A[i][j]   = A_input[i][j];
            A[n+i][j] = 0.0;
        }
        A[n+i][i] = 1.0;
    }
    while (RotCount != 0 && SweepCount++ <= slimit) {
        RotCount = EstColRank*(EstColRank-1)/2;
        for (j=0; j<EstColRank-1; j++)
            for (k=j+1; k<EstColRank; k++) {
                p = q = r = 0.0;
                for (i=0; i<n; i++) {
                    x0 = A[i][j]; y0 = A[i][k];
                    p += x0*y0; q += x0*x0; r += y0*y0;
                }
                S2[j] = q; S2[k] = r;
                if (q >= r) {
                    if (q<=e2*S2[0] || fabs(p)<=tol*q)
                        RotCount--;
                    else {
                        p /= q; r = 1.0-r/q; vt = sqrt(4.0*p*p+r*r);
                        c0 = sqrt(0.5*(1.0+r/vt)); s0 = p/(vt*c0);
                        for (i=0; i<2*n; i++) {
                            d1 = A[i][j]; d2 = A[i][k];
                            A[i][j] = d1*c0+d2*s0; A[i][k] = -d1*s0+d2*c0;
                        }
                    }
                } else {
                    p /= r; q = q/r-1.0; vt = sqrt(4.0*p*p+q*q);
                    s0 = sqrt(0.5*(1.0-q/vt));
                    if (p<0.0) s0 = -s0;
                    c0 = p/(vt*s0);
                    for (i=0; i<2*n; i++) {
                        d1 = A[i][j]; d2 = A[i][k];
                        A[i][j] = d1*c0+d2*s0; A[i][k] = -d1*s0+d2*c0;
                    }
                }
            }
        while (EstColRank>2 && S2[EstColRank-1]<=S2[0]*tol+tol*tol) EstColRank--;
    }
    if (SweepCount > slimit)
        printf("Warning: Reached maximum number of sweeps (%d) in SVD routine...\n", slimit);
    for (i=0; i<n; i++) S[i] = PetscSqrtReal(S2[i]);
    for (i=0; i<n; i++)
    {
        for (j=0; j<n; j++)
        {
            U[i][j] = A[i][j]/S[j];
            V[i][j] = A[n+i][j];
        }
    }
    PetscFree(S2);
    PetscFree(A);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
/*
 * Lower tail quantile for standard normal distribution function.
 *
 * This function returns an approximation of the inverse cumulative
 * standard normal distribution function.  I.e., given P, it returns
 * an approximation to the X satisfying P = Pr{Z <= X} where Z is a
 * random variable from the standard normal distribution.
 *
 * The algorithm uses a minimax approximation by rational functions
 * and the result has a relative error whose absolute value is less
 * than 1.15e-9.
 *
 * Author:      Peter John Acklam
 * Time-stamp:  2002-06-09 18:45:44 +0200
 * E-mail:      jacklam@math.uio.no
 * WWW URL:     http://www.math.uio.no/~jacklam
 *
 * C implementation adapted from Peter's Perl version
 */

#include <math.h>
#include <errno.h>

/* Coefficients in rational approximations. */
static const double a[] =
{
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
};

static const double b[] =
{
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
};

static const double c[] =
{
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
};

static const double d[] =
{
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
};

#define LOW 0.02425
#define HIGH 0.97575
static PetscReal ltqnorm(PetscReal p)
{
    PetscReal q, r;
    PetscFunctionBeginUser;
    
    errno = 0;
    
    if (p < 0 || p > 1)
    {
        errno = EDOM;
        return 0.0;
    }
    else if (p == 0)
    {
        errno = ERANGE;
        return -HUGE_VAL /* minus "infinity" */;
    }
    else if (p == 1)
    {
        errno = ERANGE;
        return HUGE_VAL /* "infinity" */;
    }
    else if (p < LOW)
    {
        /* Rational approximation for lower region */
        q = sqrt(-2*log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    else if (p > HIGH)
    {
        /* Rational approximation for upper region */
        q  = sqrt(-2*log(1-p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    else
    {
        /* Rational approximation for central region */
        q = p - 0.5;
        r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    }
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode BuildA_CN(HeatSimulation *hs)
{
    DMDALocalInfo  info;
    PetscInt       i,j;
    PetscReal      hx,hy,sx,sy;
    PetscScalar    dt = hs->dt;
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = DMDAGetLocalInfo(hs->da,&info);CHKERRQ(ierr);
    hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
    hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            PetscInt    nc = 0;
            MatStencil  row,col[5];
            PetscScalar val[5];
            row.i = i; row.j = j;
            if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
                col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
            } else {
                col[nc].i = i-1; col[nc].j = j;   val[nc++] = -.5*sx*dt;
                col[nc].i = i+1; col[nc].j = j;   val[nc++] = -.5*sx*dt;
                col[nc].i = i;   col[nc].j = j-1; val[nc++] = -.5*sy*dt;
                col[nc].i = i;   col[nc].j = j+1; val[nc++] = -.5*sy*dt;
                col[nc].i = i;   col[nc].j = j;   val[nc++] = 1 + sx*dt + sy*dt;
            }
            ierr = MatSetValuesStencil(hs->A,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(hs->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(hs->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode FormRHS_CN(HeatSimulation *hs, Vec U, Vec RHS)
{
    PetscInt       i,j;
    PetscReal      hx,hy,sx,sy;
    PetscScalar   *u, *rhs;
    PetscScalar    dt = hs->dt;
    DMDALocalInfo  info;
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    ierr = VecGetArray(RHS,&rhs);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(hs->da,&info);CHKERRQ(ierr);
    hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
    hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
    
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
                rhs[info.mx*j+i] = u[info.mx*j+i];
            } else {
                rhs[info.mx*j+i] = u[info.mx*j+i]                            * (1-sx*dt-sy*dt)
                + ( u[info.mx*j+(i+1)] + u[info.mx*j+(i-1)] ) * .5*sx*dt
                + ( u[info.mx*(j+1)+i] + u[info.mx*(j-1)+i] ) * .5*sy*dt;
            }
        }
    }
    ierr = VecRestoreArray(RHS,&rhs);
    ierr = VecRestoreArray(U,&u);
    ierr = VecAssemblyBegin(RHS);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(RHS);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(HeatSimulation *hs, Vec U)
{
    PetscInt       i,j,xs,ys,xm,ym;
    PetscReal      hx,hy,x,y,r;
    DMDALocalInfo  info;
    PetscScalar    b     = hs->b;
    PetscScalar    c     = hs->c;
    PetscScalar    rad   = hs->rad;
    PetscScalar  **u;
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = DMDAGetLocalInfo(hs->da,&info);CHKERRQ(ierr);
    hx = 1.0/(PetscReal)(info.mx-1);
    hy = 1.0/(PetscReal)(info.my-1);
    
    /* Get pointers to vector data */
    ierr = DMDAVecGetArray(hs->da,U,&u);CHKERRQ(ierr);
    
    /* Get local grid boundaries */
    ierr = DMDAGetCorners(hs->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    
    /* Compute function over the locally owned part of the grid */
    for (j=ys; j<ys+ym; j++) {
        y = j*hy;
        for (i=xs; i<xs+xm; i++) {
            x = i*hx;
            r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
            if (r < rad) u[j][i] = b * PetscExpReal(-c*r*r);
            else u[j][i] = 0.0;
        }
    }
    
    /* Restore vectors */
    ierr = DMDAVecRestoreArray(hs->da,U,&u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode BuildR(HeatSimulation* hs, Vec R)
{
    DMDALocalInfo  info;
    PetscInt       i,j,Nx,Ny,N2;
    PetscReal      mu     = hs->mu;
    PetscReal      sigma  = hs->sigma;
    PetscScalar   *r,tmp;                /* vector issued from random field by KL expansion */
    PetscScalar  **U = hs->U;
    PetscScalar   *S = hs->S;
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    
    /*------------------------------------------------------------------------
     Access coordinate field
     ---------------------------------------------------------------------*/
    ierr = DMDAGetLocalInfo(hs->da,&info);CHKERRQ(ierr);
    Nx   = info.mx;
    Ny   = info.my;
    N2   = Nx * Ny;
    
    /* Get pointers to vector data */
    ierr = VecGetArray(R,&r);CHKERRQ(ierr);
    /* Generate normal random numbers by transforming from the uniform one */
    PetscScalar *rndu, *rndn;
    ierr = PetscMalloc1(N2,&rndu);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2,&rndn);CHKERRQ(ierr);
    //  PetscRandom rnd;
    //    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);
    /* force imaginary part of random number to always be zero; thus obtain reproducible results with real and complex numbers */
    //    ierr = PetscRandomSetInterval(rnd,0.0,1.0);CHKERRQ(ierr);
    //    ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
    //    ierr = PetscRandomGetValue(rnd,&rndu);CHKERRQ(ierr);
    
    time_t t;
    srand((unsigned) time(&t));rand();//initialize random number generator in C
    //        PetscScalar mean = 0;
    for (i = 0; i < N2; i++)
    {
        rndu[i] = (PetscScalar) rand()/RAND_MAX;
        rndn[i] = ltqnorm(rndu[i]);// transform from uniform(0,1) to normal(0,1) by N = norminv(U)
        //        printf("\nuniform random sample= %f\n",rndu[i]);
        //        printf("normal random sample= %f\n",rndn[i]);
        //                mean = mean + rndu[i];
    }
    //        mean = mean/N2;
    //        printf("%f\n",mean);
    
    /* Do KL expansion by combining the above eigen decomposition and normal random numbers */
    for (i = 0; i < N2; i++)
    {
        tmp=0.0;
        for (j = 0; j < N2; j++) tmp = tmp + U[i][j] * PetscSqrtReal(S[j]) * rndn[j];
        r[i] = mu + sigma * tmp;
    }
    for (j=0; j<Ny; j++)
    {for (i=0; i<Nx; i++)
    {
        if (i == 0 || j == 0 || i == Nx-1 || j == Ny-1) {
            r[Nx*j+i] = 0.0;}
    }
    }
    /* Pring the random vector r issued from random field by KL expansion */
    //    printf("\nRandom vector r issued from random field by KL expansion\n");
    //    for (i = 0; i < N2; i++) printf("%6.8f\n", r[i]);
    /* plot r in Matlab:
     >> Lx=1;Ly=1;Nx=8;Ny=8;[X,Y]=meshgrid(linspace(0,Lx,Nx),linspace(0,Ly,Ny));
     surf(X,Y,reshape(r,Nx,Ny)');shading interp;view(2);colorbar; */
    
    //    ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
    
    ierr = PetscFree(rndu);CHKERRQ(ierr);
    ierr = PetscFree(rndn);CHKERRQ(ierr);
    
    /* Restore vectors */
    ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
/*
 Solves a single heat equation on a fixed grid
 
 Input Parameters:
 hs - the information about the grid the problem is solved on including timestep etc
 
 Output Parameter:
 QoI - the quantity of interest computed from the solution
 */
PetscErrorCode heat_solver(HeatSimulation *hs,PetscReal *QoI)
{
    PetscErrorCode  ierr;
    Vec             u;               /* solution vector */
    PetscInt        mQ;
    const PetscReal **unew_array;
    
    PetscFunctionBeginUser;

    /* Crank-Nicolson scheme */
    /* Set Matrix A */
    ierr = DMSetMatType(hs->da,MATAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(hs->da,&hs->A);CHKERRQ(ierr);
    ierr = BuildA_CN(hs);CHKERRQ(ierr);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMCreateGlobalVector(hs->da,&u);CHKERRQ(ierr);
    ierr = FormInitialSolution(hs,u);CHKERRQ(ierr);
    
    Vec            r;                 /* random vector */
    Vec            unew, uold;        /* vector for time stepping */
    Vec            rhs;               /* rhs vector for Crank-Nicolson scheme */
    KSP            ksp;               /* KSP solver for Crank-Nicolson scheme */
    PetscInt       i;
    
    ierr = VecDuplicate(u,&unew);CHKERRQ(ierr);
    ierr = VecDuplicate(u,&uold);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(hs->da,&r);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(hs->da,&rhs);CHKERRQ(ierr);
    
    for (i=0; i<hs->nt+1; i++)
    {
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Time stepping
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = VecCopy(u,uold);CHKERRQ(ierr);
        ierr = BuildR(hs,r);
        ierr = VecScale(r,PetscSqrtReal(hs->dt));CHKERRQ(ierr);
        
        /* Euler scheme */
        //      ierr = MatMultAdd(user->A,uold,r,unew);CHKERRQ(ierr);
        //      ierr = VecAXPY(unew,1.0,uold);CHKERRQ(ierr);
        
        /* Crank-Nicolson scheme */
        ierr = FormRHS_CN(hs,uold,rhs);
        ierr = VecAXPY(rhs,1.0,r);CHKERRQ(ierr);
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);
        ierr = KSPSetOperators(ksp,hs->A,hs->A);
        ierr = KSPSetFromOptions(ksp);
        ierr = KSPSolve(ksp,rhs,unew);
        
        ierr = VecCopy(unew,u);CHKERRQ(ierr);
    }
    
    /* compute quantity of interest; currently only works sequentially */
    mQ   = (PetscInt)PetscRoundReal(0.5*(hs->nx-1)*(1.0+hs->xQ));
    ierr = DMDAVecGetArray(hs->da,unew,&unew_array);CHKERRQ(ierr);
    *QoI = unew_array[mQ][mQ];
    ierr = DMDAVecRestoreArray(hs->da,unew,&unew_array);CHKERRQ(ierr);
    
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&unew);CHKERRQ(ierr);
    ierr = VecDestroy(&uold);CHKERRQ(ierr);
    ierr = VecDestroy(&rhs);CHKERRQ(ierr); /* rhs vector for Crank-Nicolson scheme*/
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr); /* KSP solver for Crank-Nicolson scheme*/
    
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
/*
 Evaluates the sum of the differences of the QoI for two adjacent levels
 */
PetscErrorCode mlmc_heat(HeatSimulation **hs,PetscInt l,PetscInt M,PetscReal sum1[])
{
    PetscErrorCode     ierr;
    PetscInt           N1;
    PetscReal          Qf,Qc;
    
    PetscFunctionBeginUser;
    sum1[0] = sum1[1] = 0;
    for (N1=0; N1<M; N1++) {
//        if (useMatlabRand) {
//            ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),"x = rand(1,1);");CHKERRQ(ierr);
//            ierr = PetscMatlabEngineGetArray(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),1,1,&w,"x");CHKERRQ(ierr);
//        } else {
//            ierr = PetscRandomGetValue(hs[0]->rand,&w);CHKERRQ(ierr);
//        }
        ierr = heat_solver(hs[l],&Qf);CHKERRQ(ierr);
        if (l == 0) {
            Qc = 0;
        } else {
            ierr = heat_solver(hs[l-1],&Qc);CHKERRQ(ierr);
        }
        sum1[0] += (Qf-Qc);
        sum1[1] += (Qf-Qc)*(Qf-Qc);
    }
    PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------------------------------- */
PetscErrorCode fmlmc(MLMC *mlmc,HeatSimulation **hs,PetscReal eps,PetscReal *EQ)
{
    PetscErrorCode ierr;
    PetscInt       L,dNl[MLMC_MAX_LEVELS];
    PetscReal      suml[MLMC_MAX_LEVELS][2],mul[MLMC_MAX_LEVELS],Vl[MLMC_MAX_LEVELS],Wl[MLMC_MAX_LEVELS],sumVlWl;
    PetscInt       sumdNl = 0,i,l,Ns[MLMC_MAX_LEVELS];
    PetscBool      firstiteration = PETSC_TRUE;
    
    PetscFunctionBeginUser;
    L    = 3;
    ierr = PetscMemzero(mlmc->Nl,sizeof(mlmc->Nl));CHKERRQ(ierr);
    ierr = PetscMemzero(suml,sizeof(suml));CHKERRQ(ierr);
    ierr = PetscMemzero(dNl,sizeof(dNl));CHKERRQ(ierr);
    dNl[0] = dNl[1] = dNl[2] = mlmc->M0;
    
    for (i=0; i<L; i++) sumdNl += dNl[i];
    while (sumdNl) {
        ierr = PetscInfo1(NULL,"Starting the MLMC loop: total dNl %D individual dNl follow\n",sumdNl);CHKERRQ(ierr);
        for (i=0; i<L; i++) {
            if (firstiteration) {
                ierr = PetscInfo3(NULL,"MLMC Level dNl[%D] %D\n",i,dNl[i],i);CHKERRQ(ierr);
                firstiteration = PETSC_FALSE;
            } else {
                ierr = PetscInfo4(NULL,"MLMC Level dNl[%D] %D  Vl[%D] %g\n",i,dNl[i],i,(double)Vl[i]);CHKERRQ(ierr);
            }
        }
        /* update sample sums */
        for (l=0; l<L; l++) {
            if (dNl[l] > 0) {
                PetscReal sums[2];
                /*        sums = feval(mlmc_l,l,dNl(l+1),q,T,h0,beta); */
                mlmc_heat(hs,l,dNl[l],sums);CHKERRQ(ierr);
                mlmc->Nl[l] += dNl[l];
                suml[l][0]  += sums[0];
                suml[l][1]  += sums[1];
            }
        }
        
        /*  compute variances for levels l=0:L (formula (6) in lecture notes)
         mul = abs(suml(1,:)./Nl);
         Vl = max(0, suml(2,:)./Nl - mul.^2);
         */
        for (i=0; i<L; i++) mul[i] = PetscAbsReal(suml[i][0]/mlmc->Nl[i]);
        for (i=0; i<L; i++) Vl[i]  = PetscMax(0.0,suml[i][1]/mlmc->Nl[i] - mul[i]*mul[i]);
        
        /* update optimal number of samples (formula (4) in lecture notes)
         Wl  = beta.^(gamma*(0:L));
         Ns  = ceil(sqrt(Vl./Wl) * sum(sqrt(Vl.*Wl)) / (theta*eps^2/C_alpha));
         dNl = max(0, Ns-Nl);
         */
        for (i=0; i<L; i++) Wl[i] = PetscPowReal(mlmc->beta,mlmc->gamma*i);
        sumVlWl = 0.0; for (i=0; i<L; i++) sumVlWl += PetscSqrtReal(Vl[i]*Wl[i]);
        for (i=0; i<L; i++) Ns[i]   = (PetscInt)PetscCeilReal(PetscSqrtReal(Vl[i]/Wl[i])*sumVlWl/(mlmc->theta*eps*eps/mlmc->C_alpha));
        for (i=0; i<L; i++) dNl[i]  = PetscMax(0,Ns[i]-mlmc->Nl[i]);
        
        /*  if (almost) converged, estimate remaining error and decide
         whether a new level is required
         
         if sum( dNl > 0.01*Nl ) == 0
         range = -2:0;
         rem = max(mul(L+1+range).*beta.^(q1*range))/(beta^q1 - 1); %formula (5)
         if rem > (1-theta)*eps
         L=L+1;
         Vl(L+1) = Vl(L) / beta^q2;   %formula (7) in lecture notes
         Nl(L+1) = 0;
         suml(1:2,L+1) = 0;
         
         Wl  = beta.^(gamma*(0:L));
         Ns  = ceil(sqrt(Vl./Wl) * sum(sqrt(Vl.*Wl)) / (theta*eps^2/C_alpha));
         dNl = max(0, Ns-Nl);
         end
         end
         */
        sumdNl = 0.0; for (i=0; i<L; i++) sumdNl += (dNl[i] > .01*mlmc->Nl[i]);
        if (!sumdNl) {
            PetscReal rem = 0.0;
            for (i=-2; i<1; i++) {
                rem = PetscMax(rem,mul[L-1+i]*PetscPowReal(mlmc->beta,mlmc->q1*i)/(PetscPowReal(mlmc->beta,mlmc->q1) - 1.0));
            }
            if (rem > (1.0 - mlmc->theta)*eps) {
                PetscInt i;
                ierr = PetscInfo(NULL,"Adding another MLMC level to the hierarchy\n");CHKERRQ(ierr);
                L = L + 1;
                ierr = HeatSimulationRefine(hs[L-2],&hs[L-1]);CHKERRQ(ierr);
                Vl[L-1] = Vl[L-2]/PetscPowReal(mlmc->beta,mlmc->q2);
                for (i=0; i<L; i++) Wl[i] = PetscPowReal(mlmc->beta,mlmc->gamma*i);
                sumVlWl = 0.0; for (i=0; i<L; i++) sumVlWl += PetscSqrtReal(Vl[i]*Wl[i]);
                for (i=0; i<L; i++) Ns[i]  = (PetscInt)PetscCeilReal(PetscSqrtReal(Vl[i]/Wl[i])*sumVlWl/(mlmc->theta*eps*eps/mlmc->C_alpha));
                for (i=0; i<L; i++) dNl[i] = PetscMax(0,Ns[i]-mlmc->Nl[i]);
            }
        }
        sumdNl = 0.0; for (i=0; i<L; i++) sumdNl += dNl[i];
    }
    
    /* finally, evaluate multilevel estimator
     EQ = sum(suml(1,:)./Nl);
     */
    *EQ = 0; for (i=0; i<L; i++) *EQ += suml[i][0]/mlmc->Nl[i];
    mlmc->L = L;
    ierr = PetscInfo2(NULL,"Completed MLMC algorithm QoI %g Number of levels %D\n",*EQ,L);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*TEST
 
 build:
 requires: !complex
 
 # When run with the Matlab random generator these results are identical
 # to those from the Matlab code if you add a call to rng('default') in
 # mlmc_heat_conv.m before each new call to mlmc()
 test:
 args: -eps .0014,.0012,.0010,.0008,.0006,.0004,.0002,.0001 -info
 filter: egrep "(MLMC|QoI)"
 
 TEST*/
