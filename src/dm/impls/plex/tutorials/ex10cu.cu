static char help[] =
 "Demonstrates how data from DMPlex may be mapped to perform computations on GPUs.\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include <petscdmadaptor.h>
#include <../src/mat/impls/aij/seq/aij.h>

/* Until we have a good implementation of InsertMatrixElements for the Mat_seqAIJCUSparse we will use this stand-in struct so that we can do assembly
 * on the GPU. We re-invent some wheels here, but in stripping things down we hopefully get a better picture of the moving pieces. This may help to
 * give us a sense for the essential functions that need to be added to the libraries. */

 typedef struct
{
  PetscInt nnz; /* number of non-zeros */
  PetscInt m; /* number of rows */
  PetscInt n; /* number of colmuns */
  PetscReal* vals; /* array containing matrix entries */
  PetscInt* rowPtr; /* pointers to first element of each row */
  PetscInt* colInd; /* column indices for each entry */
  PetscBool setupCalled;
} SimpleCSRMat;

static PetscErrorCode SimpleCSRMatDestroy(SimpleCSRMat * mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Free mat as long as it exists */
  if (mat) {
    if (mat->colInd) {
      ierr = PetscFree(mat->colInd);CHKERRQ(ierr);
    }
    if (mat->rowPtr) {
      ierr = PetscFree(mat->rowPtr);CHKERRQ(ierr);
    }
    if (mat->vals) {
      ierr = PetscFree(mat->vals);CHKERRQ(ierr);
    }
    ierr = PetscFree(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Creates a mat pointer with garbage data values. */
static PetscErrorCode SimpleCSRMatCreate(SimpleCSRMat ** mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*mat) {
    ierr = SimpleCSRMatDestroy(*mat);CHKERRQ(ierr);
  }
  ierr             = PetscMalloc1(1,mat);CHKERRQ(ierr);
  (*mat)->setupCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* Allocates the member arrays. Should only be called after nnz,m,and n are set.*/
static PetscErrorCode SimpleCSRMatSetup(SimpleCSRMat * mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(mat->setupCalled)) {
    ierr             = PetscCalloc1(mat->nnz,&mat->vals);CHKERRQ(ierr);
    ierr             = PetscCalloc2(mat->m,&mat->rowPtr,mat->nnz,&mat->colInd);CHKERRQ(ierr);
    mat->setupCalled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SimpleCSRMatCopyNonZeroStructure(SimpleCSRMat *mat,Mat source)
{
  PetscInt i;
  PetscErrorCode ierr;
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)source->data;

  PetscFunctionBegin;
  ierr = SimpleCSRMatCreate(&mat);
  mat->nnz = a->nz;
  ierr = MatGetSize(source,&mat->m,&mat->n);CHKERRQ(ierr);
  ierr = SimpleCSRMatSetup(mat);CHKERRQ(ierr);

  for(i=0; i<mat->m; ++i){
    mat->rowPtr[i] = a->i[i];
  }
  for(i=0; i<mat->nnz; ++i){
    mat->colInd[i] = a->j[i];
  }
  PetscFunctionReturn(0);
}

/* Returns the index into value array for Ith row Jth Column */
static PetscErrorCode SimpleCSRMatGetIJ(SimpleCSRMat mat,PetscInt i, PetscInt j, PetscInt* ind)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
/*
  Construct a 4-tensor for the Poisson operator on the reference element of a finite element.

  The element matrix of the Poisson operator on an element E in physical space is a matrix

  A_{i,j} = \int_E \frac{\partial}{\partial x_k} \phi_i \frac{\partial}{\partial x_k} \phi_j dx

  (This is Einstein notation).

  To compute this quantity on a reference element \hat E, we have

  A_{i,j} = \int_E (\frac{\partial \hat x_l}{\partial x_k} \frac{\partial}{\partial \hat x_l}) \phi_i (\frac{\partial \hat x_m}{\partial x_k} \frac{\partial}{\partial x_k}) \phi_j dx
          = \int_{\hat E} (\frac{\partial \hat x_l}{\partial x_k} \frac{\partial}{\partial \hat x_l}) \hat \phi_i (\frac{\partial \hat x_m}{\partial x_k} \frac{\partial}{\partial \hat x_m}) \hat \phi_j |J_E| dx,

  Where |J_E| is the determinant of the element map \hat E -> E.  If the map from \hat E to E is affine, we can write this as

  A_{i,j} = (\int_{\hat E} \frac{\partial}{\partial \hat x_l} \hat \phi_i \frac{\partial}{\partial \hat x_m} \hat \phi_j |J_E| dx) (\frac{\partial \hat x_l}{\partial x_k} \frac{\partial \hat x_m}{\partial x_k} |J_E}),

  which expresses the real element matrix as the contraction of a reference 4-tensor with indices ijlm and a matrix with indices lm.

  The 4-tensor is constructed here with the indices in the order lmij (l slowest, j fastest).  This 4-tensor can then be moved to an accelerator to
  contract with the geometric factors for the individual elements to compute all of the element matrices.

  tensor should have size (dim * dim * ndof * ndof)
*/
static PetscErrorCode PoissonReferenceTensor(PetscFE fe, PetscScalar tensor[])
{
  PetscTabulation tabulation;
  const PetscReal *D;
  PetscQuadrature quad;
  PetscInt nq, dim, ndof;
  const PetscReal *w;
  PetscInt idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fe, &ndof);CHKERRQ(ierr);
  ierr = PetscFEGetCellTabulation(fe, &tabulation);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &nq, NULL, &w);CHKERRQ(ierr);
  D = tabulation->T[1];
  ierr = PetscArrayzero(tensor, dim * dim * ndof * ndof);CHKERRQ(ierr);
  idx = 0;
  for (PetscInt l = 0; l < dim; l++) {
    for (PetscInt m = 0; m < dim; m++) {
      for (PetscInt i = 0; i < ndof; i++) {
        for (PetscInt j = 0; j < ndof; j++, idx++) {
          PetscReal val = 0.;
          for (PetscInt q = 0; q < nq; q++) val += D[(q*ndof + i)*dim + l] * D[(q*ndof + j)*dim + m] * w[q];
          tensor[idx] += val;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PoissonReferenceToReal(PetscInt dim, PetscInt ndof, const PetscScalar tensor[], const PetscReal Jinv[], PetscReal Jdet, PetscScalar elemMat[])
{
  PetscReal      JJ[9];
  PetscInt       idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscArrayzero(JJ, dim * dim);CHKERRQ(ierr);
  ierr = PetscArrayzero(elemMat, ndof * ndof);CHKERRQ(ierr);
  for (PetscInt k = 0; k < dim; k++) {
    for (PetscInt l = 0; l < dim; l++) {
      for (PetscInt m = 0; m < dim; m++) {
        JJ[l * dim + m] += Jinv[l * dim + k] * Jinv[m * dim + k] * Jdet;
      }
    }
  }

  idx = 0;
  for (PetscInt l = 0; l < dim; l++) {
    for (PetscInt m = 0; m < dim; m++) {
      for (PetscInt i = 0; i < ndof; i++) {
        for (PetscInt j = 0; j < ndof; j++, idx++) {
          elemMat[i * ndof + j] += JJ[l * dim + m] * tensor[idx];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* We are solving the one field formulation of the poisson equation
 * -\laplacian{u} = f
 * When we convert this to a weak form we get the integral equation
 * -\int_Omega \laplacian{u}v dV = \int_Omega fv dV
 *  Applying Stokes theorem on the left hand side let's us shift a derivative and gives
 *  \int_Omega \grad{u} \cdot \grad{v} dV = \int_Omega fv dV
 *  which we can use to construct the residual and jacobian functions below.
 */

/* Exact solutions for a linear RHS 
 * Where f = x + y + z, then u = -1/6*(x^3+y^3+z^3);
   */
static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt d;

  u[0] = 0;
  for (d=0; d<dim; ++d) u[0] -= PetscPowReal(x[d],3);
  u[0] /= 6;

  return 0;
}

static PetscErrorCode linear_rhs(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{ 
  PetscInt d;
  
  u[0] = 0;
  for (d = 0; d < dim; ++d) u[0] += x[d];
  return 0;
}

/* f?_v are the residual functions.
 * f0_v takes care of the term <f,v>
 * */
static void f0_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar f;

  (void)linear_rhs(dim,t,x,1,&f,NULL);
  f0[0] = -f;
}

/* f1_v takes care of the term <\grad{u}, \grad{v}> */
static void f1_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) {
    f1[c] = u_x[c];
  }
}

/* gx_yz are the jacobian functions obtained by taking the derivative of the y residual w.r.t z*/
static void g3_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g3[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g3[c*dim + c] = 1.0;
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

  PetscFunctionBegin;
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  /* All of these are independent of the user's choice of solution */
  ierr = PetscDSSetResidual(prob,0,f0_v,f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,NULL,NULL,NULL,g3_vu);CHKERRQ(ierr);

  ierr = PetscDSSetExactSolution(prob,0,linear_u,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create the finite element spaces we will use for this system */
static PetscErrorCode SetupDiscretization(DM mesh,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx *user)
{
  DM             cdm = mesh;
  PetscFE        u;
  const PetscInt dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create FE object and give them names so that options can be set*/
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,1,user->simplex,"u_",-1,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);

  /* Associate the FE objects with the mesh and setup the system */
  ierr = DMSetField(mesh,0,NULL,(PetscObject)u);CHKERRQ(ierr);
  ierr = DMCreateDS(mesh);CHKERRQ(ierr);
  ierr = (*setup)(mesh,user);CHKERRQ(ierr);

  while (cdm) {
    ierr = DMCopyDisc(mesh,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm,&cdm);CHKERRQ(ierr);
  }

  /* The Mesh now owns the fields, so we can destroy the FEs created in this
   * function */
  ierr = PetscFEDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GetElementClosureMaps(DM dm, PetscInt pStart, PetscInt pEnd, PetscInt** closureSizes, PetscInt*** closures){
  PetscInt p,i,closureSize,*closure;
  PetscErrorCode ierr;
  PetscSection lSec,gSec;

  PetscFunctionBegin;
  
  ierr = DMGetLocalSection(dm,&lSec);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm,&gSec);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd-pStart,closureSizes);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd-pStart,closures);CHKERRQ(ierr);

  /* Here is another loop that we could convert to threadwise GPU style, but this would depend on whether or not section information
   * can be copied to GPUs */
  for (p=pStart; p<pEnd; ++p){
  ierr = DMPlexGetClosureIndices(dm,lSec,gSec,p,PETSC_TRUE,&closureSize,&closure,NULL,NULL);CHKERRQ(ierr);
  (*closureSizes)[p-pStart] = closureSize;
  ierr = PetscCalloc1(closureSize,&((*closures)[p-pStart]));
    for (i=0; i<closureSize; ++i){
      (*closures)[p-pStart][i] = closure[i];
    }
  ierr = DMPlexRestoreClosureIndices(dm,lSec,gSec,p,PETSC_TRUE,&closureSize,&closure,NULL,NULL);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode BuildSystemMatrix(DM dm, Mat* systemMat){
  /* Declare basis function coefficients on the reference element. Assuming 2d uniform quadrilaterals. */
  PetscInt cStart,cEnd,c;
  PetscFE field;
  /* Taking advantage of the fact that we are using uniform quads for now. In the future we will need to determine sizes of tensor,
   * elemMat, and values for Jinv and Jdet on a per element basis. */
  PetscScalar *tensor,*elemMat;
  PetscReal Jdet,*Jinv,*J,*v;
  PetscInt *closureSizes,**closures,nDoF,dim;
  SimpleCSRMat *testCSR = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetField(dm,0,NULL,(PetscObject*)&field);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(field,&dim);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(field,&nDoF);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscCalloc3(dim,&v,dim*dim,&J,dim*dim,&Jinv);CHKERRQ(ierr);
  ierr = PetscCalloc2(nDoF*nDoF*dim*dim,&tensor,nDoF*nDoF,&elemMat);CHKERRQ(ierr);
  ierr = PoissonReferenceTensor(field,tensor);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,systemMat);CHKERRQ(ierr);
  ierr = SimpleCSRMatCopyNonZeroStructure(testCSR,*systemMat);CHKERRQ(ierr);
  ierr = MatZeroEntries(*systemMat);CHKERRQ(ierr);
  ierr = MatSetOption(*systemMat,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = GetElementClosureMaps(dm,cStart,cEnd,&closureSizes,&closures);CHKERRQ(ierr);

  /* This a loop that we will convert to threadwise GPU style */
  for (c=cStart; c<cEnd; ++c){
  ierr = DMPlexComputeCellGeometryAffineFEM(dm,c,v,J,Jinv,&Jdet);CHKERRQ(ierr);
  ierr = PoissonReferenceToReal(dim,nDoF,tensor,Jinv,Jdet,elemMat);CHKERRQ(ierr);
    ierr = MatSetValues(*systemMat,closureSizes[c-cStart],closures[c-cStart],closureSizes[c-cStart],closures[c-cStart],elemMat,ADD_VALUES);CHKERRQ(ierr);
  }
  MatAssemblyBegin(*systemMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatAssemblyEnd(*systemMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFree(closureSizes);
  for (c=cStart; c<cEnd; ++c){
    PetscFree(closures[c]);
  }
  PetscFree(closures);
  PetscFree(J);
  PetscFree(Jinv);
  PetscFree(v);
  PetscFree(tensor);
  PetscFree(elemMat);

  PetscFunctionReturn(0);
}


int main(int argc,char **argv)
{
  UserCtx         user;
  DM              dm;
  SNES            snes;
  Vec             u,v,w,z;
  PetscReal       norm;
  Mat             sysMat,JMat;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);

  /* Set up a snes */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm,SetupProblem,&user);CHKERRQ(ierr);

  /* Create system matrix on CPU using SNES */
  ierr = DMCreateGlobalVector(dm,&u);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"solution_snes");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes,u,NULL,NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u,NULL,"-solution_snes_view");CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&JMat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatViewFromOptions(JMat,NULL,"-JMat_view");CHKERRQ(ierr);
 
  /* Create system matrix using routines that we will port to GPU */ 
  ierr = BuildSystemMatrix(dm,&sysMat);CHKERRQ(ierr);
  ierr = MatViewFromOptions(sysMat,NULL,"-sysMat_view");CHKERRQ(ierr);

  /* Check that we get the same matrices from both methods using the roundabout method of checking whether the difference between vector multiplies
   * has a norm less than some small number. Maybe eventually we implement a method for C = \alpha*A + B, and check that the difference matrix has
   * entries all smaller than tolerance.*/
  ierr = DMCreateGlobalVector(dm,&v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&w);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&z);CHKERRQ(ierr);
  ierr = VecSetRandom(u,NULL);CHKERRQ(ierr);
  ierr = VecCopy(u,v);CHKERRQ(ierr);
  ierr = MatMult(JMat,u,w);CHKERRQ(ierr);
  ierr = MatMult(sysMat,v,z);CHKERRQ(ierr);
  ierr = VecWAXPY(u,-1.0,w,z);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"Matrix assembly successful: %s\n",norm<=PETSC_SMALL?"true":"false");

  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = MatDestroy(&sysMat);CHKERRQ(ierr);
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
      -u_petscspace_degree 1 \
      -u_petscspace_type poly \
      -u_petscdualspace_type lagrange \
      -dm_refine 0 \
      -ksp_rtol 1e-10 \
      -pc_use_amat true \
TEST*/
