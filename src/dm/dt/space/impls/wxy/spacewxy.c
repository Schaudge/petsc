#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h>      /*I "petscdt.h" I*/
#include <petscblaslapack.h>

static PetscErrorCode PetscSpaceWXYView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscInt          N, Nc;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(sp, &N);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "Wheeler-Xue-Yotov space in dimension %D:\n", Nc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    /* Print anything private */
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Wheeler-Xue-Yotov space in dimension %D\n", Nc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_WXY(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpaceWXYView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_WXY(PetscSpace sp)
{
  PetscSpace_WXY *wxy = (PetscSpace_WXY *) sp->data;
  PetscReal      *C, *sigma, *tmpN_C, *work;
  PetscBLASInt    N, m, n = 4, one = 1, lwork, lierr, i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(wxy->N,  &N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(wxy->Nu, &m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(PetscMax(m, n) + 5*PetscMin(m, n), &lwork);CHKERRQ(ierr);
  if (sp->Nv != sp->Nc) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_INCOMP, "The Wheeler-Xue-Yotov space is a vector space so the number of variables %D should equal the number of components %D", sp->Nv, sp->Nc);
  ierr = PetscCalloc4(m*n, &C, 4, &sigma, m*m, &tmpN_C, lwork, &work);CHKERRQ(ierr);
  /* TODO Create constraint matrix or just type in the one from Nathan */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgesvd", LAPACKgesvd_("A", "N", &m, &n, C, &m, sigma, tmpN_C, &m, NULL, &one, work, &lwork, &lierr));
  if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "gesv() error %d", lierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  for (i = 0; i < m; ++i) {
    for (j = 0; j < N; ++j) {
      wxy->N_C[i*N + j] = tmpN_C[(j+n)*m + i];
    }
  }
  ierr = PetscFree4(C, sigma, tmpN_C, work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_WXY(PetscSpace sp)
{
  PetscSpace_WXY *wxy = (PetscSpace_WXY *) sp->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(wxy->N_C);CHKERRQ(ierr);
  ierr = PetscFree(wxy->tmpB);CHKERRQ(ierr);
  ierr = PetscFree(wxy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_WXY(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_WXY *wxy = (PetscSpace_WXY *) sp->data;

  PetscFunctionBegin;
  *dim = wxy->N;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceWXYEvaluateUnconstrained(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[])
{
  PetscSpace_WXY *wxy = (PetscSpace_WXY *) sp->data;
  PetscInt        N = wxy->Nu, Nc = 3, p, i;

  PetscFunctionBegin;
  for (p = 0; p < npoints; ++p) {
    const PetscReal *x = &points[p*Nc];
    PetscReal       *P = &B[p*N*Nc];

    i = 0;
    P[i] =    1; P[i+1] =    0; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] =    1; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] =    0; P[i+2] =    1; i+=Nc;
    P[i] = x[0]; P[i+1] =    0; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] = x[0]; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] =    0; P[i+2] = x[0]; i+=Nc;
    P[i] = x[1]; P[i+1] =    0; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] = x[1]; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] =    0; P[i+2] = x[1]; i+=Nc;
    P[i] = x[2]; P[i+1] =    0; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] = x[2]; P[i+2] =    0; i+=Nc;
    P[i] =    0; P[i+1] =    0; P[i+2] = x[2]; i+=Nc;
    P[i] =                 0; P[i+1] =         x[0]*x[1]; P[i+2] =        -x[0]*x[2]; i+=Nc; /* curl(x*y*z   ,0       ,0       ) */
    P[i] =        -x[0]*x[1]; P[i+1] =                 0; P[i+2] =         x[1]*x[2]; i+=Nc; /* curl(0       ,x*y*z   ,0       ) */
    P[i] =         x[0]*x[2]; P[i+1] =        -x[1]*x[2]; P[i+2] =                 0; i+=Nc; /* curl(0       ,0       ,x*y*z   ) */
    P[i] =                 0; P[i+1] =                 0; P[i+2] =      -2*x[0]*x[1]; i+=Nc; /* curl(x*y**2  ,0       ,0       ) */
  //P[i] =       2*x[0]*x[1]; P[i+1] =        -x[1]*x[1]; P[i+2] =                 0; i+=Nc; /* curl(0       ,0       ,x*y**2  ) */ /*REMOVE*/
    P[i] =        -x[0]*x[0]; P[i+1] =                 0; P[i+2] =       2*x[0]*x[2]; i+=Nc; /* curl(0       ,x**2*z  ,0       ) */
    P[i] =                 0; P[i+1] =      -2*x[0]*x[2]; P[i+2] =                 0; i+=Nc; /* curl(0       ,0       ,x**2*z  ) */
    P[i] =                 0; P[i+1] =       2*x[1]*x[2]; P[i+2] =        -x[2]*x[2]; i+=Nc; /* curl(y*z**2  ,0       ,0       ) */
    P[i] =      -2*x[1]*x[2]; P[i+1] =                 0; P[i+2] =                 0; i+=Nc; /* curl(0       ,y*z**2  ,0       ) */
    P[i] =                 0; P[i+1] =    x[0]*x[1]*x[1]; P[i+2] = -2*x[0]*x[1]*x[2]; i+=Nc; /* curl(x*y**2*z,0       ,0       ) */
    P[i] = -2*x[0]*x[1]*x[2]; P[i+1] =                 0; P[i+2] =    x[1]*x[2]*x[2]; i+=Nc; /* curl(0       ,x*y*z**2,0       ) */
  //P[i] =    x[0]*x[0]*x[2]; P[i+1] = -2*x[0]*x[1]*x[2]; P[i+2] =                 0; i+=Nc; /* curl(0       ,0       ,x**2*y*z) */ /*REMOVE*/
  }
  PetscFunctionReturn(0);
}

/*
  This basis was constructed by (1) starting with the hexahedral basis
  found in (Wheeler,Xue,Yotov 2012), but omitting the addition of
  curl(0,0,x*y^2) and curl(0,0,x^2*y*z) to the BDDF1 space, (2)
  creating a constraint matrix to remove x*y from the top and bottom
  faces of the prism and higher order terms from the diagonal face (3)
  taking the SVD and using the eigenvectors to constrain the original
  space.
  p in [0, npoints), i in [0, pdim), c in [0, Nc)
  B[p][i][c] = B[p][i_scalar][c][c]
*/
static PetscErrorCode PetscSpaceEvaluate_Nathan(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscInt       p, dim = sp->Nv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (H) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Hessians not supported for Hdiv spaces on prisms");
  if (dim != 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Prisms only defined for spatial dimension = 3");
  if (sp->Nc != 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Hdiv prism space only defined for number of components = 3");
  for (p = 0; p < npoints; p++) {
    const PetscReal x = points[p*dim];
    const PetscReal y = points[p*dim+1];
    const PetscReal z = points[p*dim+2];

    if (B) {
      ierr = PetscArrayzero(B, npoints*18*3);CHKERRQ(ierr);
      B[p*54+ 1] =  1;
      B[p*54+ 4] =  x;
      B[p*54+ 7] =  y;
      B[p*54+10] =  z;
      B[p*54+14] =  1;
      B[p*54+17] =  x;
      B[p*54+20] =  y;
      B[p*54+23] =  z;
      B[p*54+24] =  x*z;
      B[p*54+25] = -y*z;
      B[p*54+28] = -2*x*z;
      B[p*54+30] =  0.333333333333333*x*x - 0.333333333333333*x*y + 0.577350269189626*y;
      B[p*54+31] =  0.666666666666667*x*y;
      B[p*54+32] = -1.333333333333333*x*z + 0.333333333333333*y*z;
      B[p*54+34] =  2*y*z;
      B[p*54+35] = -z*z;
      B[p*54+36] = -0.707106781186547*x + 0.707106781186547;
      B[p*54+39] =  0.707106781186547*x + 0.707106781186547;
      B[p*54+42] = -0.333333333333333*x*x - 0.666666666666667*x*y - 0.577350269189626*y;
      B[p*54+43] =  0.333333333333333*x*y;
      B[p*54+44] =  0.333333333333333*x*z + 0.666666666666667*y*z;
      B[p*54+45] = -0.666666666666667*x*x - 0.333333333333333*x*y + 0.577350269189626*y;
      B[p*54+46] = -0.333333333333333*x*y;
      B[p*54+47] =  1.666666666666667*x*z + 0.333333333333333*y*z;
      B[p*54+48] = -2*y*z;
      B[p*54+51] = -z;
    }
    if (D) {
      ierr = PetscArrayzero(D, npoints*18*3*3);CHKERRQ(ierr);
      /* D[p][0][c][d] = 0 */
      D[((p*18 + 1)*3 + 1)*3 + 0] =  1;
      D[((p*18 + 2)*3 + 1)*3 + 1] =  1;
      D[((p*18 + 3)*3 + 1)*3 + 2] =  1;
      /* D[p][4][c][d] = 0 */
      D[((p*18 + 5)*3 + 2)*3 + 0] =  1;
      D[((p*18 + 6)*3 + 2)*3 + 1] =  1;
      D[((p*18 + 7)*3 + 2)*3 + 2] =  1;
      D[((p*18 + 8)*3 + 0)*3 + 0] =  z;
      D[((p*18 + 8)*3 + 0)*3 + 2] =  x;
      D[((p*18 + 8)*3 + 1)*3 + 1] = -z;
      D[((p*18 + 8)*3 + 1)*3 + 2] = -y;
      D[((p*18 + 9)*3 + 1)*3 + 0] = -2*z;
      D[((p*18 + 9)*3 + 1)*3 + 2] = -2*x;
      D[((p*18 +10)*3 + 0)*3 + 0] =  0.666666666666667*x - 0.333333333333333*y;
      D[((p*18 +10)*3 + 0)*3 + 1] = -0.333333333333333*x + 0.577350269189626;
      D[((p*18 +10)*3 + 1)*3 + 0] =  0.666666666666667*y;
      D[((p*18 +10)*3 + 1)*3 + 1] =  0.666666666666667*x;
      D[((p*18 +10)*3 + 2)*3 + 0] = -1.333333333333333*z;
      D[((p*18 +10)*3 + 2)*3 + 1] =  0.333333333333333*z;
      D[((p*18 +10)*3 + 2)*3 + 2] = -1.333333333333333*x + 0.333333333333333*y;
      D[((p*18 +11)*3 + 1)*3 + 1] =  2*z;
      D[((p*18 +11)*3 + 1)*3 + 2] =  2*y;
      D[((p*18 +11)*3 + 2)*3 + 2] = -2*z;
      D[((p*18 +12)*3 + 0)*3 + 0] = -0.707106781186547;
      D[((p*18 +13)*3 + 0)*3 + 0] =  0.707106781186547;
      D[((p*18 +14)*3 + 0)*3 + 0] = -0.666666666666667*x - 0.666666666666667*y;
      D[((p*18 +14)*3 + 0)*3 + 1] = -0.666666666666667*x - 0.577350269189626;
      D[((p*18 +14)*3 + 1)*3 + 0] =  0.333333333333333*y;
      D[((p*18 +14)*3 + 1)*3 + 1] =  0.333333333333333*x;
      D[((p*18 +14)*3 + 2)*3 + 0] =  0.333333333333333*z;
      D[((p*18 +14)*3 + 2)*3 + 1] =  0.666666666666667*z;
      D[((p*18 +14)*3 + 2)*3 + 2] =  0.333333333333333*x + 0.666666666666667*y;
      D[((p*18 +15)*3 + 0)*3 + 0] = -1.333333333333333*x - 0.333333333333333*y;
      D[((p*18 +15)*3 + 0)*3 + 1] = -0.333333333333333*x + 0.577350269189626;
      D[((p*18 +15)*3 + 1)*3 + 0] = -0.333333333333333*y;
      D[((p*18 +15)*3 + 1)*3 + 1] = -0.333333333333333*x;
      D[((p*18 +15)*3 + 2)*3 + 0] =  1.666666666666667*z;
      D[((p*18 +15)*3 + 2)*3 + 1] =  0.333333333333333*z;
      D[((p*18 +15)*3 + 2)*3 + 2] =  1.666666666666667*x + 0.333333333333333*y;
      D[((p*18 +16)*3 + 0)*3 + 1] = -2*z;
      D[((p*18 +16)*3 + 0)*3 + 2] = -2*y;
      D[((p*18 +17)*3 + 0)*3 + 2] = -1;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_WXY(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_WXY *wxy = (PetscSpace_WXY *) sp->data;
  PetscReal      *N_C = wxy->N_C;
  PetscInt        Nc = sp->Nc, pdim = wxy->N, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (B) {ierr = PetscArrayzero(B, npoints*pdim*Nc);CHKERRQ(ierr);}
  if (D) {ierr = PetscArrayzero(D, npoints*pdim*Nc*Nc);CHKERRQ(ierr);}
  if (H) {ierr = PetscArrayzero(H, npoints*pdim*Nc*Nc*Nc);CHKERRQ(ierr);}
#if 1
  ierr = PetscSpaceEvaluate_Nathan(sp, npoints, points, B, D, H);CHKERRQ(ierr);
#else
  for (p = 0; p < npoints; ++p) {
    const PetscReal *x  = &points[p*Nc];
    PetscReal       *P  = &B[p*pdim*Nc];
    PetscReal       one = 1, zero = 0;
    PetscBLASInt    m, k, n, lda, ldb, ldc;

    ierr = PetscBLASIntCast(wxy->N,  &m);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(wxy->Nu, &k);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(Nc,      &n);CHKERRQ(ierr);
    lda = k, ldb = n, ldc = m;
    ierr = PetscSpaceWXYEvaluateUnconstrained(sp, 1, x, wxy->tmpB);CHKERRQ(ierr);
    CHKMEMQ;
    PetscStackCallBLAS("BLASgemm", BLASREALgemm_("T", "T", &m, &n, &k, &one, N_C, &lda, wxy->tmpB, &ldb, &zero, P, &ldc));
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_WXY(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = NULL;
  sp->ops->setup          = PetscSpaceSetUp_WXY;
  sp->ops->view           = PetscSpaceView_WXY;
  sp->ops->destroy        = PetscSpaceDestroy_WXY;
  sp->ops->getdimension   = PetscSpaceGetDimension_WXY;
  sp->ops->evaluate       = PetscSpaceEvaluate_WXY;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEWXY = "wxy" - A PetscSpace object that encapsulates the Wheeler-Xue-Yotov basis, intended for prisms.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_WXY(PetscSpace sp)
{
  PetscSpace_WXY *wxy;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp, &wxy);CHKERRQ(ierr);
  sp->data = wxy;

  sp->Nv = 3;
  sp->Nc = 3;
  sp->maxDegree = PETSC_MAX_INT;
  wxy->Nu  = 22;
  wxy->N   = 18;
  wxy->N_C = NULL;
  ierr = PetscMalloc1(wxy->Nu * sp->Nc, &wxy->tmpB);CHKERRQ(ierr);
  ierr = PetscMalloc1(wxy->N * wxy->Nu, &wxy->N_C);CHKERRQ(ierr);

  ierr = PetscSpaceInitialize_WXY(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
