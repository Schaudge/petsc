static char help[33] = "Test Unstructured Mesh Handling\n";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscsnes.h>
# include <petscds.h>
# include <petscksp.h>

# define PETSCVIEWERVTK          "vtk"
# define PETSCVIEWERASCII        "ascii"
# define VECSTANDARD    	 "standard"

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;

typedef struct {
  PetscLogStage  stageREAD, stageCREATE, stageREFINE, stageINSERT, stageADD, stageGVD;
  PetscLogEvent  eventREAD, eventCREATE, eventREFINE, eventINSERT, eventADD, eventGVD;
  PetscBool      simplex, perfTest, fileflg, distribute, interpolate, dmRefine, dispFlag, isView, VTKdisp, sectionDisp, arrayDisp, coordDisp, usePetscFE, useKSP, vtkSoln;
  /* Domain and mesh definition */
  PetscInt       dim, meshSize, numFields, overlap, qorder, level, commax;
  PetscScalar	 refinementLimit;
  char           filename[2048];    /* The optional mesh file */
  char 		 bar[19];
  /* Problem definition */
  BCType	 bcType;
  DMBoundaryType periodicity[3];
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);
  PetscBool      fieldBC;
  void           (**exactFields)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
} AppCtx;

static PetscErrorCode SetupProblem(DM, AppCtx*);

/* ADDITIONAL FUNCTIONS	*/
PetscErrorCode ViewISInfo(MPI_Comm comm, DM dm)
{
  PetscViewer	viewer;
  DMLabel		label;
  IS 		labelIS;
  const char 	*labelName;
  PetscInt 	numLabels, l;
  PetscErrorCode 	ierr;
  char            tbar[10] = "----------";

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
  /*	query the number and name of labels	*/
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of labels: %d\n", numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l)
  {
    ierr = DMGetLabelName(dm, l, &labelName);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Label %d: name: %s\n", l, labelName);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "IS of values\n");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = ISView(labelIS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
  }
  /*	Making sure that string literals work	*/
  ierr = PetscPrintf(comm,"\n\nCell Set label IS\n");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Cell Sets", &label);CHKERRQ(ierr);
  /*	Specifically look for Cell Sets as these seem to be vertices	*/
  if (label)
  {
    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = ISView(labelIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"%s End Label View %s\n", tbar, tbar);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode GeneralInfo(MPI_Comm comm, AppCtx user, PetscViewer genViewer)
{
  PetscErrorCode	ierr;
  const char 		*string;

  ierr = PetscViewerStringSPrintf(genViewer, "Dimension of mesh:%s>%d\n", user.bar + 3, user.dim);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Number of Fields:%s>%d", user.bar + 2, user.numFields);CHKERRQ(ierr);
  if (user.numFields == 100) {
    ierr = PetscViewerStringSPrintf(genViewer, "(default)");CHKERRQ(ierr);
  } else if (user.numFields == 1) {
    ierr = PetscViewerStringSPrintf(genViewer, "(default PetscFE)");CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Ghost point overlap:%s>%d\n", user.bar + 5, user.overlap);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "\nFile read mode:%s>%s\n", user.bar, user.fileflg ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.fileflg) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ File read name:%s>%s\n", user.bar + 2, user.filename);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "Mesh refinement:%s>%s\n", user.bar + 1, user.dmRefine ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.dmRefine) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ Refinement level:%s>%d\n", user.bar + 4, user.level);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "Distributed dm:%s>%s\n", user.bar, user.distribute ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Interpolated dm:%s>%s\n", user.bar + 1, user.interpolate ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Performance test mode:%s>%s\n", user.bar + 7, user.perfTest ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "PETScFE enabled mode:%s>%s\n", user.bar + 6, user.usePetscFE ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.usePetscFE) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ Quadrature order:%s>%d\n", user.bar + 4 , user.qorder);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "Solver Package Used:%s>%s\n", user.bar + 5, user.useKSP ? "KSP" : "SNES");CHKERRQ(ierr);


  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);

  ierr = PetscViewerStringSPrintf(genViewer, "VTKoutput mode:%s>%s\n", user.bar, user.VTKdisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Full Display mode:%s>%s\n", user.bar + 3, user.dispFlag ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "IS Display mode:%s>%s\n", user.bar + 1, user.isView ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Section Display mode:%s>%s\n", user.bar + 6, user.sectionDisp? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Array Display mode:%s>%s\n", user.bar + 4, user.arrayDisp ? "PETSC_TRUE * " : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Coord Disp mode:%s>%s\n", user.bar + 1, user.coordDisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "%s General Info %s\n", user.bar + 2, user.bar + 2);CHKERRQ(ierr);
  ierr = PetscViewerStringGetStringRead(genViewer, &string, NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, string);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s End General Info %s\n", user.bar + 2, user.bar + 5);CHKERRQ(ierr);

  return ierr;
}

/* KSP */
static PetscErrorCode ComputeLaplacianOperator(Mat *Op, void *ctx)
{
  AppCtx         	*user = (AppCtx*)ctx;
  PetscErrorCode	ierr;
  PetscInt		m = user->meshSize, n = user->meshSize, Istart, Iend, J, i, j, Ii;
  PetscScalar		v;

  PetscFunctionBeginUser;

  ierr = MatCreate(PETSC_COMM_WORLD, Op);CHKERRQ(ierr);
  ierr = MatSetSizes(*Op, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*Op);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*Op, 5, NULL, 5, NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*Op, 5, NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*Op);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(*Op, &Istart, &Iend);CHKERRQ(ierr);

  for (Ii = Istart; Ii < Iend; ++Ii) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i)   {J = Ii - n; ierr = MatSetValues(*Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i < m-1) {J = Ii + n; ierr = MatSetValues(*Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j)   {J = Ii - 1; ierr = MatSetValues(*Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(*Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(*Op, 1, &Ii, 1, &Ii, &v, INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(*Op, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Op, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(*Op, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeLaplacianOperatorWithKSP(KSP ksp, Mat jac, Mat Op, void *ctx)
{
  AppCtx         	*user = (AppCtx*)ctx;
  PetscErrorCode	ierr;
  PetscInt		m = user->meshSize, n = user->meshSize, Istart, Iend, J, i, j, Ii;
  PetscScalar		v;
  DM			dm, cdm;
  MatNullSpace		nullspace;

  PetscFunctionBeginUser;

  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &Op);CHKERRQ(ierr);
  MatView(Op, 0);
  ierr = MatSetUp(jac);CHKERRQ(ierr);
  ierr = MatSetUp(Op);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(Op, &Istart, &Iend);CHKERRQ(ierr);

  for (Ii = Istart; Ii < Iend; ++Ii) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i)   {J = Ii - n; ierr = MatSetValues(Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i < m-1) {J = Ii + n; ierr = MatSetValues(Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j)   {J = Ii - 1; ierr = MatSetValues(Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(Op,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(Op, 1, &Ii, 1, &Ii, &v, INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(Op, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Op, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(Op, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(Op, nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ComputeLaplacianJacobian(KSP ksp, Mat J, Mat jac, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             dm;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr  = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  if (!user->fileflg) {
    M = user->meshSize;
    N = user->meshSize;
    xs = 0;
    ys = 0;
    xm = xs + M;
    ym = ys + N;
  } else {
    PetscFunctionReturn(0);
  }
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  printf("%f %f %f %f %d %d\n", Hx, Hy, HxdHy, HydHx, M, N);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num=0; numi=0; numj=0;
        if (j!=0) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
          num++; numj++;
        }
        if (i!=0) {
          v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
          num++; numi++;
        }
        if (i!=M-1) {
          v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
          num++; numi++;
        }
        if (j!=N-1) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
          num++; numj++;
        }
        v[num] = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx);
        col[num].i = i;
        col[num].j = j;
        num++;
        ierr = MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatView(jac, 0);
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  AppCtx	*user = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,dim = user->dim, coordSize;
  PetscScalar    array[1];
  const PetscScalar *coordArray;
  DM		 dm;
  Vec		 coordVec;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  printf("STARTING!\n");
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm, &coordVec);CHKERRQ(ierr);
  ierr = VecGetSize(coordVec, &coordSize);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordVec, &coordArray);CHKERRQ(ierr);
  printf("SETTING ARRAY!\n");
  for (i = 0; i < coordSize/dim; ++i) {
    array[0] = -2.0*PetscCosScalar(coordArray[dim*i+1])*PetscSinScalar(coordArray[dim*i]);
    ierr = VecSetValues(b, 1, &i, array, INSERT_VALUES);CHKERRQ(ierr);
    /* Exact Solution laplace(sin(x)*cos(y)) = -2cos(y)*sin(x) */
  }
  ierr = VecRestoreArrayRead(coordVec, &coordArray);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AnalyticalSoln(DM dm, Vec *x, AppCtx *user)
{
  PetscErrorCode	ierr;
  Vec			coordVec;
  PetscScalar		array[1];
  const PetscScalar	*coordArray;
  PetscInt		i, dim = user->dim, coordSize;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dm, &coordVec);CHKERRQ(ierr);
  ierr = VecGetSize(coordVec, &coordSize);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordVec, &coordArray);CHKERRQ(ierr);
  printf("%d\n", coordSize/dim);
  for (i = 0; i < ((coordSize/dim)-1); ++i) {
    array[0] = PetscSinScalar(coordArray[dim*i])*PetscCosScalar(coordArray[dim*i+1]);
    ierr = VecSetValues(*x, 1, &i, array, INSERT_VALUES);CHKERRQ(ierr);
    /* Initial conditions u = sin(x)*cos(y) */
  }
  ierr = VecRestoreArrayRead(coordVec, &coordArray);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* SNES */
static void f0_xytrig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -2.0*PetscCosScalar(x[1])*PetscSinScalar(x[0]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    f1[d] = u_x[d]; /* 0 */
  }
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g3[d*dim+d] = 1.0;
  }
}

static PetscErrorCode AnalyticalSoln2D(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = PetscSinScalar(x[0])*PetscCosScalar(x[1]);
  return 0;
}

/* GENERAL PREPROCESSING */
static PetscErrorCode ProcessOpts(MPI_Comm comm, AppCtx *options)
{
  const char    	*bcTypes[3]  = {"neumann", "dirichlet", "none"};
  PetscErrorCode 	ierr;
  PetscInt		bd, bc;
  PetscBool		snesReq = PETSC_FALSE;

  PetscFunctionBeginUser;
  options->simplex		= PETSC_FALSE;
  options->perfTest             = PETSC_FALSE;
  options->fileflg		= PETSC_FALSE;
  options->distribute		= PETSC_FALSE;
  options->interpolate		= PETSC_TRUE;
  options->dmRefine		= PETSC_FALSE;
  options->dispFlag		= PETSC_FALSE;
  options->isView		= PETSC_FALSE;
  options->VTKdisp		= PETSC_FALSE;
  options->sectionDisp		= PETSC_FALSE;
  options->arrayDisp		= PETSC_FALSE;
  options->coordDisp		= PETSC_FALSE;
  options->usePetscFE		= PETSC_FALSE;
  options->useKSP		= PETSC_FALSE;
  options->vtkSoln		= PETSC_FALSE;
  options->periodicity[0]	= DM_BOUNDARY_GHOSTED;
  options->periodicity[1]	= DM_BOUNDARY_GHOSTED;
  options->periodicity[2]	= DM_BOUNDARY_GHOSTED;
  options->filename[0]		= '\0';
  options->bcType		= DIRICHLET;
  options->fieldBC		= PETSC_FALSE;
  options->meshSize		= 2;
  options->dim			= 2;
  options->numFields		= 1;
  options->overlap		= 0;
  options->qorder		= -1;
  options->level		= 0;
  options->refinementLimit      = 0.0;
  options->commax		= 100;
  ierr = PetscStrncpy(options->bar, "-----------------\0", 19);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, NULL, "Speedtest Options", "");CHKERRQ(ierr); {
    ierr = PetscOptionsBool("-speed", "Streamline program to only perform necessary operations for performance testing", "", options->perfTest, &options->perfTest, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_ksp", "Use ksp isntead of snes", "", options->useKSP, &options->useKSP, &snesReq);CHKERRQ(ierr);
    if (snesReq && options->useKSP) options->useKSP = PETSC_FALSE;
    ierr = PetscOptionsBool("-vtkout", "enable mesh distribution visualization", "", options->VTKdisp, &options->VTKdisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-all_view", "Turn on all displays", "", options->dispFlag, &options->dispFlag, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-is_view_custom", "Turn on ISView for single threaded", "", options->isView, &options->isView, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-section_view","Turn on SectionView", "", options->sectionDisp, &options->sectionDisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-arr_view", "Turn on array display", "", options->arrayDisp, &options->arrayDisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-coord_view","Turn on coordinate display", "", options->coordDisp, &options->coordDisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vtk_soln","Get solution vector in VTK output", "", options->vtkSoln, &options->vtkSoln, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-f", options->filename, PETSC_MAX_PATH_LEN, &options->fileflg); CHKERRQ(ierr);

    ierr = PetscOptionsEList("-x_periodicity", "The x-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[0]], &bd, NULL);CHKERRQ(ierr);
    options->periodicity[0] = (DMBoundaryType) bd;
    bd = options->periodicity[1];
    ierr = PetscOptionsEList("-y_periodicity", "The y-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[1]], &bd, NULL);CHKERRQ(ierr);
    options->periodicity[1] = (DMBoundaryType) bd;
    bd = options->periodicity[2];
    ierr = PetscOptionsEList("-z_periodicity", "The z-boundary periodicity", "ex12.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[2]], &bd, NULL);CHKERRQ(ierr);
    options->periodicity[2] = (DMBoundaryType) bd;

    bc   = options->bcType;
    ierr = PetscOptionsEList("-bc_type", "Type of boundary condition", "ex12.c", bcTypes, 3, bcTypes[options->bcType], &bc, NULL);CHKERRQ(ierr);
    options->bcType = (BCType) bc;

    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &options->meshSize, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-dim", &options->dim, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-num_field", &options->numFields, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &options->overlap, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-petscfe", "Enable only making a petscFE", "", options->usePetscFE, &options->usePetscFE, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-qorder", &options->qorder, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-refine_dm_level", &options->level, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-refine_limit", &options->refinementLimit, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-max_com", &options->commax, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (options->dispFlag) {
    options->isView = PETSC_TRUE;
    options->sectionDisp = PETSC_TRUE;
    options->arrayDisp = PETSC_TRUE;
    options->coordDisp = PETSC_TRUE;
  }
  if (options->usePetscFE) {
    options->numFields = 1;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode	ierr;
  DM			dmDist;
  const char		*filename = user->filename;
  PetscInt		nFaces = user->meshSize, dim = user->dim, overlap = user->overlap, i, faces[dim];
  PetscBool		hasLabel = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (user->fileflg) {
    char	*dup, filenameAlt[PETSC_MAX_PATH_LEN];
    sprintf(filenameAlt, "%s%s", "./meshes/", (dup = strdup(filename)));
    free(dup);
    ierr = PetscLogStageRegister("READ Mesh Stage", &user->stageREAD);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("READ Mesh", 0, &user->eventREAD);CHKERRQ(ierr);
    ierr = PetscLogStagePush(user->stageREAD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(user->eventREAD, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMPlexCreateFromFile(comm, filenameAlt, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->eventREAD, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, user->filename);CHKERRQ(ierr);
  } else {
    for (i = 0; i < dim; i++){
      /* Make the default box mesh creation with CLI options	*/
      faces[i] = nFaces;
    }
    ierr = PetscLogStageRegister("CREATE Box Mesh Stage", &user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("CREATE Box Mesh", 0, &user->eventCREATE);CHKERRQ(ierr);
    ierr = PetscLogStagePush(user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, faces, NULL, NULL, user->periodicity, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Generated_Box_Mesh");CHKERRQ(ierr);
  }

  ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  dim = user->dim;
  if (!user->fileflg) {
    DM		dmf;
    PetscInt 	level = user->level;
    PetscScalar refinementLimit = user->refinementLimit;
    if (level || refinementLimit) {
      PetscPartitioner	part;

      ierr = PetscLogStageRegister("REFINE Mesh Stage", &user->stageREFINE);CHKERRQ(ierr);
      ierr = PetscLogEventRegister("REFINE Mesh", 0, &user->eventREFINE);CHKERRQ(ierr);
      ierr = PetscLogStagePush(user->stageREFINE);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(user->eventREFINE, 0, 0, 0, 0);CHKERRQ(ierr);
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
      if (level) {
        for (i = 0; i < level; i++) {
          ierr = DMRefine(*dm, comm, &dmf);CHKERRQ(ierr);
          if (dmf) {
            const char	*name;
            ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) dmf, name);CHKERRQ(ierr);
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm = dmf;
          }
          ierr = DMPlexDistribute(*dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
          if (dmDist) {
            const char	*name;
            ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm = dmDist;
            user->distribute = PETSC_TRUE;
          }
        }
      } else {
        ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
        ierr = DMRefine(*dm, comm, &dmf);CHKERRQ(ierr);
        if (dmf) {
          const char *name;

          ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) dmf, name);CHKERRQ(ierr);
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm  = dmf;
        }
        /* Distribute mesh over processes */
        ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
        if (dmDist) {
          const char	*name;
          ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm  = dmDist;
        }
      }
      ierr = PetscLogEventEnd(user->eventREFINE, 0, 0, 0, 0);CHKERRQ(ierr);
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      user->dmRefine = PETSC_TRUE;
    }
  }
  ierr = DMPlexDistribute(*dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    const char	*name;
    ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm = dmDist;
    user->distribute = PETSC_TRUE;
  }
  if (!user->distribute && user->isView) {
    ierr = PetscPrintf(comm, "%s Label View %s\n", user->bar, user->bar);CHKERRQ(ierr);
    ierr = ViewISInfo(comm, *dm);CHKERRQ(ierr);
  }
  if (user->interpolate) {
    DM	dmInterp;
    ierr = DMPlexInterpolate(*dm, &dmInterp);CHKERRQ(ierr);
    if (dmInterp) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmInterp;
      user->interpolate = PETSC_TRUE;
    }
    if (user->bcType == NEUMANN) {
      DMLabel   label;
      ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
      ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    } else if (user->bcType == DIRICHLET) {
      ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {
        DMLabel	label;
        ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
        ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
        ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscErrorCode	ierr;
  DM			cdm = dm;
  MPI_Comm		comm;
  PetscFE		fe;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, user->dim, 1, user->simplex, NULL, user->qorder, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "LaplaceFE");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    if (user->bcType == DIRICHLET && user->interpolate) {
      PetscBool hasLabel;

      ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {
        DMLabel	label;

        ierr = DMCreateLabel(cdm, "marker");CHKERRQ(ierr);
        ierr = DMGetLabel(cdm, "marker", &label);CHKERRQ(ierr);
        ierr = DMPlexMarkBoundaryFaces(cdm, 1, label);CHKERRQ(ierr);
        ierr = DMPlexLabelComplete(cdm, label);CHKERRQ(ierr);
      }
    }
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS 	 ds;
  PetscErrorCode ierr;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  if (!user->useKSP) {
    ierr = PetscDSSetResidual(ds, 0, f0_xytrig_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    user->exactFuncs[0] = AnalyticalSoln2D;
    ierr = PetscDSAddBoundary(ds, user->bcType == DIRICHLET ? (user->fieldBC ? DM_BC_ESSENTIAL_FIELD : DM_BC_ESSENTIAL) : DM_BC_NATURAL, "wall", user->bcType == DIRICHLET ? "marker" : "boundary", 0, 0, NULL, user->fieldBC ? (void (*)(void)) user->exactFields[0] : (void (*)(void)) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(ds, 0, user->exactFuncs[0], user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* 	Main	*/
int main(int argc, char **argv)
{
  MPI_Comm		comm;
  AppCtx		user;
  PetscErrorCode	ierr;
  PetscViewer		genViewer;
  PetscPartitioner	partitioner;
  PetscPartitionerType	partitionername;
  DM			dm;
  SNES			snes;
  KSP			ksp;
  Mat			A, J;
  MatNullSpace		nullspace = NULL;
  IS			bcPointsIS, globalCellNumIS, globalVertNumIS;
  PetscSection		section;
  Vec			funcVecSin, funcVecCos, solVecLocal, solVecGlobal, coordinates, VDot, u, b, AnalyticalSolnVec, resVec;
  PetscInt		i, j, k, numBC = 1, vecsize = 1000, nCoords, nVertex, globalVertSize, globalCellSize, commiter;
  PetscInt		bcField[numBC];
  PetscScalar 		dot, VDotResult, res;
  PetscScalar		*coords, *array;
  char			genInfo[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerStringOpen(comm, genInfo, sizeof(genInfo), &genViewer);CHKERRQ(ierr);

  ierr = ProcessOpts(comm, &user);CHKERRQ(ierr);
  ierr = ProcessMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);
  ierr = PetscMalloc2(1, &user.exactFuncs, 1, &user.exactFields);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &AnalyticalSolnVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution vector");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "RHS vector");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) AnalyticalSolnVec, "Analytical Solution");CHKERRQ(ierr);
  //ierr = AnalyticalSoln(dm, &AnalyticalSolnVec, &user);CHKERRQ(ierr);
  if (user.useKSP) {
    Mat laplacian;

    ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
    ierr = KSPSetDM(ksp,(DM) dm);CHKERRQ(ierr);
    ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);
    if (user.fileflg) {
      ierr = KSPSetComputeOperators(ksp, ComputeLaplacianOperatorWithKSP, &user);CHKERRQ(ierr);
    } else {
      ierr = ComputeLaplacianOperator(&laplacian, &user);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp, laplacian, laplacian);CHKERRQ(ierr);
    }
    ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    ierr = ComputeRHS(ksp, b, &user);CHKERRQ(ierr);
    //ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);CHKERRQ(ierr);
    //ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, u);CHKERRQ(ierr);
    ierr = KSPGetSolution(ksp, &u);CHKERRQ(ierr);
    ierr = KSPGetRhs(ksp, &b);CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-vec_view_u");CHKERRQ(ierr);
    ierr = VecViewFromOptions(b, NULL, "-vec_view_b");CHKERRQ(ierr);
    ierr = VecViewFromOptions(AnalyticalSolnVec, NULL, "-vec_view_soln");CHKERRQ(ierr);
    ierr = KSPReasonViewFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "KSP RESIDUAL: %f\n", res);CHKERRQ(ierr);
    ierr = VecAXPY(u, -1.0, AnalyticalSolnVec);CHKERRQ(ierr);
    ierr = VecNorm(u, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "CALCULATED ERROR: %f\n", res);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&laplacian);CHKERRQ(ierr);
  } else {
    ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

    ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
    A = J;
    if (user.bcType != DIRICHLET) {
      ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullspace);CHKERRQ(ierr);
      ierr = MatSetNullSpace(A, nullspace);CHKERRQ(ierr);
    }
    ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    if (user.fieldBC) {
      ierr = DMProjectField(dm, 0.0, u, user.exactFields, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
    } else {
      ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, b);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, b, u);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetRhs(snes, &b);CHKERRQ(ierr);
    //ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
    ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-vec_view_u");CHKERRQ(ierr);
    ierr = VecViewFromOptions(b, NULL, "-vec_view_b");CHKERRQ(ierr);
    ierr = VecViewFromOptions(AnalyticalSolnVec, NULL, "-vec_view_soln");CHKERRQ(ierr);
    ierr = SNESGetFunction(snes, &resVec, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, u, resVec);CHKERRQ(ierr);
    ierr = VecNorm(resVec, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "RESIDUAL VEC NORM: %f\n", res);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    if (A!=J) ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = PetscFree2(user.exactFuncs, user.exactFields);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  }

  /*	Set up DM and initialize fields OLD
  {
    PetscInt	numDOF[numFields*(dim+1)], numComp[numFields];
    // 	Init number of Field Components
    for (k = 0; k < numFields; k++){numComp[k] = 1;}
    //	Init numDOF[field componentID] = Not Used
    for (k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
    //	numDOF[field componentID] = Used
    numDOF[0] = 1;
    //	bcField[boundary conditionID] = Dirichtlet Val
    bcField[0] = 0;

    //	Assign BC using IS of LOCAL boundaries
    ierr = DMGetStratumIS(dm, "depth", dim, &bcPointsIS);CHKERRQ(ierr);
    ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, "Default_Field");CHKERRQ(ierr);
    ierr = DMSetSection(dm, section);CHKERRQ(ierr);
    if (sectionDisp) {
      ierr = PetscPrintf(comm,"%s Petsc Section View %s\n", bar, bar);CHKERRQ(ierr);
      ierr = PetscSectionView(section, 0);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"%s End Petsc Section View %s\n",bar, bar);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
    ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
    */

  /*	Create PetscFE OLD
  if (usePetscFE) {
    PetscFE	defaultFE;
    ierr = PetscFECreateDefault(comm, dim, dim, simplex, NULL, qorder, &defaultFE);CHKERRQ(ierr);
    ierr = PetscFESetName(defaultFE, "Default_FE");CHKERRQ(ierr);
    ierr = DMSetField(dm, 0, NULL, (PetscObject) defaultFE);CHKERRQ(ierr);
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&defaultFE);CHKERRQ(ierr);
  }
   */

  /*	Create Vector for per process function evaluation	*/

  if (!user.perfTest && user.arrayDisp) {
    ierr = VecCreate(PETSC_COMM_SELF, &funcVecSin);CHKERRQ(ierr);
    ierr = VecSetType(funcVecSin, VECSTANDARD);CHKERRQ(ierr);
    ierr = VecSetSizes(funcVecSin, PETSC_DECIDE, vecsize);CHKERRQ(ierr);
    ierr = VecSetFromOptions(funcVecSin);CHKERRQ(ierr);
    ierr = VecDuplicate(funcVecSin, &funcVecCos);CHKERRQ(ierr);
    ierr = VecSet(funcVecSin, PetscSinReal(PETSC_PI));CHKERRQ(ierr);
    ierr = VecSet(funcVecCos, PetscCosReal(PETSC_PI));CHKERRQ(ierr);
    ierr = VecAssemblyBegin(funcVecSin);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(funcVecSin);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(funcVecCos);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(funcVecCos);CHKERRQ(ierr);
  }

  /* Display Mesh Partition and write mesh to vtk output file */
  if (user.VTKdisp) {
    PetscViewer	vtkviewerpart, vtkviewermesh;
    Vec		partition;
    char	meshName[PETSC_MAX_PATH_LEN];

    ierr = DMPlexCreateRankField(dm, &partition);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewerpart,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewerpart,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewerpart, "partition-map.vtk");CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vtkviewerpart,FILE_MODE_WRITE);CHKERRQ(ierr);
    //ierr = VecView(partition, vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewerpart);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);

    if (user.fileflg) {
      char	*fileEnding, *fixedFile = 0;
      size_t	lenTotal, lenEnding;

      ierr = PetscStrlen(user.filename, &lenTotal);CHKERRQ(ierr);
      ierr = PetscStrrchr(user.filename, '.', &fileEnding);CHKERRQ(ierr);
      ierr = PetscStrlen(fileEnding, &lenEnding);CHKERRQ(ierr);
      if (lenTotal > lenEnding) {
        ierr = PetscMalloc1(lenTotal, &fixedFile);CHKERRQ(ierr);
        ierr = PetscStrncpy(fixedFile, user.filename, lenTotal-lenEnding);CHKERRQ(ierr);
      } else {
        ierr = PetscStrallocpy(user.filename, &fixedFile);CHKERRQ(ierr);
      }
      ierr = PetscStrcat(meshName, fixedFile);CHKERRQ(ierr);
      ierr = PetscFree(fixedFile);CHKERRQ(ierr);
    } else {
      char	dateStr[PETSC_MAX_PATH_LEN] = {"generated-"};
      size_t	stringlen;

      ierr = PetscStrlen(dateStr, &stringlen);CHKERRQ(ierr);
      ierr = PetscGetDate(dateStr+stringlen, 20);CHKERRQ(ierr);
      ierr = PetscStrcat(meshName, dateStr);CHKERRQ(ierr);
    }
    ierr = PetscStrcat(meshName, "-mesh.vtu");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewermesh);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewermesh,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewermesh, meshName);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewermesh,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vtkviewermesh,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerSetUp(vtkviewermesh);CHKERRQ(ierr);
    ierr = DMPlexVTKWriteAll((PetscObject) dm, vtkviewermesh);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewermesh);CHKERRQ(ierr);
  }

  /*	LOOP OVER ALL VERTICES ON LOCAL MESH UNLESS ITS A SPEEDTEST */
  if (!user.perfTest && user.arrayDisp) {
    /*	Perform Function on LOCAL array	*/
    ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
    ierr = VecGetLocalSize(solVecLocal, &nVertex);CHKERRQ(ierr);
    ierr = VecGetArray(solVecLocal, &array);CHKERRQ(ierr);
    if (user.arrayDisp) {PetscPrintf(comm,"%s Array %s\n", user.bar, user.bar);
      ierr = PetscPrintf(comm, "Before Op | After Op\n");CHKERRQ(ierr);
    }
    for(j = 0; j < nVertex; ++j) {
      if (user.arrayDisp) { ierr = PetscPrintf(comm, "%.3f", array[j]);CHKERRQ(ierr); }
      ierr = VecDot(funcVecCos, funcVecSin, &dot);CHKERRQ(ierr);
      array[j] = dot;
      if (user.arrayDisp) {ierr = PetscPrintf(comm, "\t  |%.3f\n", array[j]);CHKERRQ(ierr);}
    }
    if (user.arrayDisp) {
      ierr = PetscPrintf(comm,"%d Number of LOCAL elements\n", nVertex);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"%s Array End %s\n", user.bar, user.bar);CHKERRQ(ierr);
    }
    /*	Put LOCAL with changed values back into GLOBAL	*/
    ierr = VecDestroy(&funcVecSin);CHKERRQ(ierr);
    ierr = VecDestroy(&funcVecCos);CHKERRQ(ierr);
    ierr = VecRestoreArray(solVecLocal, &array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  }

  /*	Perform setup before timing	*/
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);

  /*	Init INSERT_VALUES timing only log	*/
  ierr = PetscLogStageRegister("CommStageINSERT", &user.stageINSERT);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommINSERT", 0, &user.eventINSERT);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageINSERT);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  }
  /*	Push LocalToGlobal time to log	*/
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*	Perform setup before timing	*/
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);

  /*	Init ADD_VALUES Log	*/
  ierr = PetscLogStageRegister("CommStageADDVAL", &user.stageADD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommADDVAL", 0, &user.eventADD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageADD);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    /*	Global to Local aren't implemented	*/
  }
  /*	Push time to log	*/
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*	Perform setup before timing	*/
  ierr = DMCreateGlobalVector(dm, &VDot);CHKERRQ(ierr);
  ierr = VecSet(VDot, 1);CHKERRQ(ierr);
  ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);

  /*	Init VecDot Log	*/
  ierr = PetscLogStageRegister("CommStageGlblVecDot", &user.stageGVD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommGlblVecDot", 0, &user.eventGVD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageGVD);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
    ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  }
  /*	Push time to log	*/
  ierr = VecDestroy(&VDot);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  if (user.coordDisp) {
    /*	Get LOCAL coordinates for debug	*/
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coordinates, &nCoords);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);

    /*	LOOP OVER ALL COORDINATES PAIRS ON LOCAL MESH
     NOTE: This is not the same as looping over values of a matrix A
     representing the "vertices" but instead gives you the (x,y)
     coordinates corresponding to an entry Aij. Rule of thumb for checking
     is that there should be twice as many local coords as local vertices!	*/

    ierr = PetscPrintf(comm,"%s Coords %s\n", user.bar, user.bar);CHKERRQ(ierr);
    for(i = 0; i < nCoords/user.dim; i++) {
      ierr = PetscPrintf(comm,"(%.2f,%.2f)\n", coords[user.dim*i], coords[(user.dim*i)+1]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm,"%d Number of LOCAL coordinates\n",i);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"%s Coords End %s\n", user.bar, user.bar);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  }

  /*	Output vtk of global solution vector	*/
  if (user.vtkSoln) {
    PetscViewer	vtkviewersoln;

    ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewersoln);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewersoln,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk");CHKERRQ(ierr);
    ierr = VecView(solVecGlobal, vtkviewersoln);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln);CHKERRQ(ierr);
  }

  /*	Get Some additional data about the mesh mainly for printing */
  ierr = DMPlexGetVertexNumbering(dm, &globalVertNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalVertNumIS, &globalVertSize);CHKERRQ(ierr);
  ierr = DMPlexGetCellNumbering(dm, &globalCellNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalCellNumIS, &globalCellSize);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm, &partitioner);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscPartitionerGetType(partitioner, &partitionername);CHKERRQ(ierr);

  /*	Aggregate all of the information for printing	*/
  ierr = PetscViewerStringSPrintf(genViewer, "Partitioner Used:%s>%s\n", user.bar + 2, partitionername);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Node Num:%s>%d\n", user.bar + 1, globalVertSize);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Cell Num:%s>%d\n", user.bar + 1, globalCellSize);CHKERRQ(ierr);

  ierr = GeneralInfo(comm, user, genViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
  ierr = VecDestroy(&AnalyticalSolnVec);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
