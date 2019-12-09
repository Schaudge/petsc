static char help[33] = "Test Unstructured Mesh Handling\n";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscds.h>
# include <petsc/private/matimpl.h>
# include <petscsf.h>

# define PETSCVIEWERVTK          "vtk"
# define PETSCVIEWERASCII        "ascii"
# define VECSTANDARD             "standard"

typedef struct {
  PetscLogStage  stageREAD, stageCREATE, stageREFINE, stageINSERT, stageADD, stageGVD, stagePETSCFE, stageCREATEDS, stageZEROGVD;
  PetscLogEvent  eventREAD, eventCREATE, eventREFINE, eventINSERT, eventADD, eventGVD, eventPETSCFE, eventCREATEDS, eventZEROGVD;
  PetscBool      simplex, perfTest, fileflg, distribute, interpolate, dmRefine, VTKdisp, vtkSoln, meshout, balance;
  /* Domain and mesh definition */
  PetscInt       dim, numFields, overlap, qorder, level, commax, VSL, VSG;
  PetscScalar    refinementLimit;
  char           filename[2048];    /* The optional mesh file */
  char           bar[19];
  VecType	 ctype;
} AppCtx;

/* ADDITIONAL FUNCTIONS */
PetscErrorCode GeneralInfo(MPI_Comm comm, AppCtx user, PetscViewer genViewer)
{
  PetscErrorCode ierr;
  const char    *string;

  PetscFunctionBeginUser;
  ierr = PetscViewerStringSPrintf(genViewer, "Dimension of mesh:%s>%d\n", user.bar + 3, user.dim);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Number of Fields:%s>%d", user.bar + 2, user.numFields);CHKERRQ(ierr);
  if (user.numFields == 100) {
    ierr = PetscViewerStringSPrintf(genViewer, "(default)");CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Ghost point overlap:%s>%d\n", user.bar + 5, user.overlap);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "DM Partition Balance?:%s>%s\n", user.bar + 7, user.balance ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
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

  ierr = PetscViewerStringSPrintf(genViewer, "DM Vec Type Used:%s>%s\n", user.bar + 2, user.ctype);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Comm Loop Bound:%s>%d\n", user.bar + 1, user.commax);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "VTKoutput mode:%s>%s\n", user.bar, user.VTKdisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "%s General Info %s\n", user.bar + 2, user.bar + 2);CHKERRQ(ierr);
  ierr = PetscViewerStringGetStringRead(genViewer, &string, NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, string);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s End General Info %s\n", user.bar + 2, user.bar + 5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ThrowItInTheBin(MPI_Comm comm, PetscScalar local, PetscInt root, const char header[])
{
  PetscMPIInt     rank, size;
  PetscScalar     *rootArray, *binnedArray;
  PetscInt        i, numBins;
  PetscInt        *bins;
  Vec             vecPerProcess;
  VecTagger       tagger;
  VecTaggerBox    *box;
  IS              PerProcessTaggedIS;
  char            completedHeader[PETSC_MAX_PATH_LEN] = {'\0'};
  size_t          len;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(header, 4);
  ierr = PetscStrlen(header, &len);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscMalloc1(size, &rootArray);CHKERRQ(ierr);
  }
  ierr = MPI_Gather(&local, 1, MPIU_SCALAR, rootArray, 1, MPIU_SCALAR, root, comm);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);
  ierr = PetscSortReal(size, rootArray);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, size, rootArray, &vecPerProcess);CHKERRQ(ierr);
  ierr = VecSetUp(vecPerProcess);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecPerProcess);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecPerProcess);CHKERRQ(ierr);
  ierr = VecUniqueEntries(vecPerProcess, &numBins, &binnedArray);CHKERRQ(ierr);
  ierr = PetscCalloc1(numBins, &bins);CHKERRQ(ierr);
  for (i = 0; i < numBins; ++i) {
    ierr = VecTaggerCreate(PETSC_COMM_SELF, &tagger);CHKERRQ(ierr);
    ierr = VecTaggerSetType(tagger, VECTAGGERABSOLUTE);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &box);CHKERRQ(ierr);
    box->min = binnedArray[i]-0.5;
    box->max = binnedArray[i]+0.5;
    ierr = VecTaggerAbsoluteSetBox(tagger, box);CHKERRQ(ierr);
    ierr = VecTaggerSetUp(tagger);CHKERRQ(ierr);
    ierr = PetscFree(box);CHKERRQ(ierr);
    ierr = VecTaggerComputeIS(tagger, vecPerProcess, &PerProcessTaggedIS);CHKERRQ(ierr);
    ierr = ISGetSize(PerProcessTaggedIS, &bins[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&PerProcessTaggedIS);CHKERRQ(ierr);
    ierr = VecTaggerDestroy(&tagger);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vecPerProcess);CHKERRQ(ierr);
  ierr = PetscStrcpy(completedHeader, header);CHKERRQ(ierr);
  ierr = PetscStrcat(completedHeader, " Per Process Range: %.0f - %.0f\n");CHKERRQ(ierr);
  ierr = PetscPrintf(comm, completedHeader, binnedArray[0], binnedArray[numBins-1]);CHKERRQ(ierr);
  for (i = 0; i < numBins; i++) {
    if (!i) {
      ierr = PetscPrintf(comm, "Num Per Proc. - Num Proc.\n");CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "\t%5.0f - %d\n", binnedArray[i], bins[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(rootArray);CHKERRQ(ierr);
  ierr = PetscFree(binnedArray);CHKERRQ(ierr);
  ierr = PetscFree(bins);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* PARTITIONING */
static PetscErrorCode MatPartitioningApply_Cube(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       n,N,p,rstart,rend,*color,col,x,y,z,i,subcubes;
  PetscMPIInt    size, rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)part),&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  if (part->n != size) SETERRQ(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Currently only supports one domain per processor");
  p = (PetscInt)PetscCbrtReal((PetscReal)part->n);
  if (p*p*p != part->n) SETERRQ2(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Square partitioning requires \"perfect square\" number of domains. p = %D n = %D", p, part->n);

  ierr = MatGetSize(part->adj,&N,NULL);CHKERRQ(ierr);
  n = (PetscInt)PetscCbrtReal((PetscReal)N);
  if (n*n*n != N) SETERRQ(PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Square partitioning requires square domain");
  if (n%p != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Square partitioning requires p to divide n");

  ierr = MatGetOwnershipRange(part->adj,&rstart,&rend);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "RSTART %D REND %D\nn %D p %D N %D\n", rstart, rend, n, p, N);
  ierr = PetscMalloc1(rend-rstart, &color);CHKERRQ(ierr);
  /* for (int cell=rstart; cell<rend; cell++) { color[cell-rstart] = ((cell%n) < (n/2)) + 2 * ((cell/n) <x (n/2)); } */
  col = n/p;
  subcubes = p/2;
  i = 0;
  if (rank) {
    n = 0;
  }
  for (z = 0; z < n; ++z) {
    for (y = 0; y < n; ++y) {
      for (x = 0; x < n; ++x) {
        PetscPrintf(PETSC_COMM_WORLD, "x %3D y %3D z %3D i %3D -> ", x, y, z, i);
        color[i] = (x/col) + (p*(y/col)) + (p*p*(z/col));
        ++i;
        PetscPrintf(PETSC_COMM_WORLD, "%3D COLOR SUBCUBE %2D [EQ1: %3D EQ2: %3D EQ3 %3D]\n", color[i-1], subcubes, x/col, p*(y/col), p*p*(z/col));
       }
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),rend-rstart,color,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningCreate_Cube(MatPartitioning part)
{
  part->ops->apply   = MatPartitioningApply_Cube;
  part->ops->view    = 0;
  part->ops->destroy = 0;
  return(0);
}

/* GENERAL PREPROCESSING */
static PetscErrorCode ProcessOpts(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode        ierr;

  PetscFunctionBeginUser;
  options->simplex              = PETSC_FALSE;
  options->perfTest             = PETSC_FALSE;
  options->fileflg              = PETSC_FALSE;
  options->distribute           = PETSC_FALSE;
  options->interpolate          = PETSC_TRUE;
  options->dmRefine             = PETSC_FALSE;
  options->VTKdisp              = PETSC_FALSE;
  options->vtkSoln              = PETSC_FALSE;
  options->meshout		= PETSC_FALSE;
  options->balance              = PETSC_FALSE;
  options->filename[0]          = '\0';
  options->dim                  = 2;
  options->numFields            = 1;
  options->overlap              = 0;
  options->qorder               = -1;
  options->level                = 0;
  options->refinementLimit      = 0.0;
  options->commax               = 100;
  options->VSL                  = 0;
  options->VSG                  = 0;
  ierr = PetscStrncpy(options->bar, "-----------------\0", 19);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, NULL, "Speedtest Options", "");CHKERRQ(ierr); {
    ierr = PetscOptionsBool("-speed", "Streamline program to only perform necessary operations for performance testing", "", options->perfTest, &options->perfTest, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vtk_partition", "enable mesh distribution visualization", "", options->VTKdisp, &options->VTKdisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vtk_soln","Get solution vector in VTK output", "", options->vtkSoln, &options->vtkSoln, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mesh_out","Output mesh in vtk format", "", options->meshout, &options->meshout, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-f", options->filename, PETSC_MAX_PATH_LEN, &options->fileflg); CHKERRQ(ierr);

    ierr = PetscOptionsGetInt(NULL, NULL, "-dim", &options->dim, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-num_fields", &options->numFields, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &options->overlap, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-qorder", &options->qorder, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-refine_dm_level", &options->level, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-refine_limit", &options->refinementLimit, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-max_com", &options->commax, NULL);CHKERRQ(ierr);

    ierr = PetscOptionsGetInt(NULL, NULL, "-vec_size_local", &options->VSL, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-vec_size_global", &options->VSG, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!options->VSG) {
    options->VSG = PETSC_DETERMINE;
    ierr = PetscPrintf(comm, "DUMMY VEC GLOBAL SIZE PETSC_DETERMINE\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode  ierr;
  DM              dmDist;
  const char      *filename = user->filename;
  PetscInt        dim = user->dim, overlap = user->overlap, i;

  PetscFunctionBeginUser;
  ierr = MatPartitioningRegister("cube", MatPartitioningCreate_Cube);CHKERRQ(ierr);
  if (user->fileflg) {
    char        *dup, filenameAlt[PETSC_MAX_PATH_LEN];
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
    ierr = PetscLogStageRegister("CREATE Box Mesh Stage", &user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("CREATE Box Mesh", 0, &user->eventCREATE);CHKERRQ(ierr);
    ierr = PetscLogStagePush(user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, NULL, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Generated_Box_Mesh");CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  dim = user->dim;
  if (!user->fileflg) {
    DM          dmf;
    PetscInt    level = user->level;
    PetscScalar refinementLimit = user->refinementLimit;
    if (level || refinementLimit) {
      PetscPartitioner  part;

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
            const char  *name;
            ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) dmf, name);CHKERRQ(ierr);
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm = dmf;
          }
          ierr = DMPlexDistribute(*dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
          if (dmDist) {
            const char  *name;
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
          const char    *name;
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
    const char  *name;
    ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm = dmDist;
    user->distribute = PETSC_TRUE;
  }
  if (user->interpolate) {
    DM  dmInterp;
    ierr = DMPlexInterpolate(*dm, &dmInterp);CHKERRQ(ierr);
    if (dmInterp) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmInterp;
      user->interpolate = PETSC_TRUE;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
# if defined(PETSC_HAVE_CUDA)
  ierr = DMSetVecType(*dm, VECCUDA);CHKERRQ(ierr);
# endif
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscFE               fe;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("CommStagePETSCFE", &user->stagePETSCFE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommPETSCFE", 0, &user->eventPETSCFE);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stagePETSCFE);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user->eventPETSCFE, 0, 0, 0, 0);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, user->dim, user->numFields, user->simplex, NULL, user->qorder, &fe);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->eventPETSCFE, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "LaplaceFE");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("CommStageCREATEDS", &user->stageCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommCREATEDS", 0, &user->eventCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stageCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user->eventCREATEDS, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->eventCREATEDS, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*      Main    */
int main(int argc, char **argv)
{
  MPI_Comm              comm;
  PetscMPIInt		rank, size;
  AppCtx                user;
  PetscErrorCode        ierr;
  PetscViewer           genViewer;
  PetscPartitioner      partitioner;
  PetscPartitionerType  partitionername;
  DM                    dm;
  IS                    globalCellNumIS, globalVertNumIS;
  Vec                   solVecLocal, solVecGlobal, VDot, VDotZERO, dummyVecGlobal, dummyVecGlobal2, dummyVecLocal;
  PetscInt              globalVertSize, globalCellSize, commiter;
  PetscScalar           VDotResult;
  char                  genInfo[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscViewerStringOpen(comm, genInfo, sizeof(genInfo), &genViewer);CHKERRQ(ierr);

  ierr = ProcessOpts(comm, &user);CHKERRQ(ierr);
  ierr = ProcessMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  if (user.distribute) {
    MPI_Group             in, out;
    PetscSF               pointSF, processSF, sectionSF;
    IS                    processRanks;
    const PetscMPIInt     *RootRanks, *LeafRanks;
    PetscMPIInt           size_in, size_out, size_total;
    PetscInt              nRanks, nRootRanks, nLeafRanks;

    ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
    ierr = DMPlexCreateProcessSF(dm, pointSF, &processRanks, &processSF);CHKERRQ(ierr);
    ierr = PetscSFSetUp(processSF);CHKERRQ(ierr);
    ierr = DMGetSectionSF(dm, &sectionSF);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sectionSF);CHKERRQ(ierr);
    ierr = PetscSFViewFromOptions(pointSF, NULL, "-petscsf_point_view");CHKERRQ(ierr);
    ierr = PetscSFViewFromOptions(processSF, NULL, "-petscsf_processsf_view");CHKERRQ(ierr);
    ierr = ISViewFromOptions(processRanks, NULL, "-is_process_view");CHKERRQ(ierr);
    ierr = PetscSFViewFromOptions(sectionSF, NULL, "-petscsf_section_view");CHKERRQ(ierr);
    ierr = PetscSFGetGroups(processSF, &in, &out);CHKERRQ(ierr);
    ierr = MPI_Group_size(in, &size_in);CHKERRQ(ierr);
    ierr = MPI_Group_size(out, &size_out);CHKERRQ(ierr);
    size_total = size_in + size_out;
    ierr = PetscSFGetRootRanks(sectionSF, &nRootRanks, &RootRanks, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(sectionSF, &nLeafRanks, &LeafRanks, NULL, NULL);CHKERRQ(ierr);
    nRanks = nRootRanks + nLeafRanks;
    ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
    ierr = ThrowItInTheBin(comm,(PetscScalar) size_total, 0, "PetscSF Neighbors FROM GROUPS");CHKERRQ(ierr);
    ierr = ThrowItInTheBin(comm,(PetscScalar) nRanks, 0, "PetscSF Neighbors FROM SECTION");CHKERRQ(ierr);
    ierr = PetscSFDestroy(&processSF);CHKERRQ(ierr);
    ierr = ISDestroy(&processRanks);CHKERRQ(ierr);
  }
  /* Display Mesh Partition and write mesh to vtk output file */
  if (user.VTKdisp) {
    PetscViewer vtkviewerpart;
    Vec         partition;
    char        dateStr[PETSC_MAX_PATH_LEN] = {"partition-map-generated-"}, rawdate[PETSC_MAX_PATH_LEN] = {"\0"}, meshName[PETSC_MAX_PATH_LEN] = {"\0"};
    size_t      stringlen;

    ierr = PetscStrlen(dateStr, &stringlen);CHKERRQ(ierr);
    ierr = PetscGetDate(rawdate, 20);CHKERRQ(ierr);
    ierr = PetscStrcat(dateStr, rawdate+11);CHKERRQ(ierr);
    ierr = PetscStrcat(meshName, dateStr);CHKERRQ(ierr);
    ierr = PetscStrcat(meshName, ".vtu");CHKERRQ(ierr);
    ierr = DMPlexCreateRankField(dm, &partition);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewerpart,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewerpart,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewerpart, meshName);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vtkviewerpart,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = VecView(partition, vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewerpart);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);

  } else if (user.meshout) {
    PetscViewer vtkviewermesh;
    char        meshName[PETSC_MAX_PATH_LEN] = {"\0"};
    if (user.fileflg) {
      char      *fileEnding, *fixedFile = 0;
      size_t    lenTotal, lenEnding;

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
      char      dateStr[PETSC_MAX_PATH_LEN] = {"generated-"}, rawdate[PETSC_MAX_PATH_LEN] = {"\0"};
      size_t    stringlen;

      ierr = PetscStrlen(dateStr, &stringlen);CHKERRQ(ierr);
      ierr = PetscGetDate(rawdate, 20);CHKERRQ(ierr);
      ierr = PetscStrcat(dateStr, rawdate+11);CHKERRQ(ierr);
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

  /*    Perform setup before timing     */
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = VecDuplicate(solVecGlobal, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(solVecLocal, &dummyVecLocal);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 0.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecLocal, 0.0);CHKERRQ(ierr);
  ierr = VecAXPY(solVecGlobal, 0.0, dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecAXPY(solVecLocal, 0.0, dummyVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecLocal);CHKERRQ(ierr);

  /*	AXPY to flush kernels	*/
  ierr = VecCreate(PETSC_COMM_SELF, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecSetSizes(dummyVecGlobal, user.VSL, user.VSG);CHKERRQ(ierr);
  ierr = VecSetFromOptions(dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(dummyVecGlobal, &dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 1.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal2, 1.0);CHKERRQ(ierr);
  ierr = VecAXPY(dummyVecGlobal, 1.0, dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal2);CHKERRQ(ierr);

  /*    Init INSERT_VALUES timing only log      */
  ierr = PetscLogStageRegister("CommStageINSERT", &user.stageINSERT);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommINSERT", 0, &user.eventINSERT);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageINSERT);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  }
  /*    Push LocalToGlobal time to log  */
  ierr = PetscLogEventEnd(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);

  /*    Perform setup before timing     */
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);

  /*	AXPY to flush kernels	*/
  ierr = VecCreate(PETSC_COMM_SELF, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecSetSizes(dummyVecGlobal, user.VSL, user.VSG);CHKERRQ(ierr);
  ierr = VecSetFromOptions(dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(dummyVecGlobal, &dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 1.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal2, 1.0);CHKERRQ(ierr);
  ierr = VecAXPY(dummyVecGlobal, 1.0, dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal2);CHKERRQ(ierr);

  /*    Init ADD_VALUES Log     */
  ierr = PetscLogStageRegister("CommStageADDVAL", &user.stageADD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommADDVAL", 0, &user.eventADD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageADD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    /*  Global to Local aren't implemented      */
  }
  /*    Push time to log        */
  ierr = PetscLogEventEnd(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);

  /*    Perform setup before timing     */
  ierr = DMCreateGlobalVector(dm, &VDot);CHKERRQ(ierr);
  ierr = VecSet(VDot, 1);CHKERRQ(ierr);
  ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);

  /*	AXPY to flush kernels	*/
  ierr = VecCreate(PETSC_COMM_SELF, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecSetSizes(dummyVecGlobal, user.VSL, user.VSG);CHKERRQ(ierr);
  ierr = VecSetFromOptions(dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(dummyVecGlobal, &dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 1.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal2, 1.0);CHKERRQ(ierr);
  ierr = VecAXPY(dummyVecGlobal, 1.0, dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal2);CHKERRQ(ierr);

  /*    Init VecDot Log */
  ierr = PetscLogStageRegister("CommStageGlblVecDot", &user.stageGVD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommGlblVecDot", 0, &user.eventGVD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageGVD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
    ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  }
  /*    Push time to log        */
  ierr = PetscLogEventEnd(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = VecDestroy(&VDot);CHKERRQ(ierr);

  /*    Perform setup before timing     */
  ierr = VecCreate(comm, &VDotZERO);CHKERRQ(ierr);
  ierr = VecSetSizes(VDotZERO, 0, 0);CHKERRQ(ierr);
  ierr = VecSetUp(VDotZERO);CHKERRQ(ierr);
  ierr = VecDotBegin(VDotZERO, VDotZERO, &VDotResult);CHKERRQ(ierr);
  ierr = VecDotEnd(VDotZERO, VDotZERO, &VDotResult);CHKERRQ(ierr);

  /*	AXPY to flush kernels	*/
  ierr = VecCreate(PETSC_COMM_SELF, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecSetSizes(dummyVecGlobal, user.VSL, user.VSG);CHKERRQ(ierr);
  ierr = VecSetFromOptions(dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(dummyVecGlobal, &dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 1.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal2, 1.0);CHKERRQ(ierr);
  ierr = VecAXPY(dummyVecGlobal, 1.0, dummyVecGlobal2);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal2);CHKERRQ(ierr);

  /*    Init VecDot Zero Size Log */
  ierr = PetscLogStageRegister("CommStageZEROVecDot", &user.stageZEROGVD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommZEROVecDot", 0, &user.eventZEROGVD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageZEROGVD);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventZEROGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = VecDotBegin(VDotZERO, VDotZERO, &VDotResult);CHKERRQ(ierr);
    ierr = VecDotEnd(VDotZERO, VDotZERO, &VDotResult);CHKERRQ(ierr);
  }
  /*    Push time to log        */
  ierr = PetscLogEventEnd(user.eventZEROGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = VecDestroy(&VDotZERO);CHKERRQ(ierr);

  /*    Output vtk of global solution vector    */
  if (user.vtkSoln) {
    PetscViewer vtkviewersoln;

    ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewersoln);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewersoln,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk");CHKERRQ(ierr);
    ierr = VecView(solVecGlobal, vtkviewersoln);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln);CHKERRQ(ierr);
  }

  /*    Get Some additional data about the mesh mainly for printing */
  ierr = DMPlexGetVertexNumbering(dm, &globalVertNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalVertNumIS, &globalVertSize);CHKERRQ(ierr);
  ierr = DMPlexGetCellNumbering(dm, &globalCellNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalCellNumIS, &globalCellSize);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm, &partitioner);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscPartitionerGetType(partitioner, &partitionername);CHKERRQ(ierr);
  ierr = DMPlexGetPartitionBalance(dm, &user.balance);CHKERRQ(ierr);

  /*    Aggregate all of the information for printing   */
  ierr = PetscViewerStringSPrintf(genViewer, "Partitioner Used:%s>%s\n", user.bar + 2, partitionername);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Node Num:%s>%d\n", user.bar + 1, globalVertSize);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Cell Num:%s>%d\n", user.bar + 1, globalCellSize);CHKERRQ(ierr);
  ierr = DMGetVecType(dm, &user.ctype);CHKERRQ(ierr);

  ierr = GeneralInfo(comm, user, genViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
