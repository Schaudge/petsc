const char help[] = "Visualize PetscFE finite elements with SAWs";

#include <petsc.h>
#include <petscviewersaws.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmimpl.h>
#include <SAWs.h>

typedef struct _n_PetscFESAWsArrayLink *PetscFESAWsArrayLink;

struct _n_PetscFESAWsArrayLink
{
  void                *mem;
  PetscFESAWsArrayLink next;
};

static PetscErrorCode PetscFESAWsArrayLinkDestroy(PetscFESAWsArrayLink *link)
{
  PetscFunctionBegin;
  PetscCall(PetscFree((*link)->mem));
  PetscCall(PetscFree(*link));
  PetscFunctionReturn(0);
}

typedef struct _p_PetscFESAWs *PetscFESAWs;

struct _PetscFESAWsOps {};

struct _p_PetscFESAWs
{
  PETSCHEADER(struct _PetscFESAWsOps);
  size_t               dir_alloc;
  size_t               dir_tip;
  char                *dir;
  PetscFESAWsArrayLink arrays;
  int                  number_of_elements;
};

PetscClassId PETSCFESAWS_CLASSID;

static PetscErrorCode PetscFESAWsDestroy(PetscFESAWs *fes)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsBlock((PetscObject) *fes));
  {
    PetscFESAWsArrayLink head = (*fes)->arrays;
    while (head) {
      PetscFESAWsArrayLink next = head->next;
      PetscCall(PetscFESAWsArrayLinkDestroy(&head));
      head = next;
    }
  }
  PetscCall(PetscObjectSAWsViewOff((PetscObject) *fes));
  PetscCall(PetscHeaderDestroy(fes));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsCreateArray(PetscFESAWs fes, SAWs_Data_type dtype, size_t count, void *data_pointer)
{
  MPI_Datatype mpi_dtype = MPI_BYTE;
  PetscFESAWsArrayLink link;
  int          dsize;

  PetscFunctionBegin;
  switch (dtype) {
  case SAWs_STRING:
  case SAWs_CHAR:
    mpi_dtype = MPI_CHAR;
    break;
  case SAWs_INT:
    mpi_dtype = MPI_INT;
    break;
  case SAWs_FLOAT:
    mpi_dtype = MPI_FLOAT;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscFESAWs does not create arrays of this type");
  }
  PetscCallMPI(MPI_Type_size(mpi_dtype,&dsize));
  PetscCall(PetscNewLog(fes, &link));
  link->next = fes->arrays;
  fes->arrays = link;
  PetscCall(PetscMalloc(count*dsize, &(link->mem)));
  if (dtype == SAWs_STRING) {
    *((void **) data_pointer) = &(link->mem);
  } else {
    *((void **) data_pointer) = link->mem;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsWriteProperty(PetscFESAWs fes, const char *property, const void *data, int count, SAWs_Memory_type mtype, SAWs_Data_type dtype)
{
  size_t buf_size = fes->dir_alloc + BUFSIZ;
  size_t printed_size;
  char   *buf;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(buf_size, &buf));
  PetscCall(PetscSNPrintfCount(buf, buf_size, "%s/%s", &printed_size, fes->dir, property));
  if (printed_size >= buf_size) {
    PetscCall(PetscFree(buf));
    buf_size = printed_size + 1;
    PetscCall(PetscMalloc1(buf_size, &buf));
    PetscCall(PetscSNPrintfCount(buf, buf_size, "%s/%s", &printed_size, fes->dir, property));
  }
  PetscStackCallSAWs(SAWs_Register,(buf, (void *) data, count, mtype, dtype));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscFESAWsView_SAWs(PetscFESAWs fes, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectName((PetscObject)fes));
  PetscCall(PetscObjectViewSAWs((PetscObject) fes, viewer));
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_elements", &(fes->number_of_elements), 1, SAWs_READ, SAWs_INT ));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsCreate(MPI_Comm comm, PetscFESAWs *fes)
{
  PetscFunctionBegin;
  PetscCall(PetscHeaderCreate(*fes, PETSCFESAWS_CLASSID, "PetscFESAWs", "PetscFE SAWs Manager", "", comm, PetscFESAWsDestroy, PetscFESAWsView_SAWs));
  (*fes)->number_of_elements = 0;
  PetscCall(PetscObjectSAWsSetBlock((PetscObject) *fes, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsSetBaseDirectory(PetscFESAWs fes, const char dir[])
{
  PetscFunctionBegin;
  PetscCall(PetscFree(fes->dir));
  fes->dir_alloc = BUFSIZ;
  PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
  PetscCall(PetscSNPrintfCount(fes->dir, fes->dir_alloc, "/PETSc/%s/", &(fes->dir_tip), dir));
  fes->dir_tip--;
  if (fes->dir_tip >= fes->dir_alloc) {
    PetscCall(PetscFree(fes->dir));
    fes->dir_alloc = (fes->dir_tip + 1) * 2;
    PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
    PetscCall(PetscSNPrintfCount(fes->dir, fes->dir_alloc, "/PETSc/%s/", &(fes->dir_tip), dir));
    fes->dir_tip--;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsDirectoryPush(PetscFESAWs fes, const char dir[])
{
  size_t tip_inc;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintfCount(&fes->dir[fes->dir_tip], fes->dir_alloc - fes->dir_tip, "%s/", &tip_inc, dir));
  tip_inc--;
  if (fes->dir_tip + tip_inc >= fes->dir_alloc) {
    const char *dir_old = fes->dir;

    fes->dir_alloc = (fes->dir_tip + tip_inc + 1) * 2;
    PetscCall(PetscMalloc1(fes->dir_alloc, &(fes->dir)));
    PetscCall(PetscArraycpy(fes->dir, dir_old, fes->dir_tip));
    PetscCall(PetscSNPrintfCount(&fes->dir[fes->dir_tip], fes->dir_alloc - fes->dir_tip, "%s/", &tip_inc, dir));
    tip_inc--;
    PetscCall(PetscFree(dir_old));
  }
  fes->dir_tip += tip_inc;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsDirectoryPop(PetscFESAWs fes)
{
  PetscFunctionBegin;
  while (fes->dir_tip > 0 && fes->dir[fes->dir_tip-1] == '/') fes->dir_tip--;
  while (fes->dir_tip > 0 && fes->dir[fes->dir_tip-1] != '/') fes->dir_tip--;
  fes->dir[fes->dir_tip] = '\0';
  PetscFunctionReturn(0);
}

PetscFESAWs PetscFESAWsManager = NULL;

static PetscErrorCode PetscFESAWsViewReferenceElement(PetscFESAWs fes, DM refel)
{
  PetscInt num_points;
  PetscInt v_start, v_end;
  PetscInt dim;
  int num_coords, offset;
  int *i_array;
  Vec coord_vec;
  float *coords;
  PetscScalar *these_coords;
  DM coord_dm;

  PetscFunctionBegin;
  PetscCall(DMPlexGetChart(refel, NULL, &num_points));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 2*num_points + 1, &i_array));
  i_array[2*num_points] = num_points;
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_mesh_points", &i_array[2*num_points], 1, SAWs_READ, SAWs_INT));

  PetscCall(DMGetCoordinateDim(refel, &dim));
  PetscCall(DMPlexGetDepthStratum(refel, 0, &v_start, &v_end));
  num_coords = 0;
  for (PetscInt v = v_start; v < v_end; v++) {
    PetscInt star_size = 0;
    PetscInt *star = NULL;

    PetscCall(DMPlexGetTransitiveClosure(refel, v, PETSC_FALSE, &star_size, &star));
    num_coords += star_size * dim;
    PetscCall(DMPlexRestoreTransitiveClosure(refel, v, PETSC_FALSE, &star_size, &star));
  }
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_FLOAT, num_coords * dim, &coords));
  PetscCall(DMGetCoordinatesLocalNoncollective(refel, &coord_vec));
  PetscCall(DMGetCoordinateDM(refel, &coord_dm));
  PetscCall(PetscMalloc1(num_coords * dim, &these_coords));
  offset = 0;
  for (PetscInt p = 0; p < num_points; p++) {
    DMPolytopeType ptype;
    char point_string[3];
    PetscInt csize = num_coords * dim;
    PetscInt p_num_verts;
    PetscInt p_dim;
    PetscInt coord_start = offset;

    PetscCall(PetscSNPrintf(point_string, 3, "%d", p));
    PetscCall(PetscFESAWsDirectoryPush(fes, point_string));
    PetscCall(DMPlexGetCellType(refel, p, &ptype));
    PetscCall(DMPlexGetPointDepth(refel, p, &p_dim));
    PetscCall(PetscFESAWsWriteProperty(fes, "polytope", &DMPolytopeTypes[ptype], 1, SAWs_READ, SAWs_STRING));
    PetscCall(DMPlexVecGetClosure(coord_dm, NULL, coord_vec, p, &csize, &these_coords));
    i_array[2*p] = p_num_verts = csize / dim;
    i_array[2*p+1] = p_dim;
    for (PetscInt d = 0; d < dim; d++) {
      for (PetscInt v = 0; v < p_num_verts; v++) {
        coords[offset++] = PetscRealPart(these_coords[v * dim + d]);
      }
    }
    PetscCall(PetscFESAWsWriteProperty(fes, "number_of_vertices", &i_array[2*p], 1, SAWs_READ, SAWs_INT));
    PetscCall(PetscFESAWsWriteProperty(fes, "dimension", &i_array[2*p+1], 1, SAWs_READ, SAWs_INT));
    PetscCall(PetscFESAWsWriteProperty(fes, "coordinates", &coords[coord_start], csize, SAWs_READ, SAWs_FLOAT));
    PetscCall(PetscFESAWsDirectoryPop(fes));
  }
  PetscCall(PetscFree(these_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsViewBasisSpace(PetscFESAWs fes, PetscSpace sp)
{
  PetscInt min_degree, max_degree;
  PetscInt *i_array;

  PetscFunctionBegin;
  PetscCall(PetscSpaceGetDegree(sp, &min_degree, &max_degree));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 2, &i_array));
  i_array[0] = min_degree;
  i_array[1] = max_degree;
  PetscCall(PetscFESAWsWriteProperty(fes, "minimum_degree", &i_array[0], 1, SAWs_READ, SAWs_INT));
  PetscCall(PetscFESAWsWriteProperty(fes, "maximum_degree", &i_array[1], 1, SAWs_READ, SAWs_INT));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsViewDualSpace(PetscFESAWs fes, PetscDualSpace dsp)
{
  DM          dm;
  PetscInt    form_degree, dim;
  PetscBool   continuous;
  size_t      var_len;
  char      **variance;
  const char *variance_static;
  const char *variances[] = {
    "invariant | \\\\(H^1\\\\) | \\\\(0\\\\)-form",
    "covariant | \\\\(H(\\\\mathrm{curl})\\\\) | \\\\(1\\\\)-form", 
    "contravariant | \\\\(H(\\\\mathrm{div})\\\\) | \\\\((\\\\star X)\\\\)-form", 
    "\\\\(L^2\\\\) | \\\\(X\\\\)-form",
    "\\\\(X\\\\)-form"
  };
  int          continuity_index;
  static const char *continuities[] = {
    "discontinuous", 
    "continuous", 
    "tangentially continuous", 
    "normally continuous", 
    "trace continuous", 
  };
  const char  dims[] = "0123456789";

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(PetscDualSpaceGetFormDegree(dsp, &form_degree));
  PetscCall(PetscDualSpaceLagrangeGetContinuity(dsp, &continuous));
  variance_static =
    (form_degree == 0) ? variances[0] :
    (form_degree == dim) ? variances[3] :
    (form_degree == 1) ? variances[1] :
    (form_degree == -(dim-1)) ? variances[2] :
    variances[4];
  PetscCall(PetscStrlen(variance_static, &var_len));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, var_len + 1, &variance));
  PetscCall(PetscArraycpy(*variance, variance_static, var_len));
  (*variance)[var_len] = '\0';
  for (size_t i = 0; i < var_len; i++) {
    if ((*variance)[i] == 'X') (*variance)[i] = dims[PetscAbsInt(form_degree)];
  }
  PetscCall(PetscFESAWsWriteProperty(fes, "variance", variance, 1, SAWs_READ, SAWs_STRING));

  continuity_index =
    (!continuous) ? 0 :
    (form_degree == 0) ? 1 :
    (form_degree == 1) ? 2 :
    (form_degree == -(dim-1)) ? 3 :
    4;
  PetscCall(PetscFESAWsWriteProperty(fes, "continuity", &continuities[continuity_index], 1, SAWs_READ, SAWs_STRING));

  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESAWsAddFE(PetscFESAWs fes, PetscFE fe)
{
  char num_string[3];

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(num_string, 3, "%d", fes->number_of_elements));
  PetscCall(PetscFESAWsDirectoryPush(fes, num_string));
  fes->number_of_elements++;
  {
    PetscInt dim, Nb, Nc;
    PetscSpace space;
    PetscDualSpace dsp;
    DM refel;
    int *i_dim_Nb_Nc;
    const char *name;
    const char *prefix;
    char **aname, **aprefix;
    size_t name_len, prefix_len;

    PetscCall(PetscObjectName((PetscObject) fe));
    PetscCall(PetscObjectGetName((PetscObject) fe, &name));
    PetscCall(PetscStrlen(name, &name_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, name_len+1, &aname));
    PetscCall(PetscArraycpy(*aname, name, name_len+1));
    (*aname)[name_len] = '\0';
    PetscCall(PetscFESAWsWriteProperty(fes, "name", aname, 1, SAWs_READ, SAWs_STRING));

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject) fe, &prefix));
    PetscCall(PetscStrlen(prefix, &prefix_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_STRING, prefix_len+1, &aprefix));
    PetscCall(PetscArraycpy(*aprefix, prefix, prefix_len));
    (*aprefix)[prefix_len] = '\0';
    PetscCall(PetscFESAWsWriteProperty(fes, "options_prefix", aprefix, 1, SAWs_READ, SAWs_STRING));

    PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 3, &i_dim_Nb_Nc));

    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    i_dim_Nb_Nc[0] = dim;
    PetscCall(PetscFESAWsWriteProperty(fes, "spatial_dimension", &i_dim_Nb_Nc[0], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetDimension(fe, &Nb));
    i_dim_Nb_Nc[1] = Nb;
    PetscCall(PetscFESAWsWriteProperty(fes, "dimension", &i_dim_Nb_Nc[1], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetNumComponents(fe, &Nc));
    i_dim_Nb_Nc[2] = Nc;
    PetscCall(PetscFESAWsWriteProperty(fes, "number_of_components", &i_dim_Nb_Nc[2], 1, SAWs_READ, SAWs_INT));

    PetscCall(PetscFEGetDualSpace(fe, &dsp));
    PetscCall(PetscFESAWsDirectoryPush(fes, "dual_space"));
    PetscCall(PetscFESAWsViewDualSpace(fes, dsp));
    PetscCall(PetscFESAWsDirectoryPop(fes));

    PetscCall(PetscDualSpaceGetDM(dsp, &refel));
    PetscCall(PetscFESAWsDirectoryPush(fes, "reference_element"));
    PetscCall(PetscFESAWsViewReferenceElement(fes, refel));
    PetscCall(PetscFESAWsDirectoryPop(fes));

    PetscCall(PetscFEGetBasisSpace(fe, &space));
    PetscCall(PetscFESAWsDirectoryPush(fes, "basis_space"));
    PetscCall(PetscFESAWsViewBasisSpace(fes, space));
    PetscCall(PetscFESAWsDirectoryPop(fes));
  }
  PetscCall(PetscFESAWsDirectoryPop(fes));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEView_SAWs(PetscFE fe, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (!PetscFESAWsManager) {
    PetscCall(PetscFESAWsCreate(PETSC_COMM_WORLD, &PetscFESAWsManager));
    PetscCall(PetscObjectSetName((PetscObject) PetscFESAWsManager, "PetscFESAWs"));
    PetscCall(PetscFESAWsSetBaseDirectory(PetscFESAWsManager, "FE"));
    PetscCall(PetscFESAWsView_SAWs(PetscFESAWsManager, viewer));
    PetscCall(PetscObjectRegisterDestroy((PetscObject) PetscFESAWsManager));
  }
  PetscFESAWsAddFE(PetscFESAWsManager, fe);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscClassIdRegister("PetscFE SAWs Manager",&PETSCFESAWS_CLASSID));
  for (PetscInt dim = 1; dim <= 3; dim++) {
    PetscFE   fe;
    PetscInt  Nc = 1;
    PetscBool isSimplex = PETSC_TRUE;
    PetscInt  qorder = PETSC_DETERMINE;

    PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, dim, Nc, isSimplex, 0, qorder, &fe));
    if (dim == 3) {
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fe, "threeD_"));
    }
    PetscCall(PetscFEView_SAWs(fe, PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)));
    PetscCall(PetscFEDestroy(&fe));
  }
  printf("here\n");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -saws_port 8000

TEST*/
