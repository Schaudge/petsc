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
  case SAWs_CHAR:
    mpi_dtype = MPI_CHAR;
    break;
  case SAWs_INT:
    mpi_dtype = MPI_INT;
    break;
  case SAWs_DOUBLE:
    mpi_dtype = MPI_DOUBLE;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscFESAWs does not create arrays of this type");
  }
  PetscCallMPI(MPI_Type_size(mpi_dtype,&dsize));
  PetscCall(PetscNewLog(fes, &link));
  link->next = fes->arrays;
  fes->arrays = link;
  PetscCall(PetscMalloc(count*dsize, &(link->mem)));
  *((void **) data_pointer) = link->mem;
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
  PetscInt v_start, v_end;
  DMPolytopeType ptype;
  size_t type_name_len;
  char *type_name;
  int *i_array;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepthStratum(refel, 0, &v_start, &v_end));
  PetscCall(PetscFESAWsCreateArray(fes, SAWs_INT, 1, &i_array));
  i_array[0] = v_end - v_start;
  PetscCall(PetscFESAWsWriteProperty(fes, "number_of_vertices", &i_array[0], 1, SAWs_READ, SAWs_INT));
  PetscCall(DMPlexGetCellType(refel, 0, &ptype));
  PetscCall(PetscStrlen(DMPolytopeTypes[ptype], &type_name_len));
  PetscCall(PetscFESAWsWriteProperty(fes, "polytope", DMPolytopeTypes[ptype], type_name_len, SAWs_READ, SAWs_CHAR));
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
    PetscDualSpace dsp;
    DM refel;
    int *i_dim_Nb_Nc;
    const char *name;
    const char *prefix;
    char *aname, *aprefix;
    size_t name_len, prefix_len;

    PetscCall(PetscObjectName((PetscObject) fe));
    PetscCall(PetscObjectGetName((PetscObject) fe, &name));
    PetscCall(PetscStrlen(name, &name_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_CHAR, name_len+1, &aname));
    PetscCall(PetscArraycpy(aname, name, name_len+1));
    PetscCall(PetscFESAWsWriteProperty(fes, "name", aname, name_len, SAWs_READ, SAWs_CHAR));

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject) fe, &prefix));
    PetscCall(PetscStrlen(prefix, &prefix_len));
    PetscCall(PetscFESAWsCreateArray(fes, SAWs_CHAR, prefix_len+1, &aprefix));
    aprefix[prefix_len] = '\0';
    PetscCall(PetscArraycpy(aprefix, prefix, prefix_len));
    PetscCall(PetscFESAWsWriteProperty(fes, "options_prefix", aprefix, prefix_len, SAWs_READ, SAWs_CHAR));

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
    PetscCall(PetscDualSpaceGetDM(dsp, &refel));
    PetscCall(PetscFESAWsDirectoryPush(fes, "reference_element"));
    PetscCall(PetscFESAWsViewReferenceElement(fes, refel));
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

    PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, dim, Nc, isSimplex, dim == 3 ? "threeD_" : NULL, qorder, &fe));
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
