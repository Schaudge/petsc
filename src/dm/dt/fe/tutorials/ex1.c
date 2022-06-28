const char help[] = "Visualize PetscFE finite elements with plotly.js";

#include <petsc.h>
#include <petscviewersaws.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmimpl.h>
#include <SAWs.h>

static PetscErrorCode PetscObjectSAWsWriteProperty(PetscObject object, const char *property, void *data, int count, SAWs_Memory_type mtype, SAWs_Data_type dtype)
{
  const char *name;
  char buf[BUFSIZ];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName(object, &name));
  PetscCall(PetscSNPrintf(buf, BUFSIZ-1, "/PETSc/Objects/%s/%s", name, property));
  PetscStackCallSAWs(SAWs_Register,(buf, data, count, mtype, dtype));
  PetscFunctionReturn(0);
}

typedef struct _n_PetscFEPlotlyLink *PetscFEPlotlyLink;

struct _n_PetscFEPlotlyLink
{
  PetscFE fe;
  PetscFEPlotlyLink next;
};

// TODO: if PetscInt != int
static PetscErrorCode ReferenceElementView_SAWs(DM refel, PetscViewer viewer)
{
  Vec coord_vec;
  PetscInt vec_size;
  const PetscScalar *coords;

  PetscFunctionBegin;
  if (((PetscObject) refel)->amsmem) PetscFunctionReturn(0);
  PetscCall(PetscObjectName((PetscObject) refel));
  PetscCall(PetscObjectViewSAWs((PetscObject)refel, viewer));
  PetscCall(PetscObjectSAWsWriteProperty((PetscObject)refel, "coordinate_dimension", &(refel->dimEmbed), 1, SAWs_READ, SAWs_INT));
  PetscCall(DMGetCoordinatesLocalNoncollective(refel, &coord_vec));
  PetscCall(VecGetSize(coord_vec, &vec_size));
  PetscCall(VecGetArrayRead(coord_vec, &coords));
  PetscCall(PetscObjectSAWsWriteProperty((PetscObject)refel, "coordinates", (void *) coords, vec_size, SAWs_READ, SAWs_DOUBLE));
  PetscCall(VecRestoreArrayRead(coord_vec, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyLinkCreate(PetscFE fe, PetscViewer viewer, PetscFEPlotlyLink *fep_link)
{
  PetscDualSpace    dsp;
  DM                dm;
  PetscFEPlotlyLink link;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDualSpace(fe, &dsp));
  PetscCall(PetscDualSpaceGetDM(dsp, &dm));
  PetscCall(ReferenceElementView_SAWs(dm, viewer));
  PetscCall(PetscNew(&link));
  PetscCall(PetscObjectReference((PetscObject) fe));
  link->fe = fe;
  *fep_link = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyLinkDestroy(PetscFEPlotlyLink *fep_link)
{
  PetscFunctionBegin;
  PetscCall(PetscFEDestroy(&(*fep_link)->fe));
  PetscCall(PetscFree(*fep_link));
  *fep_link = NULL;
  PetscFunctionReturn(0);
}

typedef struct _p_PetscFEPlotly *PetscFEPlotly;

struct _PetscFEPlotlyOps {};

struct _p_PetscFEPlotly
{
  PETSCHEADER(struct _PetscFEPlotlyOps);
  int number_of_elements;
  PetscFEPlotlyLink head;
};

PetscClassId PETSCFEPLOTLY_CLASSID;

static PetscErrorCode PetscFEPlotlyDestroy(PetscFEPlotly *fep)
{
  PetscFEPlotlyLink head;

  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsBlock((PetscObject) *fep));
  head = (*fep)->head;
  while (head) {
    PetscFEPlotlyLink next = head->next;

    PetscCall(PetscFEPlotlyLinkDestroy(&head));
    head = next;
  }
  PetscCall(PetscObjectSAWsViewOff((PetscObject) *fep));
  PetscCall(PetscHeaderDestroy(fep));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyView_SAWs(PetscFEPlotly fep, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectViewSAWs((PetscObject) fep, viewer));
  PetscCall(PetscObjectName((PetscObject)fep));
  PetscCall(PetscObjectSAWsWriteProperty((PetscObject)fep, "number_of_elements", &(fep->number_of_elements), 1, SAWs_READ, SAWs_INT ));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyCreate(MPI_Comm comm, PetscFEPlotly *fep)
{
  PetscFunctionBegin;
  PetscCall(PetscHeaderCreate(*fep, PETSCFEPLOTLY_CLASSID, "PetscFEPlotly", "PetscFE Plotly Manager", "", comm, PetscFEPlotlyDestroy, PetscFEPlotlyView_SAWs));
  (*fep)->number_of_elements = 0;
  PetscCall(PetscObjectSAWsSetBlock((PetscObject) *fep, PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscFEPlotly PetscFEPlotlyManager = NULL;

static PetscErrorCode PetscFEPlotlyAddLink(PetscFEPlotly fep, PetscFEPlotlyLink link)
{
  char property[BUFSIZ];
  int old_number_of_elements;
  PetscDualSpace dsp;
  DM refel;

  PetscFunctionBegin;
  link->next = fep->head;
  fep->head = link;
  old_number_of_elements = PetscFEPlotlyManager->number_of_elements;
  PetscCall(PetscFEGetDualSpace(link->fe, &dsp));
  PetscCall(PetscDualSpaceGetDM(dsp, &refel));
  PetscCall(PetscSNPrintf(property, BUFSIZ-1, "%d/reference_element", old_number_of_elements));
  PetscCall(PetscObjectSAWsWriteProperty((PetscObject) fep, property, (void *) &(((PetscObject) refel)->name), 1, SAWs_READ, SAWs_STRING));
  PetscFEPlotlyManager->number_of_elements = old_number_of_elements + 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEView_SAWs_Plotly(PetscFE fe, PetscViewer viewer)
{
  PetscFEPlotlyLink link;

  PetscFunctionBegin;
  PetscCall(PetscFEPlotlyLinkCreate(fe, viewer, &link));
  if (!PetscFEPlotlyManager) {
    PetscCall(PetscFEPlotlyCreate(PETSC_COMM_WORLD, &PetscFEPlotlyManager));
    PetscCall(PetscObjectSetName((PetscObject) PetscFEPlotlyManager, "PetscFEPlotly"));
    PetscCall(PetscFEPlotlyView_SAWs(PetscFEPlotlyManager, viewer));
    PetscCall(PetscObjectRegisterDestroy((PetscObject) PetscFEPlotlyManager));
  }
  PetscCall(PetscFEPlotlyAddLink(PetscFEPlotlyManager, link));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscClassIdRegister("PetscFE Plotly Manager",&PETSCFEPLOTLY_CLASSID));
  for (PetscInt dim = 1; dim <= 3; dim++) {
    PetscFE   fe;
    PetscInt  Nc = 1;
    PetscBool isSimplex = PETSC_TRUE;
    PetscInt  qorder = PETSC_DETERMINE;

    PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, dim, Nc, isSimplex, NULL, qorder, &fe));
    PetscCall(PetscFEView_SAWs_Plotly(fe, PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)));
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
