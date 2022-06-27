const char help[] = "Visualize PetscFE finite elements with plotly.js";

#include <petsc.h>
#include <petscviewersaws.h>
#include <petsc/private/petscimpl.h>
#include <SAWs.h>

typedef struct _p_PetscFEPlotly *PetscFEPlotly;

struct _PetscFEPlotlyOps {};

struct _p_PetscFEPlotly
{
  PETSCHEADER(struct _PetscFEPlotlyOps);
  int number_of_elements;
};

PetscClassId PETSCFEPLOTLY_CLASSID;

static PetscErrorCode PetscFEPlotlyDestroy(PetscFEPlotly *fep)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsBlock((PetscObject) *fep));
  PetscCall(PetscObjectSAWsViewOff((PetscObject) *fep));
  PetscCall(PetscHeaderDestroy(fep));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyView_SAWS(PetscFEPlotly fep, PetscViewer viewer)
{
  const char *name;
  char number_of_elements_str[BUFSIZ];

  PetscFunctionBegin;
  PetscCall(PetscObjectViewSAWs((PetscObject) fep, viewer));
  PetscCall(PetscObjectGetName((PetscObject)fep, &name));
  PetscCall(PetscSNPrintf(number_of_elements_str, BUFSIZ-1, "/PETSc/Objects/%s/number_of_elements", name));
  PetscStackCallSAWs(SAWs_Register, (number_of_elements_str, &(fep->number_of_elements), 1, SAWs_READ, SAWs_INT));
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)fep));
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)fep));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEPlotlyCreate(MPI_Comm comm, PetscFEPlotly *fep)
{
  PetscFunctionBegin;
  PetscCall(PetscHeaderCreate(*fep, PETSCFEPLOTLY_CLASSID, "PetscFEPlotly", "PetscFE Plotly Manager", "", comm, PetscFEPlotlyDestroy, PetscFEPlotlyView_SAWS));
  (*fep)->number_of_elements = 0;
  PetscCall(PetscObjectSAWsSetBlock((PetscObject) *fep, PETSC_TRUE));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscFEPlotly fep;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscClassIdRegister("PetscFE Plotly Manager",&PETSCFEPLOTLY_CLASSID));
  PetscCall(PetscFEPlotlyCreate(PETSC_COMM_WORLD, &fep));
  PetscCall(PetscObjectSetName((PetscObject) fep, "PetscFEPlotly"));
  PetscCall(PetscFEPlotlyView_SAWS(fep, PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)));
  PetscCall(PetscFEPlotlyDestroy(&fep));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
