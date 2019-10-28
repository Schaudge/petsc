#include <petsc/private/viewerconduitimpl.h>
#include <petsc/private/viewerimpl.h>
#if !defined(PETSC_HAVE_MPIUNI)
#  include <conduit/conduit_relay_mpi_io_blueprint.hpp>
#endif
#include <conduit/conduit_relay.hpp>

static PetscErrorCode PetscViewerDestroy_Conduit(PetscViewer viewer)
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cd->filename);CHKERRQ(ierr);
  ierr = PetscFree(cd);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFlush_Conduit(PetscViewer viewer)
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;

  PetscFunctionBegin;
  if (cd->mesh) {
#if defined(PETSC_HAVE_MPIUNI)
    conduit::relay::io_blueprint::save(*cd->mesh, cd->filename);
#else
    conduit::relay::mpi::io_blueprint::save(*cd->mesh, cd->filename, PetscObjectComm((PetscObject)viewer));
    //conduit::relay::mpi::io::save(*cd->mesh, cd->filename, PetscObjectComm((PetscObject)viewer));
#endif
  }
  delete cd->mesh;
  cd->mesh = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetName_Conduit(PetscViewer viewer,const char name[])
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;
  PetscErrorCode  ierr;
  char *filename;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(name, &filename);CHKERRQ(ierr);
  cd->filename = filename;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_Conduit(PetscViewer viewer,const char **name)
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;
  PetscFunctionBegin;
  *name = cd->filename;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetMode_Conduit(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;

  PetscFunctionBegin;
  cd->filemode = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetMode_Conduit(PetscViewer viewer,PetscFileMode *type)
{
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;

  PetscFunctionBegin;
  *type = cd->filemode;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERCONDUIT - A viewer that exposes data via Conduit

   Notes: Conduit is an in-memory interface that supports in-situ visualization and analysis as well as writing the data to files.

   The present implementation only supports writing files.

.seealso:  PetscViewerConduitOpen(), PetscViewerCreate(), VecView(), DMView(), PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/
PETSC_INTERN PetscErrorCode PetscViewerCreate_Conduit(PetscViewer v)
{
  PetscViewer_Conduit *cd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,&cd);CHKERRQ(ierr);

  v->data         = (void*)cd;
  v->ops->destroy = PetscViewerDestroy_Conduit;
  v->ops->flush   = PetscViewerFlush_Conduit;
  cd->filemode    = (PetscFileMode) -1;
  cd->filename    = 0;

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_Conduit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_Conduit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_Conduit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_Conduit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerConduitOpen - Opens a file for Conduit output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input (not currently supported)
$    FILE_MODE_APPEND - open existing file for binary output (not currently supported)

   Output Parameter:
.  viewer - PetscViewer for Conduit input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().


.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(),
          PetscFileMode, PetscViewer
@*/
PetscErrorCode PetscViewerConduitOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSCVIEWERCONDUIT);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
