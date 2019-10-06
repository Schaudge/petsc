static char help[]= "Tests ISLocalToGlobalMappingGetInfo() for bs > 1.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  ISLocalToGlobalMapping ltog;
  PetscInt               *p,*ns,**ids;
  PetscInt               i,j,n,np,bs = 1,test = 0;
  PetscViewer            viewer;
  PetscMPIInt            rank,size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  switch (test) {
  case 1: /* quads */
    if (size > 1) {
      if (size == 4) {
        if (rank == 0) {
          PetscInt id[4] = {0,1,2,3};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 1) {
          PetscInt id[4] = {2,3,6,7};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 2) {
          PetscInt id[4] = {1,4,3,5};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 3) {
          PetscInt id[8] = {3,5,7,8};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        }
      } else {
        if (rank == 0) {
          PetscInt id[8] = {0,1,2,3,1,4,3,5};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,8,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == size-1) {
          PetscInt id[8] = {2,3,6,7,3,5,7,8};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,8,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else {
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,0,NULL,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        }
      }
    } else {
      PetscInt id[16] = {0,1,2,3,1,4,3,5,2,3,6,7,3,5,7,8};
      ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,16,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    }
    break;
  case 2: /* mix quads and tets with holes */
    if (size > 1) {
      if (size == 4) {
        if (rank == 0) {
          PetscInt id[3] = {1,2,3};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,3,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 1) {
          PetscInt id[4] = {1,4,5,3};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 2) {
          PetscInt id[3] = {3,6,2};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,3,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == 3) {
          PetscInt id[3] = {3,5,8};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,3,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        }
      } else {
        if (rank == 0) {
          PetscInt id[9] = {1,2,3,3,5,8,3,6,2};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,9,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else if (rank == size-1) {
          PetscInt id[4] = {5,3,1,4};
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,4,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        } else {
          ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,0,NULL,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
        }
      }
    } else {
      PetscInt id[13] = {1,2,3,1,4,5,3,6,3,2,5,3,8};
      ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,13,id,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    }
    break;
  default:
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,bs,0,NULL,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    break;
  }
  ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ltog),&viewer);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingView(ltog,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"GETINFO OUTPUT\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetInfo(ltog,&np,&p,&ns,&ids);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local NP %D\n",rank,np);CHKERRQ(ierr);
  for (i=0;i<np;i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   procs[%D] = %D, shared %D\n",rank,i,p[i],ns[i]);CHKERRQ(ierr);
    for (j=0;j<ns[i];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]     ids[%D] = %D\n",rank,j,ids[i][j]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreInfo(ltog,&np,&p,&ns,&ids);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"GETNODEINFO OUTPUT\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetNodeInfo(ltog,&n,&ns,&ids);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local N %D\n",rank,n);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   sharedby[%D] = %D\n",rank,i,ns[i]);CHKERRQ(ierr);
    for (j=0;j<ns[i];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]     ids[%D] = %D\n",rank,j,ids[i][j]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreNodeInfo(ltog,&n,&ns,&ids);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
     suffix: ltog_info
     nsize: {{1 2 3 4 5}separate output}
     args: -bs {{1 3}separate output} -test {{0 1 2}separate output}


TEST*/
