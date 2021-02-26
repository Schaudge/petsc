static char help[] = "Example for PetscOptionsInsertFileYAML\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  char            filename[PETSC_MAX_PATH_LEN];
  PetscBool       flg;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);
  if (flg) {
    ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,NULL,filename,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(NULL,NULL,"-yaml",filename,sizeof(filename),&flg);
  if (flg) {
    PetscBool monitor = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
    if (monitor) {
      ierr = PetscOptionsMonitorSet(PetscOptionsMonitorDefault,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsClear(NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,NULL,filename,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PetscOptionsView(NULL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscOptionsClear(NULL);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   testset:
     requires: yaml
     filter:  egrep -v "(malloc_dump|malloc_test|saws_port_auto_select|display|check_pointer_intensity|error_output_stdout|nox)"
     localrunfiles: petsc.yml

     test:
        suffix: 1
        args: -f petsc.yml -options_left 0

     test:
        suffix: 2
        args: -options_file_yaml petsc.yml -options_left 0


   testset:
     nsize: {{1 2}}
     requires: yaml

     test:
        suffix: 3_empty
        args: -yaml ex47-empty.yaml
        localrunfiles: ex47-empty.yaml

     test:
        suffix: 3_merge
        args: -yaml ex47-merge.yaml -monitor
        localrunfiles: ex47-merge.yaml

     test:
        suffix: 3_options
        args: -yaml ex47-options.yaml
        localrunfiles: ex47-options.yaml

     test:
        suffix: 3_include
        args: -yaml ex47-include.yaml
        localrunfiles: ex47-include.yaml ex47-empty.yaml ex47-options.yaml


TEST*/
