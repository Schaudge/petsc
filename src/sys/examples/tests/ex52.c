
static char help[] = "Demonstrates PetscPointerGetType().\n\n";

#include <petsc/private/petscsystypes.h>

typedef struct {
  _p_PetscObject hdr;
} UserContext_2;


const char *PetscHeaderTypeNames[] = {
  "null pointer",
  "invalid pointer: members are not accessible, implies corrupt data, or object not inherited from PetscObject",
  "PetscHeader has been freed",
  "PetscHeader invalid: likely PetscHeaderCreate() has not been called",
  "PetscHeader valid",
  "PETSC_HEADER_",
  0
};

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  char            c = 'c';
  short           s = 0;
  PetscInt        pi = 4;
  PetscClassId    pci = 335;
  PetscInt        *pn = NULL;
  PetscContainer  container = NULL;
  SNES            snes;
  char            *abc;
  PetscObject     po;
  UserContext_2   *ctx2;
  PetscClassId    MY_CLASSID;
  PetscHeaderType type;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscMalloc1(1,&abc);CHKERRQ(ierr);
  abc[0] = 'c';
  ierr = PetscPointerGetPetscHeaderType(abc,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"char*\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(&c,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"char\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(&s,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"short\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(&pi,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"PetscInt\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(pn,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"NULL PetscInt*\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(&pci,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"PetscClassId\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscPointerGetPetscHeaderType(container,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"NULL PetscContainer\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscContainerCreate(PETSC_COMM_WORLD,&container);CHKERRQ(ierr);
  ierr = PetscPointerGetPetscHeaderType(container,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"PetscContainer\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = PetscPointerGetPetscHeaderType(snes,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"SNES\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscNew(&po);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("User Application Context",&MY_CLASSID);CHKERRQ(ierr);
  ierr = PetscHeaderCreate(po,MY_CLASSID,"AppCtx","User Application Context","AppCtx",PETSC_COMM_SELF,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPointerGetPetscHeaderType(po,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"user-PetscObject\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);
  
  ierr = PetscCalloc1(1,&ctx2);CHKERRQ(ierr);
  ierr = PetscStrallocpy("UserContext_2",&ctx2->hdr.class_name);CHKERRQ(ierr);
  ierr = PetscPointerGetPetscHeaderType(ctx2,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"context  \"UserContext_2\", type = %s\n",PetscHeaderTypeNames[(int)type]);CHKERRQ(ierr);

  ierr = PetscFree(ctx2->hdr.class_name);CHKERRQ(ierr);
  ierr = PetscFree(ctx2);CHKERRQ(ierr);
  ierr = PetscFree(abc);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST
 
   test:
 
TEST*/
