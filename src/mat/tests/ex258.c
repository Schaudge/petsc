static char help[] = "Test MatProduct_AtB --with-scalar-type=complex. \n\
Modified from the code contributed by Pierre Jolivet \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt    n = 2,convert;
  Mat         array[1],B,Bdense,Conjugate;
  PetscBool   conjugate = PETSC_FALSE,equal;
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD,n,n,n,n,-1.0,array));
  PetscCall(MatConvert(array[0],MATDENSE,MAT_INPLACE_MATRIX,array));
  PetscCall(MatSetRandom(array[0],NULL));
  PetscCall(MatViewFromOptions(array[0],NULL,"-A_view"));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-conjugate",&conjugate,NULL));

  for (convert = 0; convert<2; convert++) {
    /* convert dense matrix array[0] to aij format */
    if (convert) {
      PetscCall(MatConvert(array[0],MATAIJ,MAT_INPLACE_MATRIX,array));
    }

    /* compute B = array[0]^T * array[0] or  B = array[0]^H * array[0] */
    PetscCall(MatProductCreate(array[0],array[0],NULL,&B));
    PetscCall(MatProductSetType(B,MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));
    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(MatDuplicate(array[0], MAT_COPY_VALUES, &Conjugate));
      if (conjugate) PetscCall(MatConjugate(Conjugate));

      PetscCall(MatProductReplaceMats(Conjugate,array[0],NULL,B));
    }
    PetscCall(MatProductNumeric(B));
    PetscCall(MatViewFromOptions(B,NULL,"-product_view"));

    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(MatDestroy(&Conjugate));
    }
    if (!convert) {
      Bdense = B; B = NULL;
    }
  }

  /* Compare Bdense and B */
  PetscCall(MatMultEqual(Bdense,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Bdense != B");

  PetscCall(MatDestroy(&Bdense));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(array));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -conjugate false
      output_file: output/ex258_1.out

   test:
      suffix: 2
      args: -conjugate true
      output_file: output/ex258_1.out

TEST*/
