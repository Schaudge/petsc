static const char help[] = "Test PetscSF handling of MPIU_2INT, MPIU_REAL, MPIU_COMPLEX\n\n";
/*
  Users can pass any MPI datatype to PetscSF, which will decode the datatype and build suitable ops for the type.
  For an input type, the SF will first check its type cache to see if there is a match. Care must be taken in the
  matching test. For example, with 64-bit indices, PETSc builds MPIU_2INT from two MPI_INTs; with quad-precision,
  PETSc builds MPIU_REAL, MPIU_COMPLEX from two and four MPI_DOUBLEs respectively. If in the cache, there is an
  MPIU_COMPLEX, and users call SF routines with their own type created from MPI_Type_contiguous(4,MPI_DOUBLE,&newtype),
  then we must not mess up MPIU_COMPLEX and newtype, since they support very different ops such as sum, mult. Wrong
  matching can result in silent errors.
*/
#include <petscsf.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscSF        sf;
  PetscInt       i,nroots=1,nleaves=2;
  PetscSFNode    iremote[2];
  PetscMPIInt    size;
  MPI_Datatype   my2int,my2real,my4double;
  PetscInt       ileafdata[4] = {1,2,3,4},irootdata[2] = {3,6},itmp[2];
  PetscReal      rleafdata[4] = {1,2,3,4},rrootdata[2] = {5,6};
  double         dleafdata[8] = {1,2,3,4,5,6,7,8},drootdata[4] = {9,10,11,12},dtmp[4];
#if defined(PETSC_HAVE_COMPLEX)
  PetscComplex   cleafdata[2],crootdata[1];
#endif
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* A very simple SF: one root, two leaves, uniprocessor */
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  for (i=0; i<nleaves; i++) {
    iremote[i].rank  = 0;
    iremote[i].index = 0;
  }
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);

  /* Test MPIU_2INT. With 64-bit indices, MPIU_2INT is a petsc-builtin datatype made of two MPIU_INTs. */

  /* Add leaves {1,2}, {3,4} to root {3,6} with MPI_MAXLOC, which requires <unit> to be MPIU_2INT or MPI_2INT */
  ierr  = PetscSFReduceBegin(sf,MPIU_2INT/*unit*/,ileafdata,irootdata,MPI_MAXLOC);CHKERRQ(ierr);
  ierr  = PetscSFReduceEnd(sf,MPIU_2INT,ileafdata,irootdata,MPI_MAXLOC);CHKERRQ(ierr);
  itmp[0] = irootdata[0];
  itmp[1] = irootdata[1];
  if (itmp[0] != 3 || itmp[1] != 4) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong rootdata with MPIU_2INT: (%D,%D)",itmp[0],itmp[1]);

  /* Add leaves {1,2}, {3,4} to root {3,4} with MPI_SUM. MPI requires the datatype used in MPI_SUM to be a single integer (or FP etc).
     PetscSF allows <unit> to be multiple such types and performs element-wise reduction on <unit>
  */
  ierr = MPI_Type_contiguous(2,MPIU_INT,&my2int);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&my2int);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,my2int,ileafdata,irootdata,MPI_SUM);CHKERRQ(ierr); /* Test if SF can differentiate my2int and MPIU_2INT */
  ierr = PetscSFReduceEnd(sf,my2int,ileafdata,irootdata,MPI_SUM);CHKERRQ(ierr);
  itmp[0] = irootdata[0];
  itmp[1] = irootdata[1];
  if (itmp[0] != 7 || itmp[1] != 10) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong rootdata with my2int: (%D,%D)",itmp[0],itmp[1]);

  /* Test MPIU_COMPLEX. With quad-precision, MPIU_COMPLEX is a petsc-builtin type made of four MPI_DOUBLEs. */
#if defined(PETSC_HAVE_COMPLEX)
  /* rootdata = (1 + 2i)(3 + 4i)(5 + 6i) */
  cleafdata[0] = PetscCMPLX(1,2);
  cleafdata[1] = PetscCMPLX(3,4);
  crootdata[0] = PetscCMPLX(5,6);
  ierr  = PetscSFReduceBegin(sf,MPIU_COMPLEX,cleafdata,crootdata,MPI_PROD);CHKERRQ(ierr);
  ierr  = PetscSFReduceEnd(sf,MPIU_COMPLEX,cleafdata,crootdata,MPI_PROD);CHKERRQ(ierr);
  dtmp[0] = (double)PetscRealPartComplex(crootdata[0]);
  dtmp[1] = (double)PetscImaginaryPartComplex(crootdata[0]);
  if (dtmp[0] != -85 || dtmp[1] != 20) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong rootdata with MPIU_COMPLEX: (%g + %gi)",dtmp[0],dtmp[1]);
#endif

  /* rootdata = {1*3*5, 2*4*6} */
  ierr = MPI_Type_contiguous(2,MPIU_REAL,&my2real);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&my2real);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,my2real,rleafdata,rrootdata,MPI_PROD);CHKERRQ(ierr); /* Test if SF can differentiate my2real and MPIU_COMPLEX */
  ierr = PetscSFReduceEnd(sf,my2real,rleafdata,rrootdata,MPI_PROD);CHKERRQ(ierr);
  dtmp[0] = (double)rrootdata[0];
  dtmp[1] = (double)rrootdata[1];
  if (dtmp[0] != 15 || dtmp[1] != 48) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong rootdata with my2real: {%g, %g}",dtmp[0],dtmp[1]);

  /* rootdata = {1*5*9, 2*6*10, 3*7*11,4*8*12} */
  ierr = MPI_Type_contiguous(4,MPI_DOUBLE,&my4double);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&my4double);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,my4double,dleafdata,drootdata,MPI_PROD);CHKERRQ(ierr); /* Test if SF can differentiate my4double and MPIU_COMPLEX */
  ierr = PetscSFReduceEnd(sf,my4double,dleafdata,drootdata,MPI_PROD);CHKERRQ(ierr);
  for (i=0; i<4; i++) dtmp[i] = drootdata[i];
  if (dtmp[0] != 45 || dtmp[1] != 120 || dtmp[2] != 231 || dtmp[3] != 384) SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Wrong rootdata with my4double: {%g, %g, %g, %g}",dtmp[0],dtmp[1],dtmp[2],dtmp[3]);

  ierr = MPI_Type_free(&my2int);CHKERRQ(ierr);
  ierr = MPI_Type_free(&my2real);CHKERRQ(ierr);
  ierr = MPI_Type_free(&my4double);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
TEST*/
