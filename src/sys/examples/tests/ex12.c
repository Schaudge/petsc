#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex12.c,v 1.5 1999/05/04 20:29:49 balay Exp balay $";
#endif

static char help[] = "Tests timing PetscSortInt().\n\n";

#include "petsc.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         ierr,i,n = 1000,*values,event,flag;
  PetscRandom rand;
  Scalar      value;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flag);CHKERRA(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRA(ierr);

  values = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(values);
  for ( i=0; i<n; i++ ) {
    PetscRandomGetValue(rand,&value);
    values[i] = (int) (n*PetscReal(value) + 2.0);
  }
  PetscSortInt(n,values);

  PLogEventRegister(&event,"Sort",PETSC_NULL);
  PLogEventBegin(event,0,0,0,0);
  values = (int *) PetscMalloc( n*sizeof(int) );CHKPTRA(values);
  for ( i=0; i<n; i++ ) {
    PetscRandomGetValue(rand,&value);
    values[i] = (int) (n*PetscReal(value) + 2.0);
  }
  PetscSortInt(n,values);
  PLogEventEnd(event,0,0,0,0);

  for ( i=1; i<n; i++ ) {
    if (values[i] < values[i-1]) SETERRA(1,1,"Values not sorted");
  }
  ierr = PetscFree(values);CHKERRA(ierr);
  PetscRandomDestroy(rand);

  PetscFinalize();
  return 0;
}
 
