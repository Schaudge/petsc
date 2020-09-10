#include "limiters.h"

PETSC_STATIC_INLINE PetscReal MinMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : PetscSign(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }

void Limit_Upwind_Uni(const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt,PetscInt m)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = 0;
}
void Limit_LaxWendroff_Uni(const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt,PetscInt m)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = jR[i];
}
void Limit_BeamWarming_Uni(const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt,PetscInt m)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = jL[i];
}
void Limit_Fromm_Uni(const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt,PetscInt m)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = 0.5*(jL[i]+jR[i]);
}
void Limit_Minmod_Uni(const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt,PetscInt m)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = MinMod2(jL[i],jR[i]);
}