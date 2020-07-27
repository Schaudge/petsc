#include "limiters.h"

PETSC_STATIC_INLINE PetscReal MinMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : PetscSign(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MaxMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : PetscSign(a)*PetscMax(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MinMod3(PetscReal a,PetscReal b,PetscReal c) {return (a*b<0 || a*c<0) ? 0 : PetscSign(a)*PetscMin(PetscAbs(a),PetscMin(PetscAbs(b),PetscAbs(c))); }

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