#include "limiters.h"

PETSC_STATIC_INLINE PetscReal Sgn(PetscReal a) { return (a<0) ? -1 : 1; }
PETSC_STATIC_INLINE PetscReal Abs(PetscReal a) { return (a<0) ? 0 : a; }
PETSC_STATIC_INLINE PetscReal Sqr(PetscReal a) { return a*a; }
//PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }
PETSC_UNUSED PETSC_STATIC_INLINE PetscReal MinAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) < PetscAbs(b)) ? a : b; }
PETSC_STATIC_INLINE PetscReal MinMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MaxMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMax(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MinMod3(PetscReal a,PetscReal b,PetscReal c) {return (a*b<0 || a*c<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscMin(PetscAbs(b),PetscAbs(c))); }

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