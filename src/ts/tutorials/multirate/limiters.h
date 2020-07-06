
/* Collection of uniform mesh limiters*/
void Limit_Upwind_Uni(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
void Limit_LaxWendroff_Uni(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
void Limit_BeamWarming_Uni(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
void Limit_Fromm_Uni(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
void Limit_Minmod_Uni(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);