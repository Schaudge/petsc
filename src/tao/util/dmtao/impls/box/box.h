#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  PetscReal *lb_real, *ub_real; //Setting it as pointer, as it may be NULL, not zero, if vector is set

  Vec lb_vec, ub_vec;
} DMTao_Box;
