
#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "sles.h"

struct _p_SLES {
  PETSCHEADER(int)
  int        setupcalled;
  PetscTruth dscale;      /* diagonal scale system; used with SLESSetDiagonalScale() */
  Vec        diagonal;    /* 1/sqrt(diag of matrix) */
  PC         pc;
  KSP        ksp;
};

#endif
