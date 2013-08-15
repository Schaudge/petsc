#if !defined(_PETSCFEIMPL_H)
#define _PETSCFEIMPL_H

#include <petscfe.h>
#include <petsc-private/petscimpl.h>

typedef struct _PetscSpaceOps *PetscSpaceOps;
struct _PetscSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscSpace);
  PetscErrorCode (*setup)(PetscSpace);
  PetscErrorCode (*view)(PetscSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscSpace);

  PetscErrorCode (*getdimension)(PetscSpace,PetscInt*);
  PetscErrorCode (*evaluate)(PetscSpace,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
};

struct _p_PetscSpace {
  PETSCHEADER(struct _PetscSpaceOps);
  void    *data;  /* Implementation object */
  PetscInt order; /* The approximation order of the space */
  DM       dm;    /* Shell to use for temp allocation */
};

typedef struct {
  PetscInt   numVariables; /* The number of variables in the space, e.g. x and y */
  PetscBool  symmetric;    /* Use only symmetric polynomials */
  PetscInt  *degrees;      /* Degrees of single variable which we need to compute */
} PetscSpace_Poly;

typedef struct _PetscDualSpaceOps *PetscDualSpaceOps;
struct _PetscDualSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscDualSpace);
  PetscErrorCode (*setup)(PetscDualSpace);
  PetscErrorCode (*view)(PetscDualSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscDualSpace);

  PetscErrorCode (*getdimension)(PetscDualSpace,PetscInt*);
  PetscErrorCode (*getnumdof)(PetscDualSpace,const PetscInt**);
};

struct _p_PetscDualSpace {
  PETSCHEADER(struct _PetscDualSpaceOps);
  void            *data;       /* Implementation object */
  DM               dm;         /* The integration region K */
  PetscInt         order;      /* The approximation order of the space */
  PetscQuadrature *functional; /* The basis of functionals for this space */
};

typedef struct {
  PetscInt  cellType;
  PetscInt *numDof;
} PetscDualSpace_Lag;

typedef struct _PetscFEOps *PetscFEOps;
struct _PetscFEOps {
  PetscErrorCode (*setfromoptions)(PetscFE);
  PetscErrorCode (*setup)(PetscFE);
  PetscErrorCode (*view)(PetscFE,PetscViewer);
  PetscErrorCode (*destroy)(PetscFE);
  /* Element integration */
  PetscErrorCode (*integrateresidual)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      PetscScalar[]);
  PetscErrorCode (*integratebdresidual)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        PetscScalar[]);
  PetscErrorCode (*integratejacobianaction)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[], const PetscScalar[],
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            PetscScalar[]);
  PetscErrorCode (*integratejacobian)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscInt, PetscCellGeometry, const PetscScalar[],
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      PetscScalar[]);
};

struct _p_PetscFE {
  PETSCHEADER(struct _PetscFEOps);
  void           *data;          /* Implementation object */
  PetscSpace      basisSpace;    /* The basis space P */
  PetscDualSpace  dualSpace;     /* The dual space P' */
  PetscInt        numComponents; /* The number of field components */
  PetscQuadrature quadrature;    /* Suitable quadrature on K */
  PetscInt       *numDof;        /* The number of dof on mesh points of each depth */
  PetscReal      *B, *D, *H;     /* Tabulation of basis and derivatives at quadrature points */
};

typedef struct {
  PetscInt cellType;
} PetscFE_Basic;

#endif
