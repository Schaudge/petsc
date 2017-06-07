#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petscfe.h>
#include <petscdmplex.h>
#include <petscds.h>

typedef struct _n_DMField_DS
{
  PetscInt    fieldNum;
  Vec         vec;
  PetscObject disc;
  PetscBool   multifieldVec;
}
DMField_DS;

static PetscErrorCode DMFieldDestroy_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  ierr = VecDestroy(&dsfield->vec);CHKERRQ(ierr);
  ierr = PetscObjectDereference(dsfield->disc);CHKERRQ(ierr);
  dsfield->disc = NULL;
  ierr = PetscFree(dsfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DS(DMField field,PetscViewer viewer)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;
  PetscBool      iascii;
  PetscDS        ds;
  PetscObject    disc;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm   = field->dm;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  disc = dsfield->disc;
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "PetscDS field %D\n",dsfield->fieldNum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscObjectView(disc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (dsfield->multifieldVec) {
    SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"View of subfield not implemented yet");
  } else {
    ierr = VecView(dsfield->vec,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluate_DS(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented yet");
  PetscFunctionReturn(0);
}

#define DMFieldDSdot(y,A,b,m,n,c,cast)                                           \
  do {                                                                           \
    PetscInt _i, _j, _k;                                                         \
    for (_i = 0; _i < (m); _i++) {                                               \
      for (_k = 0; _k < (c); _k++) {                                             \
        (y)[_i * (c) + _k] = 0.;                                                 \
      }                                                                          \
      for (_j = 0; _j < (n); _j++) {                                             \
        for (_k = 0; _k < (c); _k++) {                                           \
          (y)[_i * (c) + _k] += (A)[(_i * (n) + _j) * (c) + _k] * cast((b)[_j]); \
        }                                                                        \
      }                                                                          \
    }                                                                            \
  } while (0)

static PetscErrorCode DMFieldEvaluateFE_DS(DMField field, IS cellIS, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *) field->data;
  DM              dm;
  PetscObject     disc;
  PetscClassId    classid;
  PetscInt        nq, nc, dim, numCells;
  PetscSection    section;
  const PetscReal *qpoints;
  PetscBool       isStride;
  const PetscInt  *cells = NULL;
  PetscInt        sfirst = -1, stride = -1;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  dm   = field->dm;
  nc   = field->numComponents;
  disc = dsfield->disc;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = PetscSectionGetField(section,dsfield->fieldNum,&section);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(disc,&classid);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad,NULL,NULL,&nq,&qpoints,NULL);CHKERRQ(ierr);
  /* TODO: batch */
  ierr = PetscObjectTypeCompare((PetscObject)cellIS,ISSTRIDE,&isStride);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS,&numCells);CHKERRQ(ierr);
  if (isStride) {
    ierr = ISStrideGetInfo(cellIS,&sfirst,&stride);CHKERRQ(ierr);
  } else {
    ierr = ISGetIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  if (classid == PETSCFE_CLASSID) {
    PetscFE      fe = (PetscFE) disc;
    PetscInt     feDim, i;
    PetscReal    *fB = NULL, *fD = NULL, *fH = NULL;
    PetscInt     closureSize;
    PetscScalar  *elem = NULL;

    ierr = PetscFEGetDimension(fe,&feDim);CHKERRQ(ierr);
    ierr = PetscFEGetTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
    closureSize = feDim;
    for (i = 0; i < numCells; i++) {
      PetscInt c = isStride ? (sfirst + i * stride) : cells[i];

      ierr = DMPlexVecGetClosure(dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
      if (B) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cB = &((PetscScalar *) B)[nc * nq * i];

          DMFieldDSdot(cB,fB,elem,nq,feDim,nc,(PetscScalar));
        } else {
          PetscReal *cB = &((PetscReal *) B)[nc * nq * i];

          DMFieldDSdot(cB,fB,elem,nq,feDim,nc,PetscRealPart);
        }
      }
      if (D) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cD = &((PetscScalar *) D)[nc * nq * dim * i];

          DMFieldDSdot(cD,fD,elem,nq,feDim,(nc * dim),(PetscScalar));
        } else {
          PetscReal *cD = &((PetscReal *) D)[nc * nq * dim * i];

          DMFieldDSdot(cD,fD,elem,nq,feDim,(nc * dim),PetscRealPart);
        }
      }
      if (H) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cH = &((PetscScalar *) H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH,fH,elem,nq,feDim,(nc * dim * dim),(PetscScalar));
        } else {
          PetscReal *cH = &((PetscReal *) H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH,fH,elem,nq,feDim,(nc * dim * dim),PetscRealPart);
        }
      }
    }
    ierr = DMRestoreWorkArray(dm,feDim,PETSC_SCALAR,&elem);CHKERRQ(ierr);
    ierr = PetscFERestoreTabulation(fe,nq,qpoints,B ? &fB : NULL,D ? &fD : NULL,H ? &fH : NULL);CHKERRQ(ierr);
  } else {SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented");}
  if (!isStride) {
    ierr = ISRestoreIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldGetFEInvariance_DS(DMField field, IS cellIS, PetscBool *isConstant, PetscBool *isAffine, PetscBool *isQuadratic)
{
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  disc = dsfield->disc;
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE    fe = (PetscFE) disc;
    PetscInt   order, maxOrder;
    PetscBool  tensor = PETSC_FALSE;
    PetscSpace sp;

    ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
    ierr = PetscSpaceGetOrder(sp,&order);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialGetTensor(sp,&tensor);CHKERRQ(ierr);
    if (tensor) {
      PetscInt dim;

      ierr = DMGetDimension(field->dm,&dim);CHKERRQ(ierr);
      maxOrder = order * dim;
    } else {
      maxOrder = order;
    }
    if (isConstant)  *isConstant  = (maxOrder < 1) ? PETSC_TRUE : PETSC_FALSE;
    if (isAffine)    *isAffine    = (maxOrder < 2) ? PETSC_TRUE : PETSC_FALSE;
    if (isQuadratic) *isQuadratic = (maxOrder < 3) ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DS(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy         = DMFieldDestroy_DS;
  field->ops->evaluate        = DMFieldEvaluate_DS;
  field->ops->evaluateFE      = DMFieldEvaluateFE_DS;
  field->ops->getFEInvariance = DMFieldGetFEInvariance_DS;
  field->ops->view            = DMFieldView_DS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dsfield);CHKERRQ(ierr);
  field->data = dsfield;
  ierr = DMFieldInitialize_DS(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDS(DM dm, PetscInt fieldNum, Vec vec,DMField *field)
{
  DMField        b;
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscBool      isContainer = PETSC_FALSE;
  PetscClassId   id = -1;
  PetscInt       numComponents = -1;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(section,fieldNum,&numComponents);CHKERRQ(ierr);
  ierr = DMGetField(dm,fieldNum,&disc);CHKERRQ(ierr);
  if (disc) {
    ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
    isContainer = (id == PETSC_CONTAINER_CLASSID) ? PETSC_TRUE : PETSC_FALSE;
  }
  if (!disc || isContainer) {
    MPI_Comm        comm = PetscObjectComm((PetscObject) dm);
    PetscInt        cStart, cEnd, dim;
    PetscInt        localConeSize = 0, coneSize;
    PetscFE         fe;
    PetscDualSpace  Q;
    PetscSpace      P;
    DM              K;
    PetscQuadrature quad, fquad;
    PetscBool       isSimplex;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (cEnd > cStart) {
      ierr = DMPlexGetConeSize(dm, cStart, &localConeSize);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&localConeSize,&coneSize,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
    isSimplex = (coneSize == (dim + 1)) ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSpaceCreate(comm, &P);CHKERRQ(ierr);
    ierr = PetscSpaceSetOrder(P, 1);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(P, numComponents);CHKERRQ(ierr);
    ierr = PetscSpaceSetType(P,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetTensor(P, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetNumComponents(Q, numComponents);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetOrder(Q, 1);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(Q, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
    ierr = PetscFECreate(comm, &fe);CHKERRQ(ierr);
    ierr = PetscFESetType(fe,PETSCFEBASIC);CHKERRQ(ierr);
    ierr = PetscFESetBasisSpace(fe, P);CHKERRQ(ierr);
    ierr = PetscFESetDualSpace(fe, Q);CHKERRQ(ierr);
    ierr = PetscFESetNumComponents(fe, numComponents);CHKERRQ(ierr);
    ierr = PetscFESetUp(fe);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
    ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
    if (isSimplex) {
      ierr = PetscDTGaussJacobiQuadrature(dim,   1, 1, -1.0, 1.0, &quad);CHKERRQ(ierr);
      ierr = PetscDTGaussJacobiQuadrature(dim-1, 1, 1, -1.0, 1.0, &fquad);CHKERRQ(ierr);
    }
    else {
      ierr = PetscDTGaussTensorQuadrature(dim,   1, 1, -1.0, 1.0, &quad);CHKERRQ(ierr);
      ierr = PetscDTGaussTensorQuadrature(dim-1, 1, 1, -1.0, 1.0, &fquad);CHKERRQ(ierr);
    }
    ierr = PetscFESetQuadrature(fe, quad);CHKERRQ(ierr);
    ierr = PetscFESetFaceQuadrature(fe, fquad);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&fquad);CHKERRQ(ierr);
    disc = (PetscObject) fe;
  } else {
    ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;

    ierr = PetscFEGetNumComponents(fe,&numComponents);CHKERRQ(ierr);
  } else {SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented");}
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDS);CHKERRQ(ierr);
  dsfield = (DMField_DS *) b->data;
  dsfield->fieldNum = fieldNum;
  dsfield->disc = disc;
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  dsfield->vec = vec;
  *field = b;
  PetscFunctionReturn(0);
}
