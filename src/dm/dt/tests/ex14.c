const char help[] = "Test FEEC projections.\n\n";

#include <petscfe.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>

static PetscErrorCode createFE(DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscBool useMoments, PetscInt origDegree, PetscFE * fem)
{
  PetscBool      tensor = PETSC_FALSE;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dim = DMPolytopeTypeGetDim(tope);
  switch (tope) {
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
    tensor = PETSC_TRUE;
    break;
  default:
    break;
  }
  if (degree == 0 && PetscAbsInt(formDegree) != dim) {
    degree = 1;
    trimmed = PETSC_TRUE;
  }
  ierr = PetscFECreateFEEC(PETSC_COMM_SELF, tope, degree, formDegree, PETSC_DETERMINE, 1, tensor, trimmed, useMoments, origDegree, fem);
  if (ierr == PETSC_ERR_SUP) {
    PetscFunctionReturn(ierr);
  }
  CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test(DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscBool useMoments, PetscInt origDegree, PetscReal *error)
{
  PetscFE           F, dF;
  PetscDualSpace    X, dX;
  Mat               E1, Proj1;
  Mat               D1, D2, D1P1; // differential matrix
  Mat               DP, PD;
  PetscInt          Nb, Nform1, Nform2;
  PetscReal        *pScalar1, *pScalar2;
  PetscQuadrature   q1, q2;
  Mat               P1, P2; // projectors
  PetscInt          nPoints1, nPoints2;
  PetscInt          dim;
  PetscTabulation   T;
  const PetscReal  *points1, *points2;
  PetscInt         Nb1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  dim = DMPolytopeTypeGetDim(tope);
  ierr = PetscDTBinomialInt(dim + origDegree, dim, &Nb);CHKERRQ(ierr);

  // create the original finite element and project the polynomials into the space
  ierr = createFE(tope, degree, formDegree, trimmed, useMoments, origDegree, &F);
  if (ierr == PETSC_ERR_SUP) {
    PetscFunctionReturn(ierr);
  }
  CHKERRQ(ierr);
  // create the differential finite element
  ierr = createFE(tope, trimmed ? degree : degree - 1, PetscAbsInt(formDegree) + 1, trimmed, useMoments, origDegree, &dF);
  if (ierr == PETSC_ERR_SUP) {
    PetscErrorCode ierr2 = PetscFEDestroy(&F);CHKERRQ(ierr2);
    PetscFunctionReturn(ierr);
  }
  CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(F, &X);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(X, &q1, &P1);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q1, NULL, NULL, &nPoints1, &points1, NULL);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nform1);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nb * nPoints1, &pScalar1);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, nPoints1, points1, origDegree, 0, pScalar1);CHKERRQ(ierr);
  // E1: basis evaluated at points1
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, nPoints1 * Nform1, Nb * Nform1, NULL, &E1);CHKERRQ(ierr);
  {
    PetscScalar *a;
    PetscInt b_strl = Nform1 * nPoints1;
    PetscInt f_strl = 1 + b_strl * Nb;
    PetscInt pt_strl = Nform1;

    PetscInt b_strr = nPoints1;
    PetscInt pt_strr = 1;
    ierr = MatDenseGetArrayWrite(E1, &a);CHKERRQ(ierr);
    for (PetscInt b = 0; b < Nb; b++) {
      for (PetscInt f = 0; f < Nform1; f++) {
        for (PetscInt pt = 0; pt < nPoints1; pt++) {
          a[b * b_strl + f * f_strl + pt * pt_strl] = pScalar1[b * b_strr + pt * pt_strr];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(E1, &a);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)E1, "Evaluation");CHKERRQ(ierr);
  ierr = MatMatMult(P1, E1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Proj1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Proj1, "F projection");CHKERRQ(ierr);

  ierr = PetscFEGetDualSpace(dF, &dX);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(dX, &q2, &P2);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q2, NULL, NULL, &nPoints2, &points2, NULL);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree) + 1, &Nform2);CHKERRQ(ierr);

  // Project the differential of F into dF
  ierr = PetscFECreateTabulation(F, 1, nPoints2, points2, 1, &T);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(F, &Nb1);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nform2 * nPoints2, Nb1, NULL, &D1);CHKERRQ(ierr);
  {
    PetscScalar *a;
    PetscInt (*pattern)[3] = NULL;
    PetscInt nnz = Nform2 * (PetscAbsInt(formDegree) + 1);

    PetscInt b_strl  = Nform2 * nPoints2;
    PetscInt pt_strl = Nform2;
    PetscInt fi_strl = 1;

    PetscInt pt_strr  = dim * Nform1 * Nb1;
    PetscInt b_strr   = dim * Nform1;
    PetscInt fj_strr  = dim;
    PetscInt jet_strr = 1;

    PetscReal *D = T->T[1];

    ierr = PetscMalloc1(nnz, &pattern);CHKERRQ(ierr);
    ierr = PetscDTAltVDifferentialPattern(dim, formDegree, PetscAbsInt(formDegree) + 1, pattern);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(D1, &a);CHKERRQ(ierr);
    for (PetscInt l = 0; l < nnz; l++) {
      PetscInt jetc = pattern[l][0];
      PetscInt fj = pattern[l][1];
      PetscInt fi = pattern[l][2];
      PetscInt scale = 1.;
      if (fi < 0) {
        scale = -1.;
        fi = -(fi+1);
      }
      for (PetscInt b = 0; b < Nb1; b++) {
        for (PetscInt pt = 0; pt < nPoints2; pt++) {
          a[b * b_strl + fi * fi_strl + pt * pt_strl] += scale * D[b * b_strr + jetc * jet_strr + fj * fj_strr + pt * pt_strr];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(D1, &a);CHKERRQ(ierr);
    ierr = PetscFree(pattern);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)D1, "dF");CHKERRQ(ierr);
  ierr = MatMatMult(D1, Proj1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D1P1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)D1P1, "dProj evaluation");CHKERRQ(ierr);
  ierr = MatMatMult(P2, D1P1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DP);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)DP, "DP");CHKERRQ(ierr);

  // D2: evaluate the differential of the basis points2
  ierr = PetscMalloc1(Nb * (1 + dim) * nPoints2, &pScalar2);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, nPoints2, points2, origDegree, 1, pScalar2);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nform2 * nPoints2, Nb * Nform1, NULL, &D2);CHKERRQ(ierr);
  {
    PetscScalar *a;
    PetscInt (*pattern)[3] = NULL;
    PetscInt nnz = Nform2 * (PetscAbsInt(formDegree) + 1);
    PetscInt b_strl = Nform2 * nPoints2;
    PetscInt fj_strl = b_strl * Nb;
    PetscInt fi_strl = 1;
    PetscInt pt_strl = Nform2;

    PetscInt b_strr = nPoints2 * (1 + dim);
    PetscInt jet_strr = nPoints2;
    PetscInt pt_strr = 1;

    ierr = PetscMalloc1(nnz, &pattern);CHKERRQ(ierr);
    ierr = PetscDTAltVDifferentialPattern(dim, formDegree, PetscAbsInt(formDegree) + 1, pattern);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(D2, &a);CHKERRQ(ierr);
    for (PetscInt l = 0; l < nnz; l++) {
      PetscInt jetc = pattern[l][0];
      PetscInt fj = pattern[l][1];
      PetscInt fi = pattern[l][2];
      PetscInt scale = 1.;
      if (fi < 0) {
        scale = -1.;
        fi = -(fi+1);
      }
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt pt = 0; pt < nPoints2; pt++) {
          a[b * b_strl + fi * fi_strl + fj * fj_strl + pt * pt_strl] += scale * pScalar2[b * b_strr + (1 + jetc) * jet_strr + pt * pt_strr];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(D2, &a);CHKERRQ(ierr);
    ierr = PetscFree(pattern);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)D2, "Differential evaluation");CHKERRQ(ierr);
  ierr = MatMatMult(P2, D2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PD);CHKERRQ(ierr);
  ierr = MatAXPY(PD, -1.0, DP, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(PD, NORM_INFINITY, error);CHKERRQ(ierr);

  ierr = MatDestroy(&PD);CHKERRQ(ierr);
  ierr = MatDestroy(&DP);CHKERRQ(ierr);
  ierr = MatDestroy(&D2);CHKERRQ(ierr);
  ierr = MatDestroy(&D1P1);CHKERRQ(ierr);
  ierr = MatDestroy(&D1);CHKERRQ(ierr);
  ierr = MatDestroy(&Proj1);CHKERRQ(ierr);
  ierr = MatDestroy(&E1);CHKERRQ(ierr);
  ierr = PetscFree2(pScalar1, pScalar2);CHKERRQ(ierr);
  ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&dF);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DMPolytopeType topes[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_TETRAHEDRON, DM_POLYTOPE_HEXAHEDRON, DM_POLYTOPE_TRI_PRISM};
  PetscInt       ntopes = sizeof(topes) / sizeof (DMPolytopeType);
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  for (PetscInt t = 0; t < ntopes; t++) {
    DMPolytopeType tope = topes[t];
    PetscInt dim = DMPolytopeTypeGetDim(tope);
    for (PetscInt degree = 1; degree <= 2; degree++) {
      for (PetscInt formDegree = -dim+1; formDegree < dim; formDegree++) {
        for (PetscInt trimmed = 0; trimmed <= (formDegree == 0 ? 0 : 1); trimmed++) {
          for (PetscInt useMoments = 0; useMoments <= 1; useMoments++) {
            PetscReal error;
            PetscInt  origDegree = degree + 1;
            switch (tope) {
            case DM_POLYTOPE_QUADRILATERAL:
            case DM_POLYTOPE_TRI_PRISM:
              origDegree = 2*degree + 1;
              break;
            case DM_POLYTOPE_HEXAHEDRON:
              origDegree = 3*degree + 1;
              break;
            default:
              break;
            }

            ierr = test(tope, degree, formDegree, trimmed, useMoments, origDegree, &error);
            if (ierr == PETSC_ERR_SUP) {
              ierr = PetscPrintf(PETSC_COMM_WORLD, "%s, form degree %D, %s, degree %D, origDegree %D, %s, not implemented\n", DMPolytopeTypes[tope], formDegree, trimmed ? "trimmed" : "full", degree, origDegree, useMoments ? "modal" : "nodal");CHKERRQ(ierr);
            } else {
              CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD, "%s, form degree %D, %s, degree %D, origDegree %D, %s, projection commutator error %sok (%g)\n", DMPolytopeTypes[tope], formDegree, trimmed ? "trimmed" : "full", degree, origDegree, useMoments ? "modal" : "nodal", error < PETSC_SMALL ? "": "not ", (double) error);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args:

TEST*/
