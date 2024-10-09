
#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscErrorCode TaoTermTestGradient(TaoTerm term, Vec x, Vec params, Vec g1, PetscViewer mviewer)
{
  Vec               g2, g3;
  PetscBool         complete_print = PETSC_FALSE;
  PetscReal         hcnorm, fdnorm, hcmax, fdmax, diffmax, diffnorm;
  PetscScalar       dot;
  MPI_Comm          comm;
  PetscViewer       viewer, mviewer;
  PetscViewerFormat format;
  PetscInt          tabs;
  static PetscBool  directionsprinted = PETSC_FALSE;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)term);
  PetscCall(PetscOptionsViewer("-taoterm_test_gradient_view", "View difference between hand-coded and finite difference Gradients element entries", "TaoTermTestGradient", &mviewer, &format, &complete_print));
  PetscOptionsEnd();

  PetscCall(PetscObjectGetComm((PetscObject)term, &comm));
  PetscCall(PetscViewerASCIIGetStdout(comm, &viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)term)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ---------- Testing Gradient -------------\n"));
  if (!complete_print && !directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Run with -tao_test_gradient_view and optionally -tao_test_gradient <threshold> to show difference\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    of hand-coded and finite difference gradient entries greater than <threshold>.\n"));
  }
  if (!directionsprinted) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Testing hand-coded Gradient, if (for double precision runs) ||G - Gfd||/||G|| is\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    O(1.e-8), the hand-coded Gradient is probably correct.\n"));
    directionsprinted = PETSC_TRUE;
  }
  if (complete_print) PetscCall(PetscViewerPushFormat(mviewer, format));

  PetscCall(VecDuplicate(x, &g2));
  PetscCall(VecDuplicate(x, &g3));

  /* Compute finite difference gradient, assume the gradient is already computed by TaoComputeGradient() and put into g1 */
  PetscCall(TaoTermGradientFD(term, x, g2, NULL));

  PetscCall(VecNorm(g2, NORM_2, &fdnorm));
  PetscCall(VecNorm(g1, NORM_2, &hcnorm));
  PetscCall(VecNorm(g2, NORM_INFINITY, &fdmax));
  PetscCall(VecNorm(g1, NORM_INFINITY, &hcmax));
  PetscCall(VecDot(g1, g2, &dot));
  PetscCall(VecCopy(g1, g3));
  PetscCall(VecAXPY(g3, -1.0, g2));
  PetscCall(VecNorm(g3, NORM_2, &diffnorm));
  PetscCall(VecNorm(g3, NORM_INFINITY, &diffmax));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ||Gfd|| %g, ||G|| = %g, angle cosine = (Gfd'G)/||Gfd||||G|| = %g\n", (double)fdnorm, (double)hcnorm, (double)(PetscRealPart(dot) / (fdnorm * hcnorm))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  2-norm ||G - Gfd||/||G|| = %g, ||G - Gfd|| = %g\n", (double)(diffnorm / PetscMax(hcnorm, fdnorm)), (double)diffnorm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  max-norm ||G - Gfd||/||G|| = %g, ||G - Gfd|| = %g\n", (double)(diffmax / PetscMax(hcmax, fdmax)), (double)diffmax));

  if (complete_print) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded gradient ----------\n"));
    PetscCall(VecView(g1, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Finite difference gradient ----------\n"));
    PetscCall(VecView(g2, mviewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Hand-coded minus finite-difference gradient ----------\n"));
    PetscCall(VecView(g3, mviewer));
  }
  PetscCall(VecDestroy(&g2));
  PetscCall(VecDestroy(&g3));

  if (complete_print) {
    PetscCall(PetscViewerPopFormat(mviewer));
    PetscCall(PetscViewerDestroy(&mviewer));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
