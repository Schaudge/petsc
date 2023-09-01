static char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/* Program usage: mpiexec -n 1 rosenbrock1 [-help] [all TAO options] */

#include "rosenbrock1.h"

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(RosenbrockMain());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 1
      nsize: {{1 2 3}}
      args: -tao_smonitor -tao_type nls -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 2
      args: -tao_smonitor -tao_type lmvm -tao_gatol 1.e-3

   test:
      suffix: 3
      args: -tao_smonitor -tao_type ntr -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 4
      args: -tao_smonitor -tao_type ntr -tao_mf_hessian -tao_ntr_pc_type none -tao_gatol 1.e-4

   test:
      suffix: 5
      args: -tao_smonitor -tao_type bntr -tao_gatol 1.e-4

   test:
      suffix: 6
      args: -tao_smonitor -tao_type bntl -tao_gatol 1.e-4

   test:
      suffix: 7
      args: -tao_smonitor -tao_type bnls -tao_gatol 1.e-4

   test:
      suffix: 8
      args: -tao_smonitor -tao_type bntr -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
      suffix: 9
      args: -tao_smonitor -tao_type bntl -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
      suffix: 10
      args: -tao_smonitor -tao_type bnls -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4

   test:
      suffix: 11
      args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbroyden

   test:
      suffix: 12
      args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbadbroyden

   test:
     suffix: 13
     args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbroyden

   test:
     suffix: 14
     args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbfgs

   test:
     suffix: 15
     args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmdfp

   test:
     suffix: 16
     args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 17
     args: -tao_smonitor -tao_gatol 1e-4 -tao_type bqnls

   test:
     suffix: 18
     args: -tao_smonitor -tao_gatol 1e-4 -tao_type blmvm

   test:
     suffix: 19
     args: -tao_smonitor -tao_gatol 1e-4 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 20
     args: -tao_monitor -tao_gatol 1e-4 -tao_type blmvm -tao_ls_monitor

   test:
     suffix: 21
     args: -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbadbroyden

   test:
     suffix: 22
     args: -tao_max_it 1 -tao_converged_reason

   test:
     suffix: 23
     args: -tao_max_funcs 0 -tao_converged_reason

   test:
     suffix: 24
     args: -tao_gatol 10 -tao_converged_reason

   test:
     suffix: 25
     args: -tao_grtol 10 -tao_converged_reason

   test:
     suffix: 26
     args: -tao_gttol 10 -tao_converged_reason

   test:
     suffix: 27
     args: -tao_steptol 10 -tao_converged_reason

   test:
     suffix: 28
     args: -tao_fmin 10 -tao_converged_reason

   test: 
     suffix: 29
     args: -tao_type lmvm -tao_max_it 10 -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_reorder 

   test: 
     suffix: 30
     args: -tao_type lmvm -tao_max_it 10 -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_inplace

   test: 
     suffix: 31
     args: -tao_type lmvm -tao_max_it 10 -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_reorder -tao_lmvm_mat_lmvm_scale_type none

   test: 
     suffix: 32
     args: -tao_type lmvm -tao_max_it 10 -tao_lmvm_mat_type lmvmcdbfgs -mat_lbfgs_type cd_inplace -tao_lmvm_mat_lmvm_scale_type none

   test: 
     suffix: 33
     args: -tao_type bqnls -tao_bqnls_mat_type lmvmcdbfgs -tao_monitor -mat_lbfgs_type cd_reorder 

   test: 
     suffix: 34
     args: -tao_type bqnls -tao_bqnls_mat_type lmvmcdbfgs -tao_monitor -mat_lbfgs_type cd_inplace

   test:
     suffix: snes
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -ksp_type cg

   test:
     suffix: snes_ls_armijo
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtonls -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -snes_linesearch_monitor -snes_linesearch_order 1

   test:
     suffix: snes_tr_cgnegcurve_kmdc
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none  -ksp_type cg -snes_tr_kmdc 0.01 -ksp_converged_neg_curve -ksp_converged_reason
     requires: !single

   test:
     suffix: chained_bfgs
     args: -tao_smonitor -chained -n 10000 -tao_type lmvm -tao_max_it 20 -tao_lmvm_mat_lmvm_scale_type none

TEST*/
