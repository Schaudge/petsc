#if !defined(__TAOVMNOS_H)
#define __TAO_VMNOS_H
#include <petsc/private/taoimpl.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h>
#include <../src/ksp/ksp/utils/lmvm/diagbb/diagbb.h>

typedef struct _TaoVMNOSOps *TaoVMNOSOps;

struct _TaoVMNOSOps {
  PetscErrorCode (*f1obj)(Tao, Vec, PetscReal*, void*);
  PetscErrorCode (*f1grad)(Tao, Vec,  Vec, void*);
};

typedef struct {
  PETSCHEADER(struct _TaoVMNOSOps);
  Tao                f1subtao, parent;
  Mat                vm;
  Vec                temp,temp2;
  Vec                xk,xk_old,zk,zk_old,uk,f1grad,f1grad_old;
  void*              f1objP;
  void*              f1gradP;
  PetscInt           vm_rule,lniter; //vm_rule: 0: ATOS, 1:VMNOS, 2:LMVM?
  PetscInt           proxnum; // Number of subsolver operators
  PetscReal          mu,resnorm,stepsize,bs_factor,Lip,ls_eps,ls_tol; //resnorm: ||x-z||_2
  PetscReal          gatol_vmnos,catol_vmnos,tol;
  PetscBool          linesearch;
  TaoVMNOSUpdateType vm_update;
  Tao                *subtaos;
} TAO_VMNOS;

#endif
