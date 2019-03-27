#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>

#include "ridc.h"

#include <petsc.h>

using namespace std;
// replace with petsc time stepper
TS ts;

static PetscErrorCode petscrhs(TS,PetscReal,Vec,Vec,void *);
static PetscErrorCode petscrhsjacobian(TS,PetscReal,Vec,Mat,Mat,void *);

class ImplicitOde : public ODE {
public:
  ImplicitOde(int my_neq, int my_nt, double my_ti, double my_tf, double my_dt) {
    neq = my_neq;
    nt = my_nt;
    ti = my_ti;
    tf = my_tf;
    dt = my_dt;
  }

  void rhs(double t,double *u,double *f) {
    for (int i =0; i<neq; i++) {
      f[i]=-(i+1)*t*u[i];
    }
  }

  void step(double t, double *u, double *unew) {
    Vec sol;
    PetscInt    i;
    PetscScalar *x;
    PetscScalar ptime;
    PetscInt    steps;

    TSGetTime(ts,&ptime);
    TSGetStepNumber(ts,&steps);
    TSGetSolution(ts,&sol);
    TSMonitor(ts,steps,ptime,sol);
    TSSetTime(ts,ptime);
    VecGetArray(sol,&x);
    for (i=0; i<neq; i++) x[i] = u[i];
    VecRestoreArray(sol,&x);
    TSStep(ts);
    TSGetSolution(ts,&sol);
    VecGetArray(sol,&x);
    for (i=0; i<neq; i++) unew[i] = x[i];
    VecRestoreArray(sol,&x);
  }
};

PetscErrorCode petscrhs(TS ts,PetscReal t,Vec U,Vec F,void *ctx) {
  ImplicitOde       *ode = (ImplicitOde*)ctx;
  PetscScalar       *f;
  const PetscScalar *u;
  PetscInt          i;

  PetscFunctionBeginUser;
  VecGetArrayRead(U,&u);
  VecGetArray(F,&f);
  for (i =0; i<ode->neq; i++) {
    f[i]=-(i+1)*t*u[i];
  }
  VecRestoreArrayRead(U,&u);
  VecRestoreArray(F,&f);
  PetscFunctionReturn(0);
}

PetscErrorCode petscrhsjacobian(TS ts,PetscReal t,Vec U,Mat J, Mat B,void *ctx) {
  ImplicitOde *ode = (ImplicitOde*)ctx;
  PetscScalar jac[2][2] = {{0}};
  PetscInt    i,rows[2] = {0,1},cols[2] = {0,1};

  PetscFunctionBeginUser;
  for (i=0;i<ode->neq;i++) jac[i][i] = -(i+1)*t;
  MatSetValues(J,2,rows,2,cols,&jac[0][0],INSERT_VALUES);
  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (B!=J) {
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  PetscInt     order=1, nt=10;
  PetscScalar  *sol;
  PetscInt     neq = 2;
  PetscReal    ti = 0.0;
  PetscReal    tf = 1.0,dt;
  PetscScalar  *uptr;
  Vec          u;
  Mat          J;

  PetscInitialize(&argc,&argv,NULL,NULL);
  PetscOptionsGetInt(NULL,NULL,"-order",&order,NULL);
  PetscOptionsGetInt(NULL,NULL,"-steps",&nt,NULL);
  dt = (tf - ti)/nt; // compute dt
  VecCreate(PETSC_COMM_WORLD,&u);
  VecSetSizes(u,2,PETSC_DETERMINE);
  VecSetUp(u);
  MatCreate(PETSC_COMM_WORLD,&J);
  MatSetType(J,MATSEQAIJ);
  MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2);
  MatSetFromOptions(J);
  MatSetUp(J);

  // initialize ODE variable
  ImplicitOde *ode = new ImplicitOde(neq,nt,ti,tf,dt);

  TSCreate(PETSC_COMM_WORLD,&ts);
  TSSetRHSFunction(ts,NULL,petscrhs,ode);
  TSSetRHSJacobian(ts,J,J,petscrhsjacobian,ode);
  TSSetMaxSteps(ts,10000);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,tf);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);
  TSSetFromOptions(ts);
  TSSetSolution(ts,u);

  VecGetArray(u,&uptr);
  sol = new double[neq];
  // specify initial condition
  for (int i =0; i<neq; i++) {
    sol[i] = 1.0;
    uptr[i] = 1.0;
  }
  VecRestoreArray(u,&uptr);
  // TSSolve(ts,u);
  // VecView(u,PETSC_VIEWER_STDOUT_WORLD);

  // call ridc
  ridc_be(ode, order, sol);
  // output solution to screen
  for (int i = 0; i < neq; i++)
    printf("%14.12f\n", sol[i]);
  VecDestroy(&u);
  MatDestroy(&J);
  TSDestroy(&ts);
  // printf("ref solution:\n");
  // printf("%14.12f\n", PetscExpReal(-tf*tf/2.));
  // printf("%14.12f\n", PetscExpReal(-tf*tf));

  delete [] sol;
  delete ode;
}
