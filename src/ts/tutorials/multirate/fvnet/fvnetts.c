#include "fvnet.h"

/* Adjust the time step depending on a current solutions */
PetscErrorCode FVNetworkPreStep(TS ts) 
{
  PetscInt          n,eStart,eEnd,e;
  PetscReal         t,dt;
  PetscErrorCode    ierr;
  FVNetwork         fvnet;
  FVEdge            fvedge; 
  
  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&fvnet);CHKERRQ(ierr);
  ierr = fvnet->gettimestep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&n);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  if(fvnet->monifv >= 2) 
  {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g dt %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)dt);CHKERRQ(ierr);
  }
  /* Reset edge cfl_idt values to 0 for the next time step */
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  /* Reset the edge cfl_idt values to 0 */
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    fvedge->cfl_idt = 0; 
  }

  PetscFunctionReturn(0);
}
PetscErrorCode FVNetwork_GetTimeStep_Fixed(TS ts ,PetscReal *dt) 
{
  PetscErrorCode    ierr;

  PetscFunctionBegin; 
  /* Keep the current time step */
  ierr = TSGetTimeStep(ts,dt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetwork_GetTimeStep_Adaptive(TS ts ,PetscReal *dt)
{
  PetscErrorCode    ierr;
  FVNetwork         fvnet;
  TSType            tstype;
  TSMPRKType        mprktype;
  PetscReal         cfl_idt_slow = 0, cfl_idt_fast = 0, cfl_idt = 0;
  const PetscInt    *index;
  PetscInt          i,size,e,eStart,eEnd;
  FVEdge            fvedge;
  PetscBool         flg,flg1,flg2; 

  PetscFunctionBegin; 
  ierr = TSGetApplicationContext(ts,&fvnet);CHKERRQ(ierr);
  ierr = TSGetType(ts,&tstype);CHKERRQ(ierr);
  ierr = PetscStrcmp(TSMPRK,tstype,&flg);CHKERRQ(ierr);
  if(flg) {
    /* Assumes a fast/slow partition */
    ierr = ISGetIndices(fvnet->slow_edges,&index);CHKERRQ(ierr);
    ierr = ISGetLocalSize(fvnet->slow_edges,&size);CHKERRQ(ierr);
    /* Iterate through the (local) slow edges and compute the maximum of all cfl_idt */
    for (i=0; i<size; i++) {
      e    = index[i];
      ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
      cfl_idt_slow = PetscMax(PetscAbsScalar(cfl_idt_slow),PetscAbsScalar(fvedge->cfl_idt));CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&cfl_idt_slow,&cfl_idt_slow,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)fvnet->network));CHKERRQ(ierr);
    ierr = ISGetIndices(fvnet->fast_edges,&index);CHKERRQ(ierr);
    ierr = ISGetLocalSize(fvnet->fast_edges,&size);CHKERRQ(ierr);
    /* Iterate through the (local) fast edges and compute the maximum of all cfl_idt */
    for (i=0; i<size; i++) {
      e    = index[i];
      ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
      cfl_idt_fast = PetscMax(PetscAbsScalar(cfl_idt_fast),PetscAbsScalar(fvedge->cfl_idt));CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&cfl_idt_fast,&cfl_idt_fast,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)fvnet->network));CHKERRQ(ierr);
    /* Determine the TSMPRK type */
    ierr = TSMPRKGetType(ts,&mprktype);CHKERRQ(ierr);
    ierr = PetscStrcmp(TSMPRKP2,mprktype,&flg1);CHKERRQ(ierr);
    ierr = PetscStrcmp(TSMPRK2A22,mprktype,&flg2);CHKERRQ(ierr);
    if(flg1 || flg2) { 
      cfl_idt = PetscMax(cfl_idt_slow,(cfl_idt_fast/2));
      if(fvnet->monifv >= 3) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"dt_slow -> %g    dt_fast -> %g\n",(double)(fvnet->cfl/cfl_idt_slow),(double)(2*fvnet->cfl/cfl_idt_fast));CHKERRQ(ierr);
      }
    } else {
      ierr = PetscStrcmp(TSMPRKP3,mprktype,&flg1);CHKERRQ(ierr);
      ierr = PetscStrcmp(TSMPRK2A32,mprktype,&flg2);CHKERRQ(ierr);
      if (flg1 || flg2) {
        cfl_idt = PetscMax(cfl_idt_slow,(cfl_idt_fast/3));
        if(fvnet->monifv >= 3) {
          ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"dt_slow -> %g    dt_fast -> %g\n",(double)(fvnet->cfl/cfl_idt_slow),(double)(3*fvnet->cfl/cfl_idt_fast));CHKERRQ(ierr);
        }
      } else {
        SETERRQ1(PETSC_COMM_SELF,1,"FVNet: MPRK type \"%s\" is not yet supported",mprktype);
      }
    }
  } else {
    /* Single Rate ODE Solver */

    /* Iterate through all (local) edges and compute the maximum cfl_idt */
    ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
    for (e=eStart; e<eEnd; e++) {
      ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
      cfl_idt = PetscMax(PetscAbsScalar(cfl_idt),PetscAbsScalar(fvedge->cfl_idt));CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&cfl_idt,&cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)fvnet->network));CHKERRQ(ierr);
  }
  /* Compute the next time step length */
  *dt = fvnet->cfl/cfl_idt;
  PetscFunctionReturn(0);
}