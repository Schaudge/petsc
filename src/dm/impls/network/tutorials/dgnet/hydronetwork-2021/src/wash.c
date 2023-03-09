/*
   Subroutines for WASH
*/
#include "wash.h"

/*
  Compute dt for next TSStep
*/
PetscErrorCode TSWashPreStep(TS ts)
{
  PetscInt       n;
  PetscReal      t, dt;
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            Xold;
  Wash           wash;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts, &n);
  CHKERRQ(ierr);
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);

  ierr = TSWashGetTimeStep(ts, &dt);
  CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt);
  CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);
  CHKERRQ(ierr);

  ierr = TSGetSolution(ts, &Xold);
  CHKERRQ(ierr);
  ierr = TSGetDM(ts, &networkdm);
  CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts, &wash);
  CHKERRQ(ierr);
  ierr = VecCopy(Xold, wash->Xold);
  CHKERRQ(ierr);
  //ierr = VecView(Xold,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts), "%-10s-> step %d time %g dt %g, wash->dt %g\n", PETSC_FUNCTION_NAME, n, (double)t, (double)dt, wash->dt);
  CHKERRQ(ierr);
  //printf("TSWashPreStep %d, X:\n",n);
  //VecView(Xold,0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Update solution X at boundary, and the river cell points i=0 and ncells-1:
  At each junction, we enforce:
    riverx[0].q       = junctionx.q*riverx[1].q/sumfrom;      for upperstream
    riverx[ncell-1].q = junctionx.q*riverx[ncell-2].q/sumto;  for downstream
    riverx.h          = junctionx.h

    where
    sumfrom[v] = sum(riverx[1].q);       for upperstream
    sumto[v]   = sum(riverx[ncell-2].q); for downstream

    This ensures
      juncxtion.q = sum(riverx[ncell-1].q_downstream) = sum(riverx[0].q_upperstream)
      i.e., inflow = outflow at each junction.

  For each river:
      riverx[0]        = riverx[1] (Boundary) or junction value
      riverx[ncells-1] = riverx[ncells-2] (Boundary) or junction value
*/
PetscErrorCode TSWashPostStep(TS ts)
{
  PetscErrorCode  ierr;
  Vec             X, localX, Xtmp, localXtmp;
  Wash            wash;
  DM              networkdm;
  PetscInt        v, Start, End, nedges, i, varoffset, rivervaroffset;
  const PetscInt *edges;
  PetscScalar    *xarr, *xtmparr, sumfrom, sumto;
  MPI_Comm        comm;
  PetscMPIInt     rank;
  RiverField     *juncx, *juncxtmp, *riverx;
  PetscInt        e, vfrom, vto, type;
  const PetscInt *cone;
  River           river;
  Junction        junction;
  PetscReal       tol = 1.e-6;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &X);
  CHKERRQ(ierr);
  //if (!rank) printf("TSWashPostStep, TS solution:\n");
  //ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = TSGetDM(ts, &networkdm);
  CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts, &wash);
  CHKERRQ(ierr);

  /* update ghost values of locaX localXold */
  localX = wash->localX;
  ierr   = DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = VecGetArray(localX, &xarr);
  CHKERRQ(ierr);

  ierr = DMNetworkGetVertexRange(networkdm, &Start, &End);
  CHKERRQ(ierr);

  /* Compute sumfrom and sumto: store values in Xtmp */
  Xtmp      = wash->Xtmp;
  localXtmp = wash->localXtmp;
  ierr      = VecSet(Xtmp, 0.0);
  CHKERRQ(ierr);
  for (v = Start; v < End; v++) {
    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction, NULL);
    CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(networkdm, v, &nedges, &edges);
    CHKERRQ(ierr);
#if 0
    PetscInt gv;
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,v,&gv);CHKERRQ(ierr);
    printf(" [%d] gv %d, nedges %d; Compute sumfrom and sumto\n",rank,gv,nedges);
#endif
    if (junction->type == JUNCTION) {
      PetscInt gvoffset, gvoffset1;
      ierr = DMNetworkGetGlobalVecOffset(networkdm, v, ALL_COMPONENTS, &gvoffset);
      CHKERRQ(ierr);
      gvoffset1 = gvoffset + 1;

      for (i = 0; i < nedges; i++) {
        e    = edges[i];
        ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
        CHKERRQ(ierr);
        if (type != wash->keyRiver) {
          //ierr = PetscInfo2(NULL,"Warning: vertex %d 's supporting edge %d is not a river\n",v,e);CHKERRQ(ierr);
          PetscCall(PetscInfo(NULL, "Warning: vertex %d 's supporting edge %d is not a river\n", v, e));
          continue;
        }

        ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
        CHKERRQ(ierr);
        vfrom = cone[0]; /* local ordering */
        vto   = cone[1];
        if (v == vfrom) {
          ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
          CHKERRQ(ierr);
          riverx = (RiverField *)(xarr + rivervaroffset);
          ierr   = VecSetValues(Xtmp, 1, &gvoffset1, &riverx[1].q, ADD_VALUES);
          CHKERRQ(ierr);
        } else if (v == vto) {
          ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
          CHKERRQ(ierr);
          riverx = (RiverField *)(xarr + rivervaroffset);
          ierr   = VecSetValues(Xtmp, 1, &gvoffset, &riverx[river->ncells - 2].q, ADD_VALUES);
          CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecAssemblyBegin(Xtmp);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Xtmp);
  CHKERRQ(ierr);
  //if (!rank) printf("TSWashPostStep, Xtmp:\n");
  //ierr = VecView(Xtmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

#if 0
  /* Update coupling vertices: Xtmp[cv] = Xtmp[vfrom] + Xtmp[vto] */
  if (wash->ncsubnet) {
    PetscInt     ne,offset[2];
    RiverField   *juncx_from,*juncx_to;
    PetscScalar  val[2];

    ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(networkdm,Xtmp,INSERT_VALUES,localXtmp);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(networkdm,Xtmp,INSERT_VALUES,localXtmp);CHKERRQ(ierr);

    ierr = VecGetArray(localXtmp,&xtmparr);CHKERRQ(ierr);
    for (e=0; e<ne; e++) {
      ierr = DMNetworkGetConnectedVertices(networkdm,edges[e],&cone);CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];
      //printf("[%d] coupling edge %d --> %d\n",rank,vfrom,vto);

      ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offset[0]);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableOffset(networkdm,vto,&offset[1]);CHKERRQ(ierr);
      juncx_from = (RiverField*)(xtmparr + offset[0]);
      juncx_to   = (RiverField*)(xtmparr + offset[1]);
      val[0] = juncx_from[0].q; val[1] = juncx_from[0].h; /* save */

      /* add juncx_to to juncx_from (note: use globaloffset here!) */
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vfrom,&offset[0]);CHKERRQ(ierr);
      offset[1] = offset[0] + 1;
      ierr = VecSetValues(Xtmp,2,offset,&juncx_to[0].q,ADD_VALUES);CHKERRQ(ierr);

      /* add juncx_from to juncx_to */
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vto,&offset[0]);CHKERRQ(ierr);
      offset[1] = offset[0] + 1;
      ierr = VecSetValues(Xtmp,2,offset,val,ADD_VALUES);CHKERRQ(ierr);
      ne--;
    }
    ierr = VecRestoreArray(localXtmp,&xtmparr);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(Xtmp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Xtmp);CHKERRQ(ierr);
    //if (!rank) printf("new Xtmp\n");
    //ierr = VecView(Xtmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
#endif

  /* Update solution X at boundary, and the river cell points i=0 and ncells-1 */
  ierr = DMGlobalToLocalBegin(networkdm, Xtmp, INSERT_VALUES, localXtmp);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xtmp, INSERT_VALUES, localXtmp);
  CHKERRQ(ierr);
  ierr = VecGetArray(localXtmp, &xtmparr);
  CHKERRQ(ierr);

  //printf(" [%d] vStart/End %d, %d\n",rank,Start,End);
  //ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  for (v = Start; v < End; v++) {
    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction, NULL);
    CHKERRQ(ierr);

    ierr = DMNetworkGetLocalVecOffset(networkdm, v, ALL_COMPONENTS, &varoffset);
    CHKERRQ(ierr);
    juncx = (RiverField *)(xarr + varoffset);
    ierr  = DMNetworkGetSupportingEdges(networkdm, v, &nedges, &edges);
    CHKERRQ(ierr);
#if 0
    PetscInt gv;
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,v,&gv);CHKERRQ(ierr);
#endif

    if (junction->type == JUNCTION) {
      for (i = 0; i < nedges; i++) {
        e    = edges[i];
        ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
        CHKERRQ(ierr);
        if (type != wash->keyRiver) continue;

        ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
        CHKERRQ(ierr);
        vfrom = cone[0]; /* local ordering */
        vto   = cone[1];
        if (v == vfrom) {
          ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
          CHKERRQ(ierr);
          riverx   = (RiverField *)(xarr + rivervaroffset);
          juncxtmp = (RiverField *)(xtmparr + varoffset);
          sumfrom  = juncxtmp[0].h;

          if (sumfrom == 0.0) {
            riverx[0].q = juncx[0].q / junction->nout;
          } else {
            riverx[0].q = juncx[0].q * riverx[1].q / sumfrom;
          }
          riverx[0].h = juncx[0].h;
        } else if (v == vto) {
          ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
          CHKERRQ(ierr);
          riverx   = (RiverField *)(xarr + rivervaroffset);
          juncxtmp = (RiverField *)(xtmparr + varoffset);
          sumto    = juncxtmp[0].q;

          if (sumto == 0.0) {
            riverx[river->ncells - 1].q = juncx[0].q / junction->nin;
          } else {
            riverx[river->ncells - 1].q = juncx[0].q * riverx[river->ncells - 2].q / sumto;
          }
          riverx[river->ncells - 1].h = juncx[0].h;
        }
      }
    }
    /* update boundary solutions */
    else if (junction->btype == H) {
      e    = edges[0]; /* boundary only has one supporting edge */
      ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
      CHKERRQ(ierr);
      if (type != wash->keyRiver) break;

      ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
      CHKERRQ(ierr);
      riverx = (RiverField *)(xarr + rivervaroffset);
      if (junction->nout == 1) { /* v=vfrom, upper stream */
        riverx[0].q = riverx[1].q;
        riverx[0].h = juncx[0].h;
        juncx[0].q  = riverx[0].q;
      }
      if (junction->nin == 1) { /* v=vto, down stream */
        riverx[river->ncells - 1].q = riverx[river->ncells - 2].q;
        riverx[river->ncells - 1].h = juncx[0].h;
        juncx[0].q                  = riverx[river->ncells - 1].q;
      }
    } else if (junction->btype == Q) {
      e    = edges[0]; /* boundary only has one supporting edge */
      ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
      CHKERRQ(ierr);
      if (type != wash->keyRiver) break;

      ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
      CHKERRQ(ierr);
      riverx = (RiverField *)(xarr + rivervaroffset);
      if (junction->nout == 1) { /* v=vfrom, upper stream */
        riverx[0].h = riverx[1].h;
        riverx[0].q = juncx[0].q;
        juncx[0].h  = riverx[0].h;
      }
      if (junction->nin == 1) { /* v=vto, down stream */
        riverx[river->ncells - 1].h = riverx[river->ncells - 2].h;
        riverx[river->ncells - 1].q = juncx[0].q;
        juncx[0].h                  = riverx[river->ncells - 1].h;
      }
    } else PetscCheck(0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet, v %" PetscInt_FMT ", id_phy %" PetscInt_FMT, v - Start, junction->id_phy);
  }
  //ierr = MPI_Barrier(comm);CHKERRQ(ierr);

  /* Modify solution: if h < tol, then set q = 0.0 -- must do it for case 3.4 */
  ierr = DMNetworkGetEdgeRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  for (e = Start; e < End; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;
    ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &rivervaroffset);
    CHKERRQ(ierr);
    riverx = (RiverField *)(xarr + rivervaroffset);
    for (i = 0; i < river->ncells; i++) {
      /* if (riverx[i].h < 0.0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%d-th river[%d].h = %g \n",e,i,riverx[i].h); */
      if (riverx[i].h < tol) {
        riverx[i].q = 0.0;
        /* printf("Warning: %d-th river[%d].h = %g < tol, set q=0.0\n",e,i,riverx[i].h); */
      }
    }
  }

  ierr = VecRestoreArray(localX, &xarr);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(networkdm, localX, INSERT_VALUES, X);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localX, INSERT_VALUES, X);
  CHKERRQ(ierr);
  wash->X = X;
  //if (!rank) printf("X solution:\n");
  //ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecRestoreArray(localXtmp, &xtmparr);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSWashGetTimeStep(TS ts, PetscReal *dt)
{
  PetscErrorCode     ierr;
  DM                 dmnetwork;
  PetscInt           e, eStart, eEnd, varoffset, type;
  Vec                X;
  River              river;
  PetscReal          dt_e, dt_min = 10.0, dt_max = 0.0;
  RiverField        *riverx;
  const PetscScalar *xarr;
  Wash               wash;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &X);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(X, &xarr);
  CHKERRQ(ierr);

  ierr = TSGetDM(ts, &dmnetwork);
  CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts, &wash);
  CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dmnetwork, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) { /* each edge has only one component, river */
    ierr = DMNetworkGetComponent(dmnetwork, e, 0, &type, (void **)&river, NULL);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;
    ierr = DMNetworkGetLocalVecOffset(dmnetwork, e, ALL_COMPONENTS, &varoffset);
    CHKERRQ(ierr);
    riverx = (RiverField *)(xarr + varoffset);

    ierr = RiverGetTimeStep(river, riverx, &dt_e);
    CHKERRQ(ierr);
    if (dt_e < dt_min) dt_min = dt_e;
    if (wash->test_mscale && dt_e > dt_max) dt_max = dt_e;
  }
  ierr = VecRestoreArrayRead(X, &xarr);
  CHKERRQ(ierr);
  /* dt = min(dt_min) */
  ierr = MPI_Allreduce(&dt_min, dt, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dmnetwork));
  CHKERRQ(ierr);
  if (!rank && wash->test_mscale) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] dt_min/max %g, %g\n", rank, *dt, dt_max);
    CHKERRQ(ierr);
  }

  if (*dt < 1.e-4) { /* if dt too small, set dt=wash->dt */
                     //ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> dt %g is too small, set as wash->dt %g\n",PETSC_FUNCTION_NAME,(double)(*dt),wash->dt);CHKERRQ(ierr);
    *dt = wash->dt;
  } else if (*dt > 10.0 * wash->dt) { /* if dt too large, set dt=5.0*wash->dt */
                                      //ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> dt %g is too large, set as 5.0*wash->dt %g\n",PETSC_FUNCTION_NAME,(double)(*dt),5.0*wash->dt);CHKERRQ(ierr);
    *dt = 5.0 * wash->dt;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Evaluate RHS function at
     1) river interior cell points i=1,...,river->ncells-2
     2) junction points

     TSStep() computes solution at t+dt for these points
     TSWashPostStep() updates boundary junctions, and the river cell points i=0 and ncells-1
 */
PetscErrorCode WashRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  PetscErrorCode     ierr;
  Wash               wash = (Wash)ctx;
  DM                 networkdm;
  Vec                localX, localF;
  PetscInt           type, varoffset, e, eStart, eEnd, vfrom, vto;
  PetscScalar       *farr;
  PetscReal          dt, dx;
  River              river;
  RiverField        *riverx, *riverf, *juncf;
  Junction           junction;
  const PetscInt    *cone;
  const PetscScalar *xarr;

  PetscFunctionBegin;
  localX = wash->localX;
  localF = wash->localF;

  ierr = TSGetDM(ts, &networkdm);
  CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);
  CHKERRQ(ierr);

  /* Set F and localF as zero */
  ierr = VecSet(F, 0.0);
  CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);
  CHKERRQ(ierr);

  /* update ghost values of locaX */
  ierr = DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecGetArray(localF, &farr);
  CHKERRQ(ierr);

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    /* Querry connected junctions */
    dx   = river->length / river->ncells;
    ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
    CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];

    /* Evaluate RHSFunction at river interior cell points i=1,...,river->ncells-2 */
    ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &varoffset);
    CHKERRQ(ierr);
    riverx = (RiverField *)(xarr + varoffset);
    riverf = (RiverField *)(farr + varoffset);

    river->dt = dt;
    ierr      = RiverRHSFunctionLocal(river, riverx, riverf);
    CHKERRQ(ierr);

    /* Add upper stream flux to junction function */
    ierr = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction, NULL);
    CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &varoffset);
    CHKERRQ(ierr);
    if (junction->type == JUNCTION) {
      juncf = (RiverField *)(farr + varoffset);
      juncf[0].q -= river->flux[0] / dx;
      juncf[0].h -= river->flux[1] / dx;
    }

    /* Add down stream flux to junction function */
    ierr = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction, NULL);
    CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(networkdm, vto, ALL_COMPONENTS, &varoffset);
    CHKERRQ(ierr);
    if (junction->type == JUNCTION) {
      juncf = (RiverField *)(farr + varoffset);
      juncf[0].q += river->flux[2] / dx;
      juncf[0].h += river->flux[3] / dx;
    }
  }
  ierr = VecRestoreArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(localF, &farr);
  CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);

#if 0
  /* Update F at coupling vertices -- add fluxes of the coupling verties */
  if (wash->ncsubnet) {
    ierr = DMGlobalToLocalBegin(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
    ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

    ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
    for (e=0; e<ne; e++) {
      PetscInt       offset[2];
      RiverField     *juncf_from,*juncf_to;

      ierr = DMNetworkGetConnectedVertices(networkdm,edges[e],&cone);CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];

      ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offset[0]);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableOffset(networkdm,vto,&offset[1]);CHKERRQ(ierr);
      juncf_from = (RiverField*)(farr + offset[0]);
      juncf_to   = (RiverField*)(farr + offset[1]);

      /* add juncf_to (flux collected at vto) to juncf_from (note: use globaloffset here!) */
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vfrom,&offset[0]);CHKERRQ(ierr);
      offset[1] = offset[0] + 1;
      ierr = VecSetValues(F,2,offset,&juncf_to[0].q,ADD_VALUES);CHKERRQ(ierr);

      /* add juncf_from (flux collected at vfrom) to juncf_to */
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vto,&offset[0]);CHKERRQ(ierr);
      offset[1] = offset[0] + 1;
      ierr = VecSetValues(F,2,offset,&juncf_from[0].q,ADD_VALUES);CHKERRQ(ierr);
      ne--;
    } //else if (ne > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"ne %d is not done yet",ne);
    ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
    ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  }
#endif

#if 0
  printf("RHS:\n");
  ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashSetInitialSolution(DM networkdm, Wash wash)
{
  PetscErrorCode  ierr;
  PetscInt        vfrom, vto, offsetfrom, offsetto, type, varoffset, e, Start, End;
  PetscScalar    *xarr;
  River           river;
  Junction        junction;
  const PetscInt *cone;
  RiverField     *riverx, *juncx;
  PetscBool       ghost;
  Vec             X = wash->X, localX = wash->localX;

  PetscFunctionBegin;
  ierr = VecSet(localX, 0.0);
  CHKERRQ(ierr);
  ierr = VecGetArray(localX, &xarr);
  CHKERRQ(ierr);

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  for (e = Start; e < End; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
    CHKERRQ(ierr);
    if (type == wash->keyRiver) {
      ierr = DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &varoffset);
      CHKERRQ(ierr);

      /* Set values for the river */
      riverx = (RiverField *)(xarr + varoffset);
      ierr   = RiverSetInitialSolution(wash->caseid, wash->subcaseid, river, riverx, river->q0, river->h0);
      CHKERRQ(ierr);

      /* Set values for connected junctions */
      /* Get from and to vertices */
      ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
      CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];

      ierr = DMNetworkGetLocalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &offsetfrom);
      CHKERRQ(ierr);
      ierr = DMNetworkGetLocalVecOffset(networkdm, vto, ALL_COMPONENTS, &offsetto);
      CHKERRQ(ierr);

      /* Upstream boundary */
      ierr = DMNetworkIsGhostVertex(networkdm, vfrom, &ghost);
      CHKERRQ(ierr);
      if (!ghost) {
        ierr = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction, NULL);
        CHKERRQ(ierr);
        /* Set junction values */
        juncx      = (RiverField *)(xarr + offsetfrom);
        juncx[0].q = riverx[0].q;
        juncx[0].h = riverx[0].h;

        /* Set boundary values */
        if (junction->type != JUNCTION) {
          if (junction->btype == Q) {
            juncx[0].q = junction->bval.q;
          } else if (junction->btype == H) {
            juncx[0].h = junction->bval.h;
          }
        }
      }

      /* Downstream boundary */
      ierr = DMNetworkIsGhostVertex(networkdm, vto, &ghost);
      CHKERRQ(ierr);
      if (!ghost) {
        ierr = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction, NULL);
        CHKERRQ(ierr);
        /* Set junction values */
        juncx      = (RiverField *)(xarr + offsetto);
        juncx[0].q = riverx[river->ncells - 1].q;
        juncx[0].h = riverx[river->ncells - 1].h;

        /* Set boundary values */
        if (junction->type != JUNCTION) {
          if (junction->btype == Q) {
            juncx[0].q = junction->bval.q;
          } else if (junction->btype == H) {
            juncx[0].h = junction->bval.h;
          }
        }
      }
    } else if (type == wash->keyPump) {
      ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
      CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];
      ierr  = DMNetworkGetLocalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &offsetfrom);
      CHKERRQ(ierr);
      ierr = DMNetworkGetLocalVecOffset(networkdm, vto, ALL_COMPONENTS, &offsetto);
      CHKERRQ(ierr);

      /* Upstream boundary */
      ierr = DMNetworkIsGhostVertex(networkdm, vfrom, &ghost);
      CHKERRQ(ierr);
      if (!ghost) {
        ierr = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction, NULL);
        CHKERRQ(ierr);
        juncx = (RiverField *)(xarr + offsetfrom);

        if (junction->btype == Q) {
          juncx[0].q = junction->bval.q;
        } else if (junction->btype == H) {
          juncx[0].h = junction->bval.h;
        } else if (junction->type != JUNCTION) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet");
      }

      /* Downstream boundary */
      ierr = DMNetworkIsGhostVertex(networkdm, vto, &ghost);
      CHKERRQ(ierr);
      if (!ghost) {
        ierr = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction, NULL);
        CHKERRQ(ierr);
        juncx = (RiverField *)(xarr + offsetto);

        if (junction->btype == Q) {
          juncx[0].q = junction->bval.q;
        } else if (junction->btype == H) {
          juncx[0].h = junction->bval.h;
        } else if (junction->type != JUNCTION) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet");
      }
    } else {
      //printf("...Warning:WashSetInitialSolution: edge type (coupling?) is not supported yet, skip ...\n");
    }
  }

  ierr = VecRestoreArray(localX, &xarr);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(networkdm, localX, INSERT_VALUES, X);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localX, INSERT_VALUES, X);
  CHKERRQ(ierr);

#if 0
  /* Update coupling vertices */
  if (wash->ncsubnet) {
    ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);

    ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
    if (ne == 1) {
      const PetscInt *cone;
      PetscInt       nghost=0;
      RiverField     *juncx_from,*juncx_to;
      MPI_Comm       comm;
      PetscMPIInt    rank;
      PetscReal      normfrom,normto;
      PetscScalar    q,h;
      ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

      ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];
      ierr = DMNetworkIsGhostVertex(networkdm,vfrom,&ghost);CHKERRQ(ierr);
      if (ghost) {
        nghost++;
        //ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Coupling vfrom %d is ghost\n",rank,vfrom);CHKERRQ(ierr);
      }

      ierr = DMNetworkIsGhostVertex(networkdm,vto,&ghost);CHKERRQ(ierr);
      if (ghost) {
        nghost++;
        //ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Coupling vto %d is ghost\n",rank,vto);CHKERRQ(ierr);
      }
      //if (nghost) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Coupling nghost %d >0 is not supported yet",nghost);

      ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);
      juncx_from = (RiverField*)(xarr + offsetfrom);
      juncx_to   = (RiverField*)(xarr + offsetto);

      switch (nghost) {
      case 0: /* vfrom and vto both belong to this process */
        normfrom = PetscAbsScalar(juncx_from[0].q) + PetscAbsScalar(juncx_from[0].h);
        normto   = PetscAbsScalar(juncx_to[0].q) + PetscAbsScalar(juncx_to[0].h);
        if (normfrom == 0.0) { /* vfrom is not set */
          juncx_from[0].q = juncx_to[0].q;
          juncx_from[0].h = juncx_to[0].h;
        } else if (normto == 0.0) { /* vto is not set */
          juncx_to[0].q = juncx_from[0].q;
          juncx_to[0].h = juncx_from[0].h;
        } else { /* vfrom and vto are set by their connected rivers, set same values to them */
          q = 0.5*(juncx_from[0].q + juncx_to[0].q);
          h = 0.5*(juncx_from[0].h + juncx_to[0].h);
          juncx_from[0].q = juncx_to[0].q = q;
          juncx_from[0].h = juncx_to[0].h = h;
        }
        break;
      case 1: /* vfrom and vto are set by their connected rivers, set same values to them */
        q = 0.5*(juncx_from[0].q + juncx_to[0].q);
        h = 0.5*(juncx_from[0].h + juncx_to[0].h);
        juncx_from[0].q = juncx_to[0].q = q;
        juncx_from[0].h = juncx_to[0].h = h;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nghost %d not done yet",nghost);
      }
    } else if (ne > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"ne %d is not done yet",ne);

    ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(networkdm,localX,INSERT_VALUES,X);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(networkdm,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDMNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode   ierr;
  DMNetworkMonitor monitor;

  PetscFunctionBegin;
  monitor = (DMNetworkMonitor)context;
  ierr    = DMNetworkMonitorView(monitor, x);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashDestroyVecs(Wash wash)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&wash->X);
  CHKERRQ(ierr);
  ierr = VecDestroy(&wash->Xold);
  CHKERRQ(ierr);
  ierr = VecDestroy(&wash->Xtmp);
  CHKERRQ(ierr);
  ierr = VecDestroy(&wash->localX);
  CHKERRQ(ierr);
  ierr = VecDestroy(&wash->localXtmp);
  CHKERRQ(ierr);
  ierr = VecDestroy(&wash->localF);
  CHKERRQ(ierr);
  ierr = VecScatterDestroy(&wash->vscat_junc);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashDestroy(Wash wash)
{
  PetscErrorCode  ierr;
  PetscInt        subnet, nv, ne, i, key, nsubnet = wash->nsubnet, vkey;
  DM              networkdm = wash->dm;
  const PetscInt *vtx, *edges;
  River           river;
  Junction        junction;

  PetscFunctionBegin;
  for (subnet = 0; subnet < nsubnet; subnet++) {
    ierr = DMNetworkGetSubnetwork(networkdm, subnet, &nv, &ne, &vtx, &edges);
    CHKERRQ(ierr);
    for (i = 0; i < ne; i++) {
      ierr = DMNetworkGetComponent(networkdm, edges[i], 0, &key, (void **)&river, NULL);
      CHKERRQ(ierr);
      if (key != wash->keyRiver) continue;
      ierr = RiverCleanup(river);
      CHKERRQ(ierr);
    }

    if (wash->userJac) {
      for (i = 0; i < nv; i++) {
        ierr = DMNetworkGetComponent(networkdm, vtx[i], 0, &vkey, (void **)&junction, NULL);
        CHKERRQ(ierr);
        ierr = JunctionDestroyJacobian(networkdm, vtx[i], junction);
        CHKERRQ(ierr);
      }
    }

    ierr = PetscFree(wash->subnet[subnet]);
    CHKERRQ(ierr);
  }
  ierr = PetscFree(wash->subnet);
  CHKERRQ(ierr);
  ierr = PetscFree(wash);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  WashCreateVecs - Create vectors over communicator of dmnetwork
 */
PetscErrorCode WashCreateVecs(Wash wash)
{
  PetscErrorCode ierr;
  DM             dmnetwork = wash->dm;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(dmnetwork, &wash->X);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmnetwork, &wash->Xold);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmnetwork, &wash->Xtmp);
  CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmnetwork, &wash->localX);
  CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmnetwork, &wash->localXtmp);
  CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmnetwork, &wash->localF);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashCleanUp(Wash wash, PetscInt **edgelist)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i, nsubnet = wash->nsubnet;
  WashSubnet     Subnet;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wash->comm, &rank);
  CHKERRQ(ierr);
  for (i = 0; i < nsubnet; i++) {
    Subnet = (WashSubnet)wash->subnet[i];
    ierr   = PetscFree(edgelist[i]);
    CHKERRQ(ierr);
    ierr = PetscFree3(Subnet->junction, Subnet->river, Subnet->pump);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode JunctionsSetUp(PetscInt njunctions, Junction *junctions)
{
  PetscInt i;

  PetscFunctionBegin;
  printf("JunctionsSetUp ... edgelist:\n");
  for (i = 0; i < njunctions; i++) {
    junctions[i]->nin  = 0;
    junctions[i]->nout = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Update coupling vertices -- must do it sequentially;
     otherwise, might miss ghost coupling vertices */
PetscErrorCode WashSetUpCoupleVertices(Wash wash)
{
#if 0
  PetscErrorCode ierr;
  DM             netdm = wash->dm;
  PetscMPIInt    rank;
  MPI_Comm       comm;
  PetscInt       ne;
  const PetscInt *edges;
#endif

  PetscFunctionBegin;
#if 0
  ierr = PetscObjectGetComm((PetscObject)netdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = DMNetworkGetSubnetworkCoupleInfo(netdm,0,&ne,&edges);CHKERRQ(ierr);
  if (!rank && ne) {
    const PetscInt *cone;
    Junction       cjunction[2];
    PetscInt       nin,nout,e;

    for (e=0; e<ne; e++) {
      ierr = DMNetworkGetConnectedVertices(netdm,edges[e],&cone);CHKERRQ(ierr);
      printf(" coupling vfrom %d --> %d vto\n",cone[0],cone[1]);

      /* vfrom */
      ierr = DMNetworkGetComponent(netdm,cone[0],0,NULL,(void**)&cjunction[0],NULL);CHKERRQ(ierr);
      nin                = cjunction[0]->nin;
      nout               = cjunction[0]->nout;
      cjunction[0]->type = JUNCTION;

      /* vto */
      ierr = DMNetworkGetComponent(netdm,cone[1],0,NULL,(void**)&cjunction[1],NULL);CHKERRQ(ierr);
      nin               += cjunction[1]->nin;
      nout              += cjunction[1]->nout;
      cjunction[1]->type = JUNCTION;

      cjunction[0]->nin  = cjunction[1]->nin  = nin;
      cjunction[0]->nout = cjunction[1]->nout = nout;
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Input:
    subnetid -
    washCase -
    filename -
    rrank - process[rrank] reads input data
  Output:
    wash -
 */
PetscErrorCode WashAddSubnet(PetscInt subnetid, PetscInt washCase, const char filename[], PetscMPIInt rrank, Wash wash)
{
  PetscErrorCode ierr;
  PetscInt       nedges;
  PetscMPIInt    rank;
  PetscInt       i, numVertices, numEdges, numVariables, k, v;
  PetscInt      *edgelist;
  Junction       junctions   = NULL;
  River          rivers      = NULL;
  Pump           pumps       = NULL;
  PetscBool      test_mscale = wash->test_mscale, flg;
  RiverField     xmin, xmax;
  MPI_Comm       comm      = wash->comm;
  WATERDATA     *waterdata = NULL;
  WashSubnet     subnet    = (WashSubnet)wash->subnet[subnetid];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  if (rank != rrank) {
    subnet->nedge    = 0;
    subnet->nvertex  = 0;
    subnet->edgelist = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  wash->caseid    = washCase;
  wash->subcaseid = 3;

  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;

  xmax.q = 10.0;
  xmax.h = 0.0;
  xmin.q = -xmax.q;
  xmin.h = 0.0;

  /* Set global number of edges and vertices */
  /*---------------------------------------- */
  switch (washCase) {
  case -1:
    ierr = PetscNew(&waterdata);
    CHKERRQ(ierr);

    ierr = WaterReadData(waterdata, filename);
    CHKERRQ(ierr);
    ierr = PetscCalloc1(2 * waterdata->nedge, &edgelist);
    CHKERRQ(ierr);
    ierr = GetListofEdges_Water(waterdata, edgelist);
    CHKERRQ(ierr);

    numEdges    = waterdata->nedge;   /* npipe + npump */
    numVertices = waterdata->nvertex; /* njunction + nreservoir + ntank */

    /* Add network components */
    /*------------------------*/
    ierr = PetscCalloc3(numVertices, &junctions, waterdata->npipe, &rivers, waterdata->npump, &pumps);
    CHKERRQ(ierr);

    /* vertex */
    for (i = 0; i < numVertices; i++) {
      junctions[i].id = i;

      /* set physics id */
      junctions[i].id_phy = waterdata->vertex[i].id;

      /* Set junction type */
      junctions[i].type = waterdata->vertex[i].type;

      /* set elevation */
      if (junctions[i].type == JUNCTION) {
        junctions[i].elev = waterdata->vertex[i].elev;
      } else { /* Boundary */
        if (junctions[i].type == RESERVOIR) {
          junctions[i].elev = waterdata->vertex[i].elev;
        } else if (junctions[i].type == TANK) {
          junctions[i].elev = waterdata->vertex[i].elev;
        } else if (junctions[i].type == INFLOW) {
          junctions[i].elev = waterdata->vertex[i].elev;
        } else if (junctions[i].type == STAGE) {
          junctions[i].elev = waterdata->vertex[i].elev;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary not done yet");
      }

      /* Set Boundary type */
      if (junctions[i].type == RESERVOIR) {
        junctions[i].reservoir.head = waterdata->vertex[i].res.head;
        junctions[i].btype          = H;
        junctions[i].bval.h         = junctions[i].reservoir.head;

      } else if (junctions[i].type == TANK) {
        junctions[i].tank.head = waterdata->vertex[i].tank.head;
        junctions[i].btype     = H;
        junctions[i].bval.h    = junctions[i].tank.head;

      } else if (junctions[i].type == INFLOW) {
        junctions[i].inflow.flow = waterdata->vertex[i].inflow.flow;
        junctions[i].btype       = Q;
        junctions[i].bval.q      = junctions[i].inflow.flow;

      } else if (junctions[i].type == STAGE) {
        junctions[i].stage.head = waterdata->vertex[i].stage.head;
        junctions[i].btype      = H;
        junctions[i].bval.h     = junctions[i].stage.head;

      } else if (junctions[i].type != JUNCTION) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary not done yet");

      /* compute xmin xmax for graphic display */
      if (junctions[i].btype == Q) {
        if (junctions[i].bval.q > xmax.q) {
          xmax.q = junctions[i].bval.q;
        } else if (junctions[i].bval.q < xmin.q) {
          xmin.q = junctions[i].bval.q;
        }
      } else if (junctions[i].btype == H) {
        if (junctions[i].bval.h > xmax.h) {
          xmax.h = junctions[i].bval.h;
        } else if (junctions[i].bval.h < xmin.h) {
          xmin.h = junctions[i].bval.h;
        }
      }
    }
    //printf("QMin/Max: %g, %g; HMin/Max %g, %g\n",xmin.q,xmax.q,xmin.h,xmax.h);

    /* edge: river */
    numVariables = 0; /* number of variables */
    for (i = 0; i < waterdata->npipe; i++) {
      rivers[i].id        = i;
      rivers[i].id_phy    = waterdata->edge[i].pipe.id;
      rivers[i].fr_phy    = waterdata->edge[i].pipe.node1;
      rivers[i].to_phy    = waterdata->edge[i].pipe.node2;
      rivers[i].length    = waterdata->edge[i].pipe.length;
      rivers[i].width     = waterdata->edge[i].pipe.width;
      rivers[i].roughness = waterdata->edge[i].pipe.roughness;
      rivers[i].slope     = waterdata->edge[i].pipe.slope;
      rivers[i].q0        = waterdata->edge[i].pipe.qInitial;
      rivers[i].h0        = waterdata->edge[i].pipe.hInitial;
      ierr                = RiverSetNumCells(&rivers[i], 0.1);
      CHKERRQ(ierr);
      //printf("river %d, fr_node %d,to_node %d,length %g, width %g, roughness %g, slope %g, ncells %d\n",rivers[i].id_phy,rivers[i].fr_phy,rivers[i].to_phy,rivers[i].length,rivers[i].width,rivers[i].roughness,rivers[i].slope,rivers[i].ncells);
      numVariables += (rivers[i].ncells * 2);
    }

    /* edge: pump */
    for (k = 0; k < waterdata->npump; k++) {
      pumps[i - waterdata->npipe].id = i;
      v                              = edgelist[2 * i];

      PetscInt vto                       = edgelist[2 * i + 1];
      pumps[i - waterdata->npipe].to_tag = vto;
      junctions[vto].type                = junctions[v].type; /* downstream vertex takes same type as upstream vertex type */

      junctions[vto].btype  = junctions[v].btype;
      junctions[vto].bval.q = junctions[v].bval.q;
      junctions[vto].bval.h = junctions[v].bval.h;

      pumps[i - waterdata->npipe].id_phy = waterdata->edge[i].pump.id;
      pumps[i - waterdata->npipe].fr_phy = waterdata->edge[i].pump.node1;
      pumps[i - waterdata->npipe].to_phy = waterdata->edge[i].pump.node2;
      //printf(" pump %d, fr_node %d,to_node %d\n",pumps[i-waterdata->npipe].id_phy,pumps[i-waterdata->npipe].fr_phy,pumps[i-waterdata->npipe].to_phy);
      i++;
    }

    /* Count junction.nin and nout */
    /*-----------------------------*/
    for (i = 0; i < numVertices; i++) {
      //printf("%d -> %d\n",edgelist[2*i],edgelist[2*i+1]);
      junctions[i].nin  = 0;
      junctions[i].nout = 0;
    }
    for (i = 0; i < waterdata->nedge; i++) {
      v = edgelist[2 * i];
      junctions[v].nout++;
      v = edgelist[2 * i + 1];
      junctions[v].nin++;
    }

    ierr = PetscFree(waterdata->vertex);
    CHKERRQ(ierr);
    ierr = PetscFree(waterdata->edge);
    CHKERRQ(ierr);

    numVariables += 2 * numVertices;
    ierr = PetscPrintf(PETSC_COMM_SELF, "...Loading case file is done, ...numEdges %d, numVertices %d river numVariables %d\n", numEdges, numVertices, numVariables);
    CHKERRQ(ierr);
    break;
  case 0:
    /* washCase 0: */
    /* ==================================================
    (INFLOW) v0 --E0--> v1--E1--> v2 --E2-->v3 (RESERVOIR)
    ===================================================== */
    wash->nedge   = 0;
    wash->nvertex = 0;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;

    nedges = 3;
    if (test_mscale) { nedges = 2; }
    ierr = PetscOptionsGetInt(NULL, NULL, "-nedges", &nedges, NULL);
    CHKERRQ(ierr);
    wash->nedge   = nedges;
    wash->nvertex = nedges + 1;

    PetscReal dt;
    numVertices = wash->nvertex;
    numEdges    = wash->nedge;

    ierr = PetscCalloc1(2 * numEdges, &edgelist);
    CHKERRQ(ierr);
    for (i = 0; i < numEdges; i++) {
      edgelist[2 * i]     = i;
      edgelist[2 * i + 1] = i + 1;
    }

    /* Add network components */
    /*------------------------*/
    ierr = PetscCalloc2(numVertices, &junctions, numEdges, &rivers);
    CHKERRQ(ierr);
    /* vertex */
    for (i = 0; i < numVertices; i++) {
      junctions[i].id = i;

      /* Set boundary type */
      junctions[i].type = JUNCTION; /* By default*/

      /* Set number of into and from supporting edges */
      junctions[i].nin  = 1;
      junctions[i].nout = 1;
    }
    junctions[0].type        = INFLOW;
    junctions[0].inflow.flow = 1.0; /* Qus */
    junctions[0].btype       = Q;
    junctions[0].bval.q      = junctions[0].inflow.flow;
    xmax.q                   = junctions[0].inflow.flow;

    junctions[0].nin                = 0;
    junctions[numVertices - 1].nout = 0;

    junctions[numVertices - 1].type           = RESERVOIR;
    junctions[numVertices - 1].reservoir.head = 1.0; /* Hds */
    junctions[numVertices - 1].btype          = H;
    junctions[numVertices - 1].bval.h         = junctions[numVertices - 1].reservoir.head;
    xmax.h                                    = junctions[numVertices - 1].reservoir.head;

    /* edge */
    for (i = 0; i < numEdges; i++) {
      rivers[i].id = i;

      if (i == 0) {
        rivers[i].length = 5.0;
        dt               = 0.1;
        ierr             = RiverSetNumCells(&rivers[i], dt);
        CHKERRQ(ierr);
      } else {
        if (test_mscale) {
          rivers[i].length = 0.5;
          dt               = 0.01;
        } else {
          rivers[i].length = 5.0;
          dt               = 0.1;
        }
        ierr = RiverSetNumCells(&rivers[i], dt);
        CHKERRQ(ierr);
      }
      //printf("river %d, ncells %d, approx dt %g\n",i,rivers[i].ncells,dt);
    }

    xmax.h = 4.0;
    break;
  case 1:
    /* washCase 1: */
    /* ==========================
                v2 (RESERVOIR)
                ^
                |
               E2
                |
    v0 --E0--> v3--E1--> v1
    (INFLOW)           (RESERVOIR)
    =============================  */
    nedges        = 3;
    wash->nedge   = nedges;
    wash->nvertex = nedges + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = wash->nvertex;
    numEdges    = wash->nedge;

    ierr = PetscCalloc1(2 * numEdges, &edgelist);
    CHKERRQ(ierr);
    edgelist[0] = 0;
    edgelist[1] = 3; /* edge[0] */
    edgelist[2] = 3;
    edgelist[3] = 1; /* edge[1] */
    edgelist[4] = 3;
    edgelist[5] = 2; /* edge[2] */

    /* Add network components */
    /*------------------------*/
    ierr = PetscCalloc2(numVertices, &junctions, numEdges, &rivers);
    CHKERRQ(ierr);
    /* vertex */
    for (i = 0; i < numVertices; i++) {
      junctions[i].id = i;

      /* Set GPS data */
      junctions[i].latitude  = 0.0;
      junctions[i].longitude = 0.0;
    }

    junctions[0].type        = INFLOW;
    junctions[0].inflow.flow = 1.0; /* Qus */
    junctions[0].btype       = Q;
    junctions[0].bval.q      = junctions[0].inflow.flow;

    junctions[0].nin  = 0;
    junctions[0].nout = 1;
    xmax.q            = junctions[0].bval.q;

    junctions[1].type           = RESERVOIR;
    junctions[1].reservoir.head = 1.0; /* Hds */
    junctions[1].btype          = H;
    junctions[1].bval.h         = junctions[1].reservoir.head;

    junctions[1].nin  = 1;
    junctions[1].nout = 0;
    xmax.h            = junctions[1].bval.h;

    junctions[2].type           = RESERVOIR;
    junctions[2].reservoir.head = 1.0; /* Hds */
    junctions[2].btype          = H;
    junctions[2].bval.h         = junctions[2].reservoir.head;

    junctions[2].nin  = 1;
    junctions[2].nout = 0;

    junctions[3].type = JUNCTION;
    junctions[3].nin  = 1;
    junctions[3].nout = 2;

    /* edge */
    for (i = 0; i < numEdges; i++) {
      rivers[i].id = i;

      if (i == 0) {
        rivers[i].length = 5.0;
      } else {
        rivers[i].length = 2.5;
      }
      ierr = RiverSetNumCells(&rivers[i], 0.1);
      CHKERRQ(ierr);
      printf("river %d, ncells %d\n", i, rivers[i].ncells);
    }

    xmax.h = 3.0;
    break;
  case 2:
    /* washCase 2: */
    /* ==========================
       (INFLOW)v2--> E2
                     |
          v0 --E0--> v3--E1--> v1
       (INFLOW)              (RESERVOIR)
    =============================  */

    /* Set application parameters -- to be used in function evalutions */
    nedges        = 3;
    wash->nedge   = nedges;
    wash->nvertex = nedges + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = wash->nvertex;
    numEdges    = wash->nedge;

    ierr = PetscCalloc1(2 * numEdges, &edgelist);
    CHKERRQ(ierr);
    edgelist[0] = 0;
    edgelist[1] = 3; /* edge[0] */
    edgelist[2] = 3;
    edgelist[3] = 1; /* edge[1] */
    edgelist[4] = 2;
    edgelist[5] = 3; /* edge[2] */

    /* Add network components */
    /*------------------------*/
    ierr = PetscCalloc2(numVertices, &junctions, numEdges, &rivers);
    CHKERRQ(ierr);
    /* vertex */
    for (i = 0; i < numVertices; i++) {
      junctions[i].id = i;

      /* Set GPS data */
      junctions[i].latitude  = 0.0;
      junctions[i].longitude = 0.0;
    }

    junctions[0].type        = INFLOW;
    junctions[0].inflow.flow = 1.0; /* Qus */
    junctions[0].btype       = Q;
    junctions[0].bval.q      = junctions[0].inflow.flow;

    junctions[0].nin  = 0;
    junctions[0].nout = 1;

    junctions[1].type           = RESERVOIR;
    junctions[1].reservoir.head = 1.0; /* Hds */
    junctions[1].btype          = H;
    junctions[1].bval.h         = junctions[1].reservoir.head;

    junctions[1].nin  = 1;
    junctions[1].nout = 0;
    xmax.h            = junctions[1].reservoir.head;

    junctions[2].type        = INFLOW;
    junctions[2].inflow.flow = 1.0; /* Qus */
    junctions[2].btype       = Q;
    junctions[2].bval.q      = junctions[2].inflow.flow;

    junctions[2].nin  = 0;
    junctions[2].nout = 1;
    xmax.q            = junctions[2].inflow.flow;

    junctions[3].type = JUNCTION;
    junctions[3].nin  = 2;
    junctions[3].nout = 1;

    /* edge */
    for (i = 0; i < numEdges; i++) {
      rivers[i].id = i;

      if (i == 0) {
        rivers[i].length = 5.0;
      } else {
        rivers[i].length = 2.5;
      }
      ierr = RiverSetNumCells(&rivers[i], 0.1);
      CHKERRQ(ierr);
      printf("river %d, ncells %d\n", i, rivers[i].ncells);
    }
    xmax.h = 3.0;
    break;
  case 3:
    /* washCase 3: */
    /* =================================================
    (RESERVOIR) v0--E1-->v1 (TANK)
    ==================================================== */
    numEdges    = 1;
    numVertices = 2;

    PetscInt subcase;
    ierr = PetscCalloc1(2, &edgelist);
    CHKERRQ(ierr);
    edgelist[0] = 0;
    edgelist[1] = 1;

    /* Add network components */
    /*------------------------*/
    ierr = PetscCalloc2(numVertices, &junctions, numEdges, &rivers);
    CHKERRQ(ierr);
    /* vertex */
    for (i = 0; i < numVertices; i++) {
      junctions[i].id = i;

      /* set elevation data */
      junctions[i].elev = 0.0;

      /* Set number of into and from supporting edges */
      junctions[i].nin  = 0;
      junctions[i].nout = 0;
    }
    junctions[0].type = RESERVOIR;
    junctions[1].type = TANK;
    junctions[0].nout = 1;
    junctions[1].nin  = 1;

    /* boundary test cases */
    subcase = 3;
    ierr    = PetscOptionsGetInt(NULL, NULL, "-subcase", &subcase, NULL);
    CHKERRQ(ierr);
    wash->subcaseid = subcase;
    printf("  subCase %d\n", subcase);
    switch (subcase) {
    case 1:
      junctions[0].reservoir.head          = 1.0;
      junctions[numVertices - 1].tank.head = 0.1;
      xmax.q                               = 0.5;
      break;
    case 2:
      junctions[0].reservoir.head          = 1.0;
      junctions[numVertices - 1].tank.head = 1.0;
      xmax.q                               = 1.0;
      break;
    case 3:
      junctions[0].reservoir.head          = 1.0;
      junctions[numVertices - 1].tank.head = 0.0;
      xmax.q                               = 0.15;
      break;
    case 4:
      junctions[0].reservoir.head          = 0.0;
      junctions[numVertices - 1].tank.head = 1.0;
      xmax.q                               = 0.15;
      break;
    case 5:
      junctions[0].reservoir.head          = 0.1;
      junctions[numVertices - 1].tank.head = 0.1;
      xmax.q                               = 0.1;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet");
    }
    junctions[0].btype                = H;
    junctions[0].bval.h               = junctions[0].reservoir.head;
    junctions[numVertices - 1].btype  = H;
    junctions[numVertices - 1].bval.h = junctions[numVertices - 1].tank.head;

    xmax.h = junctions[0].bval.h;
    if (junctions[numVertices - 1].bval.h > xmax.h) xmax.h = junctions[numVertices - 1].bval.h;

    /* edge */
    rivers[0].id     = 0;
    rivers[0].length = 50.0;
    rivers[0].ncells = 100;
    ierr             = PetscOptionsGetInt(NULL, NULL, "-ncells", &rivers[0].ncells, &flg);
    CHKERRQ(ierr);

    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet");
  }

  subnet->nedge    = numEdges;
  subnet->nvertex  = numVertices;
  subnet->edgelist = edgelist;
  subnet->junction = junctions;
  subnet->river    = rivers;
  subnet->pump     = pumps;
  if (waterdata) {
    subnet->nriver = waterdata->npipe;
    subnet->npump  = waterdata->npump;
    ierr           = PetscFree(waterdata);
    CHKERRQ(ierr);
  } else {
    subnet->nriver = numEdges;
    subnet->npump  = 0;
  }

  /* Set axis values for graphic dispaly */
  wash->QMax = 10. * xmax.q;
  wash->QMin = -wash->QMax;
  wash->HMax = 1.5 * xmax.h;
  wash->HMin = xmin.h - 0.1;
  //printf("Hmin/max %g %g\n",wash->HMin,wash->HMax);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashCreate(MPI_Comm comm, PetscInt nsubnet, PetscInt ncsubnet, Wash *wash_ptr)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  Wash           wash;
  PetscBool      test_mscale = PETSC_FALSE;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-test_mscale", &test_mscale, NULL);
  CHKERRQ(ierr);

  ierr = PetscNew(&wash);
  CHKERRQ(ierr);
  wash->comm = comm;
  *wash_ptr  = wash;

  ierr = PetscMalloc1(nsubnet, &wash->subnet);
  CHKERRQ(ierr);
  for (i = 0; i < nsubnet; i++) {
    ierr = PetscNew(&wash->subnet[i]);
    CHKERRQ(ierr);
  }
  wash->nsubnet     = nsubnet;
  wash->ncsubnet    = ncsubnet;
  wash->test_mscale = test_mscale;
  wash->nnodes_loc  = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
