/*
   Subroutines for WASH
*/

#include "wash.h"

PetscErrorCode WashPostSNESSetUpFieldsplit_River(SNES snes)
{
  PetscErrorCode ierr;
  DM             networkdm;
  KSP            ksp;
  PC             pc;
  MPI_Comm       comm;
  PetscBool      snes_fieldsplit = PETSC_TRUE;
  IS             juncIs, riverIs;
  PetscInt       e, eStart, eEnd, v, vStart, vEnd, kjunc, varoffset, type;
  PetscInt       numEdges_river, numVertices_nghost, nvar_junc, *junc_idx, localXsize;
  PetscBool      ghost;
  PCType         pctype;
  River          river;
  Wash           wash;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes, &comm);
  CHKERRQ(ierr);
  ierr = SNESGetApplicationContext(snes, &wash);
  CHKERRQ(ierr);

  ierr = SNESGetKSP(snes, &ksp);
  CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);
  CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);
  CHKERRQ(ierr);

  ierr = SNESGetDM(snes, &networkdm);
  CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
  CHKERRQ(ierr);
  ierr = VecGetLocalSize(wash->X, &localXsize);
  CHKERRQ(ierr);

  ierr = PCGetType(pc, &pctype);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &snes_fieldsplit);
  CHKERRQ(ierr);

  if (snes_fieldsplit) {
    numEdges_river = 0;
    for (e = eStart; e < eEnd; e++) {
      ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
      CHKERRQ(ierr);
      if (type != wash->keyRiver) continue;
      numEdges_river++;
    }

    numVertices_nghost = 0;
    for (v = vStart; v < vEnd; v++) {
      ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
      CHKERRQ(ierr);
      if (ghost) continue;
      numVertices_nghost++;
    }
    nvar_junc = 4 * numEdges_river + 2 * numVertices_nghost;

    ierr = PetscMalloc1(nvar_junc + 1, &junc_idx);
    CHKERRQ(ierr);

    kjunc = 0;
    for (e = eStart; e < eEnd; e++) { /* each edge has only one component, river */
      ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
      CHKERRQ(ierr);
      if (type != wash->keyRiver) continue;

      ierr = DMNetworkGetVariableGlobalOffset(networkdm, e, &varoffset);
      CHKERRQ(ierr);
      junc_idx[kjunc++] = varoffset;
      junc_idx[kjunc++] = varoffset + 1;
      junc_idx[kjunc++] = varoffset + 2 * river->ncells - 2;
      junc_idx[kjunc++] = varoffset + 2 * river->ncells - 1;
    }

    for (v = vStart; v < vEnd; v++) {
      ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
      CHKERRQ(ierr);
      if (ghost) continue;
      ierr = DMNetworkGetVariableGlobalOffset(networkdm, v, &varoffset);
      CHKERRQ(ierr);
      junc_idx[kjunc++] = varoffset;
      junc_idx[kjunc++] = varoffset + 1;
    }

    ierr = ISCreateGeneral(comm, nvar_junc, junc_idx, PETSC_COPY_VALUES, &juncIs);
    CHKERRQ(ierr);
    ierr = ISCreateStride(comm, localXsize, junc_idx[0], 1, &riverIs);
    CHKERRQ(ierr);
    ierr = ISSort(juncIs);
    CHKERRQ(ierr);
    ierr = ISSort(riverIs);
    CHKERRQ(ierr);

    ierr = PCFieldSplitSetBlockSize(pc, 2);
    CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc, "river", riverIs);
    CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc, "junction", juncIs);
    CHKERRQ(ierr);

    ierr = ISDestroy(&juncIs);
    CHKERRQ(ierr);
    ierr = ISDestroy(&riverIs);
    CHKERRQ(ierr);
    ierr = PetscFree(junc_idx);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashJuncSNESFuncFieldsplit(SNES snes, Vec X, Vec F, void *ctx)
{
  PetscErrorCode     ierr;
  Wash               wash;
  DM                 networkdm;
  Vec                localX, localF, Xold, localXold;
  const PetscScalar *xarr, *xoldarr;
  PetscScalar       *farr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &networkdm);
  CHKERRQ(ierr);
  ierr = SNESGetApplicationContext(snes, &wash);
  CHKERRQ(ierr);

  Xold   = wash->Xold;
  localX = wash->localX;
  localF = wash->localF;

  /* Set F and localF as zero */
  ierr = VecSet(F, 0.0);
  CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);
  CHKERRQ(ierr);

  /* update ghost values of locaX and localXold */
  ierr = DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecGetArray(localF, &farr);
  CHKERRQ(ierr);

  /* Initialize localF at junctions */
  PetscInt    type, vStart, vEnd, v, eStart, eEnd, e, varoffset;
  PetscBool   ghost;
  Junction    junction;
  River       river;
  RiverField *riverx, *riverxold, *riverf, *juncx, *juncf;

  ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
  CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
    CHKERRQ(ierr);
    if (ghost) continue;

    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, v, &varoffset);
    CHKERRQ(ierr);
    juncx = (RiverField *)(xarr + varoffset);
    juncf = (RiverField *)(farr + varoffset);

    /* junction->type == JUNCTION:
           juncf[0].q = -qJ + sum(qin) and  
           juncf[0].h =  qJ - sum(qout) -> sum(qin)-sum(qout)=0 */
    if (junction->type == JUNCTION) {
      juncf[0].q = -juncx[0].q;
      juncf[0].h = juncx[0].q;
    } else { /* localF = localX at non-ghost (ending) vertices */
      juncf[0].q = juncx[0].q;
      juncf[0].h = juncx[0].h;
      if (junction->type == INFLOW) {
        juncf[0].q = juncx[0].q - junction->inflow.qin;
      } else if (junction->type == RESERVOIR) {
        juncf[0].h = juncx[0].h - junction->reservoir.hres;
      } else if (junction->type == TANK) {
        juncf[0].h = juncx[0].h - junction->tank.elev;
      } else if (junction->type == STAGE) {
        juncf[0].h = juncx[0].h - junction->stage.head;
      }
    }
  }

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    ierr = DMNetworkGetVariableOffset(networkdm, e, &varoffset);
    CHKERRQ(ierr);
    riverxold = (RiverField *)(xoldarr + varoffset);
    riverx    = (RiverField *)(xarr + varoffset);
    riverf    = (RiverField *)(farr + varoffset);

    /* Evaluate interior cell values */
    PetscInt        i;
    const PetscInt *cone;
    PetscInt        vfrom, vto, offsetfrom, offsetto;
    PetscScalar     cL, cR;
    for (i = 1; i < river->ncells - 1; i++) {
      riverf[i].q = riverx[i].q;
      riverf[i].h = riverx[i].h;
    }

    /* Upstream Characteristics */
    cL          = PetscSqrtScalar(GRAV * riverxold[1].h);
    riverf[0].q = (riverx[0].q - riverxold[1].q) - (GRAV / cL) * (riverx[0].h - riverxold[1].h);

    /* Downstream Characteristics */
    PetscInt ncells      = river->ncells;
    cR                   = PetscSqrtScalar(GRAV * riverxold[ncells - 2].h);
    riverf[ncells - 1].q = (riverx[ncells - 1].q - riverxold[ncells - 2].q) + (GRAV / cR) * (riverx[ncells - 1].h - riverxold[ncells - 2].h);

    /* Evaluate boundary values from connected vertices */
    ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
    CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    ierr  = DMNetworkGetVariableOffset(networkdm, vfrom, &offsetfrom);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, vto, &offsetto);
    CHKERRQ(ierr);
    //printf("e %d, v%d - > v%d\n",e,vfrom,vto);

    /* Upstream boundary */
    juncx = (RiverField *)(xarr + offsetfrom);
    juncf = (RiverField *)(farr + offsetfrom);
    ierr  = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    if (junction->type == JUNCTION) {
      juncf[0].h -= riverx[0].q;
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == INFLOW) {
      riverf[0].h = riverx[0].q - juncx[0].q;
    } else if (junction->type == RESERVOIR) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == TANK) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == STAGE) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Upstream boundary type is not supported yet");

    /* Downstream boundary */
    juncx         = (RiverField *)(xarr + offsetto);
    juncf         = (RiverField *)(farr + offsetto);
    PetscInt nend = river->ncells - 1;
    ierr          = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction);
    CHKERRQ(ierr);

    if (junction->type == JUNCTION) {
      juncf[0].q += riverx[nend].q;
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == INFLOW) {
      riverf[nend].h = riverx[0].q - juncx[0].q;
    } else if (junction->type == RESERVOIR) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == TANK) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == STAGE) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Downstream boundary type is not supported yet");
  }

  ierr = VecRestoreArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(localF, &farr);
  CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);
  //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  PetscErrorCode     ierr;
  Wash               wash = (Wash)ctx;
  DM                 networkdm;
  Vec                localX, localXdot, localF, localXold, Xold;
  const PetscInt    *cone;
  PetscInt           vfrom, vto, offsetfrom, offsetto, type, varoffset;
  PetscInt           v, vStart, vEnd, e, eStart, eEnd, nend;
  PetscBool          ghost;
  PetscScalar       *farr;
  PetscReal          dt;
  River              river;
  RiverField        *riverx, *riverxold, *riverxdot, *riverf, *juncx, *juncxdot, *juncf;
  Junction           junction;
  MPI_Comm           comm;
  PetscMPIInt        rank, size;
  const PetscScalar *xarr, *xdotarr, *xoldarr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);
  CHKERRQ(ierr);

  localX    = wash->localX;
  localXdot = wash->localXdot;
  localF    = wash->localF;

  ierr = TSGetSolution(ts, &Xold); /* Note: we use Xold, thus an explicit scheme! */
  ierr = TSGetDM(ts, &networkdm);
  CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts, &dt);
  CHKERRQ(ierr);

  /* Set F and localF as zero */
  ierr = VecSet(F, 0.0);
  CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);
  CHKERRQ(ierr);

  /* update ghost values of locaX localXold and locaXdot */
  ierr = DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm, Xdot, INSERT_VALUES, localXdot);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xdot, INSERT_VALUES, localXdot);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXdot, &xdotarr);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecGetArray(localF, &farr);
  CHKERRQ(ierr);

  /* Initialize localF at junctions */
  ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
  CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
    CHKERRQ(ierr);
    if (ghost) continue;

    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, v, &varoffset);
    CHKERRQ(ierr);
    juncx    = (RiverField *)(xarr + varoffset);
    juncxdot = (RiverField *)(xdotarr + varoffset);
    juncf    = (RiverField *)(farr + varoffset);

    /* junction->type == JUNCTION:
           juncf[0].q = -qJ + sum(qin) and  
           juncf[0].h =  qJ - sum(qout) -> sum(qin)-sum(qout)=0 */
    if (junction->type == JUNCTION) {
      if (wash->caseid == 0 && wash->test_mscale) {
        juncf[0].q = juncxdot[0].q;
        juncf[0].h = juncxdot[0].h;
      } else {
        juncf[0].q = -juncx[0].q;
        juncf[0].h = juncx[0].q;
      }
    } else {
      /* localF = localX */
      juncf[0].q = juncx[0].q;
      juncf[0].h = juncx[0].h;
      if (junction->type == RESERVOIR) {
        juncf[0].h = juncx[0].h - junction->reservoir.hres;
      } else if (junction->type == TANK) {
        juncf[0].h = juncx[0].h - junction->tank.elev;
      } else if (junction->type == INFLOW) {
        juncf[0].q = juncx[0].q - junction->inflow.qin;
      } else if (junction->type == STAGE) {
        juncf[0].h = juncx[0].h - junction->stage.head;
      } //else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"junction type is not supported yet");
    }
  }

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type == wash->keyRiver) {
      ierr = DMNetworkGetVariableOffset(networkdm, e, &varoffset);
      CHKERRQ(ierr);
      riverxold = (RiverField *)(xoldarr + varoffset);
      riverx    = (RiverField *)(xarr + varoffset);
      riverxdot = (RiverField *)(xdotarr + varoffset);
      riverf    = (RiverField *)(farr + varoffset);

      /* Evaluate interior cell values for riverf[0].h, riverf[1].q, riverf[1].h,..., riverf[ncells-1].q */
      ierr = RiverIFunctionLocal(river, riverxold, riverx, riverxdot, riverf);
      CHKERRQ(ierr);

      /* Evaluate boundary values from connected vertices */
      ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
      CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];
      ierr  = DMNetworkGetVariableOffset(networkdm, vfrom, &offsetfrom);
      CHKERRQ(ierr);
      ierr = DMNetworkGetVariableOffset(networkdm, vto, &offsetto);
      CHKERRQ(ierr);
      //printf("e %d, v%d - > v%d\n",e,vfrom,vto);

      /* Upstream boundary */
      juncx = (RiverField *)(xarr + offsetfrom);
      juncf = (RiverField *)(farr + offsetfrom);
      ierr  = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction);
      CHKERRQ(ierr);

      if (junction->type == JUNCTION) {
        riverx = (RiverField *)(xarr + varoffset);
        if (wash->caseid == 0 && wash->test_mscale) {
          PetscReal dx = river->length / river->ncells;
          //printf("  e %d, vfrom %d, flux %g %g\n",e,vfrom,river->flux[0],river->flux[1]);
          juncf[0].q += river->flux[0] / dx;
          juncf[0].h += river->flux[1] / dx;
          riverf[0].q = riverx[0].q - juncx[0].q;
          riverf[0].h = riverx[0].h - juncx[0].h;
        } else {
          juncf[0].h -= riverx[0].q;
          riverf[0].q = riverx[0].h - juncx[0].h;
        }
      } else if (junction->type == RESERVOIR || junction->type == TANK || junction->type == NONE) {
        riverf[0].q = riverx[0].h - juncx[0].h;
      } else if (junction->type == INFLOW) {
        riverf[0].q = riverx[0].q - juncx[0].q;
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Upstream boundary type is not supported yet");

      /* Downstream boundary */
      juncx = (RiverField *)(xarr + offsetto);
      juncf = (RiverField *)(farr + offsetto);
      nend  = river->ncells - 1;
      ierr  = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction);
      CHKERRQ(ierr);

      if (junction->type == JUNCTION) {
        riverx = (RiverField *)(xarr + varoffset);
        if (wash->caseid == 0 && wash->test_mscale) {
          PetscReal dx = river->length / river->ncells;
          //printf("  e %d, vto %d, flux %g %g\n",e,vto,river->flux[2],river->flux[3]);
          juncf[0].q -= river->flux[2] / dx;
          juncf[0].h -= river->flux[3] / dx;
          riverf[nend].q = riverx[nend].q - juncx[0].q;
          riverf[nend].h = riverx[nend].h - juncx[0].h;
        } else {
          juncf[0].q += riverx[nend].q;
          riverf[nend].h = riverx[nend].h - juncx[0].h;
        }
      } else if (junction->type == RESERVOIR || junction->type == TANK || junction->type == STAGE) {
        riverf[nend].h = riverx[nend].h - juncx[0].h;
      } else if (junction->type == INFLOW) {
        riverf[nend].h = riverx[nend].q - juncx[0].q;
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Upstream boundary type is not supported yet");

    } else if (type == wash->keyPump) {
      RiverField *juncxfrom, *juncxto, *juncfto;
      /* Evaluate boundary values from connected vertices */
      ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
      CHKERRQ(ierr);
      vfrom = cone[0]; /* local ordering */
      vto   = cone[1];
      ierr  = DMNetworkIsGhostVertex(networkdm, vto, &ghost);
      CHKERRQ(ierr);
      if (!ghost) {
        ierr = DMNetworkGetVariableOffset(networkdm, vfrom, &offsetfrom);
        CHKERRQ(ierr);
        ierr = DMNetworkGetVariableOffset(networkdm, vto, &offsetto);
        CHKERRQ(ierr);

        juncxfrom = (RiverField *)(xarr + offsetfrom);
        juncxto   = (RiverField *)(xarr + offsetto);
        juncfto   = (RiverField *)(farr + offsetto);

        /* downstream vertex takes same type as upstream vertex type, see WashNetworkCreate_River() */
        juncfto[0].q = juncxto[0].q - juncxfrom[0].q;
        juncfto[0].h = juncxto[0].h - juncxfrom[0].h;
      }
    }
  }

  ierr = VecRestoreArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXdot, &xdotarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(localF, &farr);
  CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);
  //printf("\n t=%g, F:\n",t);
  //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashGetJuncLocalSize(Wash wash, PetscInt *nvar_junc)
{
  PetscInt       numEdges_river, e, v, type, Start, End, numVertices_nghost;
  PetscBool      ghost;
  River          river;
  PetscErrorCode ierr;
  DM             networkdm = wash->dm;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  numEdges_river = 0;
  for (e = Start; e < End; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;
    numEdges_river++;
  }

  ierr = DMNetworkGetVertexRange(networkdm, &Start, &End);
  CHKERRQ(ierr);
  numVertices_nghost = 0;
  for (v = Start; v < End; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
    CHKERRQ(ierr);
    if (ghost) continue;
    numVertices_nghost++;
  }
  *nvar_junc = 4 * numEdges_river + 2 * numVertices_nghost;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashJuncSNESFunc(SNES snes, Vec Xjunc, Vec Fjunc, void *ctx)
{
  PetscErrorCode     ierr;
  Wash               wash;
  DM                 networkdm;
  Vec                localX, localF, Xold, localXold, X, F;
  const PetscScalar *xarr, *xoldarr;
  PetscScalar       *farr;
  VecScatter         vscat;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);

  ierr = SNESGetApplicationContext(snes, &wash);
  CHKERRQ(ierr);
  networkdm = wash->dm;

  vscat = wash->vscat_junc;
  Xold  = wash->Xold;

  ierr = VecDuplicate(Xold, &X);
  CHKERRQ(ierr);
  ierr = VecDuplicate(Xold, &F);
  CHKERRQ(ierr);

  /* (1) Scatter Xjunc to X */
  ierr = VecSet(X, 0.0);
  CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat, Xjunc, X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat, Xjunc, X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);

  /*
  if (!rank) {printf("...X: \n");}
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  localX = wash->localX;
  localF = wash->localF;

  /* Set F and localF as zero */
  ierr = VecSet(F, 0.0);
  CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);
  CHKERRQ(ierr);

  /* (2) Update ghost values of locaX and localXold */
  ierr = DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX);
  CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xold, INSERT_VALUES, localXold);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecGetArray(localF, &farr);
  CHKERRQ(ierr);

  /* (3) Initialize localF at junctions */
  PetscInt    type, vStart, vEnd, v, eStart, eEnd, e, varoffset;
  PetscBool   ghost;
  Junction    junction;
  River       river;
  RiverField *riverx, *riverxold, *riverf, *juncx, *juncf;

  ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
  CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
    CHKERRQ(ierr);
    if (ghost) continue;

    ierr = DMNetworkGetComponent(networkdm, v, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, v, &varoffset);
    CHKERRQ(ierr);
    juncx = (RiverField *)(xarr + varoffset);
    juncf = (RiverField *)(farr + varoffset);

    /* junction->type == JUNCTION:
           juncf[0].q = -qJ + sum(qin) and  
           juncf[0].h =  qJ - sum(qout) -> sum(qin)-sum(qout)=0 */
    if (junction->type == JUNCTION) {
      juncf[0].q = -juncx[0].q;
      juncf[0].h = juncx[0].q;
    } else { /* localF = localX at non-ghost (ending) vertices */
      juncf[0].q = juncx[0].q;
      juncf[0].h = juncx[0].h;
      if (junction->type == INFLOW) {
        juncf[0].q = juncx[0].q - junction->inflow.qin;
      } else if (junction->type == RESERVOIR) {
        juncf[0].h = juncx[0].h - junction->reservoir.hres;
      } else if (junction->type == TANK) {
        juncf[0].h = juncx[0].h - junction->tank.elev;
      } else if (junction->type == STAGE) {
        juncf[0].h = juncx[0].h - junction->stage.head;
      }
    }
  }

  /* (4) Evaluate localF at each river(edge) */
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    ierr = DMNetworkGetVariableOffset(networkdm, e, &varoffset);
    CHKERRQ(ierr);
    riverxold = (RiverField *)(xoldarr + varoffset);
    riverx    = (RiverField *)(xarr + varoffset);
    riverf    = (RiverField *)(farr + varoffset);

    const PetscInt *cone;
    PetscInt        vfrom, vto, offsetfrom, offsetto;
    PetscScalar     cL, cR;

    /* Upstream Characteristics */
    cL          = PetscSqrtScalar(GRAV * riverxold[1].h);
    riverf[0].q = (riverx[0].q - riverxold[1].q) - (GRAV / cL) * (riverx[0].h - riverxold[1].h);

    /* Downstream Characteristics */
    PetscInt ncells      = river->ncells;
    cR                   = PetscSqrtScalar(GRAV * riverxold[ncells - 2].h);
    riverf[ncells - 1].q = (riverx[ncells - 1].q - riverxold[ncells - 2].q) + (GRAV / cR) * (riverx[ncells - 1].h - riverxold[ncells - 2].h);

    /* Evaluate boundary values from connected vertices */
    ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
    CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    ierr  = DMNetworkGetVariableOffset(networkdm, vfrom, &offsetfrom);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, vto, &offsetto);
    CHKERRQ(ierr);
    //printf("e %d, v%d - > v%d\n",e,vfrom,vto);

    /* Upstream boundary */
    juncx = (RiverField *)(xarr + offsetfrom);
    juncf = (RiverField *)(farr + offsetfrom);
    ierr  = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    if (junction->type == JUNCTION) {
      juncf[0].h -= riverx[0].q;
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == INFLOW) {
      riverf[0].h = riverx[0].q - juncx[0].q;
    } else if (junction->type == RESERVOIR) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == TANK) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else if (junction->type == STAGE) {
      riverf[0].h = riverx[0].h - juncx[0].h;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Upstream boundary type is not supported yet");

    /* Downstream boundary */
    juncx         = (RiverField *)(xarr + offsetto);
    juncf         = (RiverField *)(farr + offsetto);
    PetscInt nend = river->ncells - 1;
    ierr          = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction);
    CHKERRQ(ierr);

    if (junction->type == JUNCTION) {
      juncf[0].q += riverx[nend].q;
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == INFLOW) {
      riverf[nend].h = riverx[0].q - juncx[0].q;
    } else if (junction->type == RESERVOIR) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == TANK) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else if (junction->type == STAGE) {
      riverf[nend].h = riverx[nend].h - juncx[0].h;
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Downstream boundary type is not supported yet");
  }

  ierr = VecRestoreArrayRead(localX, &xarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXold, &xoldarr);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(localF, &farr);
  CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm, localF, ADD_VALUES, F);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm, &localXold);
  CHKERRQ(ierr);

  /* (5) Scatter F to Fjunc */
  ierr = VecScatterBegin(vscat, F, Fjunc, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat, F, Fjunc, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  ierr = VecDestroy(&X);
  CHKERRQ(ierr);
  ierr = VecDestroy(&F);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WashPostSNESSetUp_River(SNES snes)
{
  PetscErrorCode  ierr;
  MPI_Comm        comm;
  Wash            wash;
  PetscInt        nvar_junc, kjunc, rows[2], cols[2], vfrom, vto, offsetfrom, offsetto;
  Vec             Xtmp, localXtmp;
  PetscScalar    *xtmp_arr, *zeros;
  DM              networkdm;
  PetscInt        rstart_junc, e, eStart, eEnd, v, vStart, vEnd, type, varoffset, rstart, *idx1;
  PetscMPIInt     rank;
  Vec             Xjunc;
  River           river;
  PetscBool       ghost;
  MatFDColoring   fdcoloring;
  MatColoring     coloring;
  ISColoring      iscoloring;
  Mat             Jac;
  IS              is1, is2;
  const PetscInt *cone;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  CHKERRQ(ierr);
  ierr = SNESGetApplicationContext(snes, &wash);
  CHKERRQ(ierr);
  wash->snes_junc = snes;

  networkdm = wash->dm;
  ierr      = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
  CHKERRQ(ierr);

  /* Get vectors */
  /* Xtmp: map used for building Jac: Xtmp[goffset] = global index of Xjunc */
  ierr = VecDuplicate(wash->X, &Xtmp);
  CHKERRQ(ierr);
  ierr = VecSet(Xtmp, -1.0);
  CHKERRQ(ierr);

  Xjunc = wash->Xjunc;
  ierr  = VecGetOwnershipRange(Xjunc, &rstart_junc, NULL);
  CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(wash->X, &rstart, NULL);
  CHKERRQ(ierr);

  /* Create Jacobian data structure */
  ierr = WashGetJuncLocalSize(wash, &nvar_junc);
  CHKERRQ(ierr);
  ierr = MatCreate(comm, &Jac);
  CHKERRQ(ierr);
  ierr = MatSetSizes(Jac, nvar_junc, nvar_junc, PETSC_DECIDE, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = MatSetUp(Jac);
  CHKERRQ(ierr);

  /* Create scat_junc; Set index map for building Jac: dmnetwork goffset/loffset -> global index of Xjunc */
  ierr = PetscCalloc2(nvar_junc, &idx1, 4, &zeros);
  CHKERRQ(ierr);
  ierr = VecGetArray(Xtmp, &xtmp_arr);
  CHKERRQ(ierr);

  kjunc = 0; /* local index! */
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableGlobalOffset(networkdm, e, &varoffset);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    rows[0] = kjunc + rstart_junc;
    rows[1] = kjunc + 1 + rstart_junc;
    ierr    = MatSetValues(Jac, 2, rows, 2, rows, zeros, INSERT_VALUES);
    CHKERRQ(ierr);

    xtmp_arr[varoffset - rstart] = kjunc + rstart_junc; /* local idx on X -> global idx on Xjunc */
    idx1[kjunc++]                = varoffset;

    xtmp_arr[varoffset + 1 - rstart] = kjunc + rstart_junc;
    idx1[kjunc++]                    = varoffset + 1;

    rows[0] = kjunc + rstart_junc;
    rows[1] = kjunc + 1 + rstart_junc;
    ierr    = MatSetValues(Jac, 2, rows, 2, rows, zeros, INSERT_VALUES);
    CHKERRQ(ierr);

    xtmp_arr[varoffset + 2 * river->ncells - 2 - rstart] = kjunc + rstart_junc;
    idx1[kjunc++]                                        = varoffset + 2 * river->ncells - 2;

    xtmp_arr[varoffset + 2 * river->ncells - 1 - rstart] = kjunc + rstart_junc;
    idx1[kjunc++]                                        = varoffset + 2 * river->ncells - 1;
  }

  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm, v, &ghost);
    CHKERRQ(ierr);
    if (ghost) continue;
    ierr = DMNetworkGetVariableGlobalOffset(networkdm, v, &varoffset);
    CHKERRQ(ierr);

    rows[0] = kjunc + rstart_junc;
    rows[1] = kjunc + 1 + rstart_junc;
    ierr    = MatSetValues(Jac, 2, rows, 2, rows, zeros, INSERT_VALUES);
    CHKERRQ(ierr);

    xtmp_arr[varoffset - rstart] = kjunc + rstart_junc;
    idx1[kjunc++]                = varoffset;

    xtmp_arr[varoffset + 1 - rstart] = kjunc + rstart_junc;
    idx1[kjunc++]                    = varoffset + 1;
  }
  ierr = VecRestoreArray(Xtmp, &xtmp_arr);
  CHKERRQ(ierr);

  /* localXtmp: localXtmp[loffset] = global index of Xjunc */
  ierr = DMGetLocalVector(networkdm, &localXtmp);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm, Xtmp, INSERT_VALUES, localXtmp);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm, Xtmp, INSERT_VALUES, localXtmp);
  CHKERRQ(ierr);

  /* Set connected vertices into Jac using localXtmp */
  ierr = VecGetArray(localXtmp, &xtmp_arr);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;
    ierr = DMNetworkGetVariableOffset(networkdm, e, &varoffset);
    CHKERRQ(ierr);

    ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
    CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    ierr  = DMNetworkGetVariableOffset(networkdm, vfrom, &offsetfrom);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, vto, &offsetto);
    CHKERRQ(ierr);
    //printf("e %d, vfrom %d, to %d\n",e,vfrom,vto);

    /* Up stream edge couples with vfrom */
    rows[0] = xtmp_arr[varoffset];
    rows[1] = xtmp_arr[varoffset] + 1;
    cols[0] = xtmp_arr[offsetfrom];
    cols[1] = xtmp_arr[offsetfrom] + 1;
    ierr    = MatSetValues(Jac, 2, rows, 2, cols, zeros, INSERT_VALUES);
    CHKERRQ(ierr);
    ierr = MatSetValues(Jac, 2, cols, 2, rows, zeros, INSERT_VALUES);
    CHKERRQ(ierr);

    /* Down stream edge couples with vto: */
    rows[0] = xtmp_arr[varoffset + 2 * river->ncells - 2];
    rows[1] = xtmp_arr[varoffset + 2 * river->ncells - 2] + 1;
    cols[0] = xtmp_arr[offsetto];
    cols[1] = xtmp_arr[offsetto] + 1;
    ierr    = MatSetValues(Jac, 2, rows, 2, cols, zeros, INSERT_VALUES);
    CHKERRQ(ierr);
    ierr = MatSetValues(Jac, 2, cols, 2, rows, zeros, INSERT_VALUES);
    CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(localXtmp, &xtmp_arr);
  CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  /* Create IS for vscat */
  ierr = ISCreateGeneral(comm, nvar_junc, idx1, PETSC_COPY_VALUES, &is1);
  CHKERRQ(ierr);
  ierr = ISCreateStride(comm, nvar_junc, rstart_junc, 1, &is2);
  CHKERRQ(ierr);

  ierr = VecScatterCreate(wash->X, is1, Xjunc, is2, &wash->vscat_junc);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is1);
  CHKERRQ(ierr);
  ierr = ISDestroy(&is2);
  CHKERRQ(ierr);
  ierr = PetscFree2(idx1, zeros);
  CHKERRQ(ierr);

  /* Create fdcoloring for Jac */
  ierr = MatColoringCreate(Jac, &coloring);
  CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring, MATCOLORINGSL);
  CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);
  CHKERRQ(ierr);
  ierr = MatColoringApply(coloring, &iscoloring);
  CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);
  CHKERRQ(ierr);

  ierr = MatFDColoringCreate(Jac, iscoloring, &fdcoloring);
  CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring, (PetscErrorCode(*)(void))WashJuncSNESFunc, snes);
  CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);
  CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(Jac, iscoloring, fdcoloring);
  CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, Jac, Jac, SNESComputeJacobianDefaultColor, fdcoloring);
  CHKERRQ(ierr);

  ierr = ISColoringDestroy(&iscoloring);
  CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&fdcoloring);
  CHKERRQ(ierr);
  ierr = MatDestroy(&Jac);
  CHKERRQ(ierr);
  ierr = VecDestroy(&Xtmp);
  CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm, &localXtmp);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

//---------------------------

/* 
   Set x[i] = 0.5*(x[i-1] + x[i+1]) for interior river points to enable Lax–Friedrichs scheme via TSEULER,
   writtern for experimenting '-ts_lax' in WashTSSetUp().

   Add following to WashTSSetUp:
   {
    PetscBool      poststage=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-ts_lax",&poststage,NULL);CHKERRQ(ierr);
    if (poststage) {
      // requires modification in PETSc TSStep_Euler:
      ierr = VecAYPX(update,ts->time_step,solution);CHKERRQ(ierr);
      ierr = TSPostStage(ts,ts->ptime,0,&solution);CHKERRQ(ierr);
      ierr = VecAYPX(update,ts->time_step,solution);CHKERRQ(ierr);
      ierr = TSSetPostStage(ts,TSWashPostStage);CHKERRQ(ierr);
    }
*/
PetscErrorCode TSWashPostStage(TS ts, PetscReal time, PetscInt stageindex, Vec *Y)
{
  PetscErrorCode     ierr;
  Vec                Xold;
  DM                 networkdm;
  Wash               wash;
  PetscInt           e, eStart, eEnd, type, varoffset, i, vfrom, vto, ncells;
  River              river;
  Junction           junction;
  RiverField        *riverx, xL, xR, *juncx;
  const PetscScalar *xarr;
  const PetscInt    *cone;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &networkdm);
  CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts, &wash);
  CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &Xold);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(Xold, &xarr);
  CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river);
    CHKERRQ(ierr);
    if (type != wash->keyRiver) continue;

    ierr = DMNetworkGetVariableOffset(networkdm, e, &varoffset);
    CHKERRQ(ierr);
    riverx = (RiverField *)(xarr + varoffset);
    ncells = river->ncells;

    /* save riverx[1] and riverx[ncells-2] to be used for averaging up and down stream boundary points */
    xL = riverx[1];
    xR = riverx[ncells - 2];

    /* Average interior river points */
    for (i = 1; i < river->ncells - 1; i++) {
      riverx[i].q = 0.5 * (riverx[i - 1].q + riverx[i + 1].q);
      riverx[i].h = 0.5 * (riverx[i - 1].h + riverx[i + 1].h);
    }

    /* Querry connected junctions */
    ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
    CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];

    /* Average upper stream */
    ierr = DMNetworkGetComponent(networkdm, vfrom, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, vfrom, &varoffset);
    CHKERRQ(ierr);
    juncx       = (RiverField *)(xarr + varoffset);
    riverx[0].q = 0.5 * (juncx[0].q + xL.q);
    riverx[0].h = 0.5 * (juncx[0].h + xL.h);

    /* Average down stream */
    ierr = DMNetworkGetComponent(networkdm, vto, 0, &type, (void **)&junction);
    CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm, vto, &varoffset);
    CHKERRQ(ierr);
    juncx                = (RiverField *)(xarr + varoffset);
    riverx[ncells - 1].q = 0.5 * (xR.q + juncx[0].q);
    riverx[ncells - 1].h = 0.5 * (xR.h + juncx[0].h);
  }
  ierr = VecRestoreArrayRead(Xold, &xarr);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------------------------
