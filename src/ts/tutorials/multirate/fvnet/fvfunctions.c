#include "fvnet.h"

PetscErrorCode PhysicsDestroy_SimpleFree_Net(void *vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode RiemannListAdd_Net(PetscFunctionList *flist,const char *name,RiemannFunction rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,rsolve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RiemannListFind_Net(PetscFunctionList flist,const char *name,RiemannFunction *rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,rsolve);CHKERRQ(ierr);
  if (!*rsolve) SETERRQ1(PETSC_COMM_SELF,1,"Riemann solver \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListAdd_Net(PetscFunctionList *flist,const char *name,ReconstructFunction r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListFind_Net(PetscFunctionList flist,const char *name,ReconstructFunction *r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,r);CHKERRQ(ierr);
  if (!*r) SETERRQ1(PETSC_COMM_SELF,1,"Reconstruction \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetCharacteristicLimit(FVNetwork fvnet,PetscScalar *uL,PetscScalar *uM,PetscScalar *uR)
{
    PetscErrorCode ierr; 
    PetscScalar    jmpL,jmpR,*cjmpL,*cjmpR,*uLtmp,*uRtmp,tmp;
    PetscInt       j,k,dof = fvnet->physics.dof;

    PetscFunctionBegin;
    /* Create characteristic jumps */
    ierr  = (*fvnet->physics.characteristic)(fvnet->physics.user,dof,uM,fvnet->R,fvnet->Rinv,fvnet->speeds);CHKERRQ(ierr);
    ierr  = PetscArrayzero(fvnet->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &fvnet->cjmpLR[0];
    cjmpR = &fvnet->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      jmpL = uM[j]-uL[j]; /* Conservative Jumps */
      jmpR = uR[j]-uM[j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += fvnet->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += fvnet->Rinv[k+j*dof]*jmpR;
      }
    }
    uLtmp = fvnet->uLR; 
    uRtmp = &fvnet->uLR[dof];
    /* Limit in the characteristic variables */
    fvnet->limit(cjmpL,cjmpR,fvnet->cslope,dof);
    /* Put back in conservative variables */ 
    for (j=0; j<dof; j++) {
      tmp = 0;
      for (k=0; k<dof; k++) tmp += fvnet->R[j+k*dof]*fvnet->cslope[k];
      uLtmp[j] = uM[j]-tmp/2;
      uRtmp[j] = uM[j]+tmp/2; 
    }
  PetscFunctionReturn(0);
}

PetscErrorCode FVNetRHS(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  FVNetwork      fvnet = (FVNetwork)ctx;    
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       v,e,vStart,vEnd,eStart,eEnd,vfrom,vto;
  PetscInt       offsetf,offset,nedges,nnodes,i,j,dof = fvnet->physics.dof;;
  const PetscInt *cone,*edges;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points (NOTE: This routine (and the others done elsewhere) need to be refactored) */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          for (i=0; i<nedges; i++) {
            e     = edges[i];
            ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
            ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
            ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
            vfrom = cone[0];
            vto   = cone[1];
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            if (v == vfrom) {
              for (j=0; j<dof; j++) {
                f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]); 
              }
            } else if (v == vto) {
              for (j=0; j<dof; j++) {
                nnodes = fvedge->nnodes;
                f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
              }
            }
          }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
            SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for(j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now ALL processors have the reconstruction data to compute the coupling flux */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        /* Now compute the coupling flux */
        junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
        for (i=0; i<junction->numedges; i++) {
          for (j=0; j<dof; j++) {
            f[offsetf+i*dof+j] = junction->flux[i*dof+j];
          }
        }
        break;
      case OUTFLOW: 
          /* Requires no new computation */
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
       4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h      = fvedge->h;
    nnodes = fvedge->nnodes;
    /* Update the vfrom vertex flux for this edge */
    for (j=0; j<dof; j++) {
      f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
    }
    /* Now reconstruct the value at the left cell of the 1/2 interface. */
    switch (junction->type) {
      case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
        for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
        }
        break;
      case OUTFLOW: 
        /* Ghost cell Reconstruction Technique: Repeat the last cell. */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
        uR = fvnet->uLR+dof; 
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = uR[j]; 
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
    /* Iterate through the interior interfaces of the fvedge */
    for (i=1; i<(nnodes - 1); i++) { 
      ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
      ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      for (j=0; j<dof; j++) {
        fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
      }
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for( j=0; j<dof; j++) {
        f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
        f[offset+i*dof+j]     += fvnet->flux[j]/h;
      }
    }
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    /* Now reconstruct the value at the 2nd to last interface. */
    switch (junction->type) {
      case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
        for (j=0; j<dof; j++) {
          fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
        }
        break;
      case OUTFLOW: 
        /* Ghost cell Reconstruction Technique: Repeat the last cell. */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
      f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
    }
    /* Update the vfrom vertex flux for this edge */
    for (j=0; j<dof; j++) {
      f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
    }
    /* We have now updated the rhs for all data for this edge. */
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Multirate Non-buffer RHS */
PetscErrorCode FVNetRHS_Multirate(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       i,j,k,ne,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist,*edgelist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;

  PetscFunctionBeginUser;
  // ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rhsctx->edgelist,&ne);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rhsctx->vtxlist,&nv);CHKERRQ(ierr);
  ierr = ISGetIndices(rhsctx->edgelist,&edgelist);CHKERRQ(ierr); 
  ierr = ISGetIndices(rhsctx->vtxlist,&vtxlist);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);

  /* Iterate through all marked vertices (including ghosts) and compute the flux/reconstruction data for the vertex. */
  for (k=0; k<nv; k++) {
    v = vtxlist[k];
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          if (v == vfrom) {
            for (j=0; j<dof; j++) {
              f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]); 
            }
          } else if(v == vto){
            for (j=0; j<dof; j++) {
              nnodes = fvedge->nnodes;
              f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
            }
          }
        }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Either take the data from the beginning of the edge or the end */
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now All processors have the reconstruction data to compute the coupling flux */
  for (k=0; k<nv; k++) {
    v = vtxlist[k];
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        /* Now compute the coupling flux */
        junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
        for (i=0; i<junction->numedges; i++) {
          for (j=0; j<dof; j++) {
            f[offsetf+i*dof+j] = junction->flux[i*dof+j];
          }
        }
        break;
      case OUTFLOW: 
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through the edges and update the cell data belonging to that edge */
  for (k=0; k<ne; k++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
    4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    e      = edgelist[k];
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h      = fvedge->h;
    nnodes = fvedge->nnodes;
    if (!fvedge->frombufferlvl) { /* Evaluate from buffer data */
      /* Update the vfrom vertex flux for this edge */
      for (j=0; j<dof; j++) {
        f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
      }
      /* Now reconstruct the value at the left cell of the 1/2 interface. */
      switch (junction->type) {
        case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
          for(j=0; j<dof; j++){
              fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
          }
          break;
      case OUTFLOW: 
          /* Ghost cell Reconstruction Technique: Repeat the last cell. */
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
          uR = fvnet->uLR+dof; 
          for (j=0; j<dof; j++){
            fvnet->uPlus[j] = uR[j]; 
          }
          break;
        default:
          SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
      } 
      /* Iterate through the interior interfaces of the fvedge up to the buffer width*/
      for (i=1; i<(bufferwidth-1); i++) {
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
          f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
          f[offset+i*dof+j]     += fvnet->flux[j]/h;
        }
      }
    }
    /* Manage the two cells next to the buffer interface */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-2)*dof],&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof]);CHKERRQ(ierr);
    if (!fvedge->frombufferlvl) {
      ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+(bufferwidth-2)*dof+j] -= fvnet->flux[j]/h;
        f[offset+(bufferwidth-1)*dof+j] += fvnet->flux[j]/h;
      }
    }
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    /* Here I assume that the nnodes > 2*bufferwidth */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof],&xarr[offset+(bufferwidth+1)*dof]);CHKERRQ(ierr);
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      if (!fvedge->frombufferlvl) f[offset+(bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
      f[offset+(bufferwidth)*dof+j] += fvnet->flux[j]/h;
    }
    /* Iterate through the interior interfaces of the fvedge that don't interact with 'buffer' data. This assumes bufferwidth>=1  */
    for (i=bufferwidth+1; i<(nnodes-bufferwidth); i++) {
      ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
      ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      for (j=0; j<dof; j++) {
        fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
      }
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
        f[offset+i*dof+j]     += fvnet->flux[j]/h;
      }
    }
    /* Now we manage the cell to the right of the to buffer interface. This assumes bufferwidth >= 2 (reasonable assumption) */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof],&xarr[offset+(nnodes-bufferwidth+1)*dof]);CHKERRQ(ierr);
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      f[offset+(nnodes-bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
      if (!fvedge->tobufferlvl) f[offset+(nnodes-bufferwidth)*dof+j] += fvnet->flux[j]/h;
    }
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    /* Iterate through the remaining cells */
    if (!fvedge->tobufferlvl) {
      for (i=nnodes-bufferwidth+1; i<(nnodes-1); i++) {
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
          f[offset+i*dof+j]     += fvnet->flux[j]/h;
        }
      }
      /* Now reconstruct the value at the 2nd to last interface */
      switch (junction->type) {
        case JUNCT:
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          for (j=0; j<dof; j++) {
            fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
          }
          break;
        case OUTFLOW: 
          /* Ghost cell Reconstruction Technique: Repeat the last cell. */
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
          break;
        default:
          SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
      } 
      ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
        f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
      }
      /* Update the vfrom vertex flux for this edge */
      for (j=0; j<dof; j++) {
        f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
      }
    }  
    /* We have now updated the rhs for all data for this edge. */    
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  /* Move Data into the expected format from the multirate ode  */
  if(!rhsctx->scatter) {
    /* build the scatter */
    ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&rhsctx->scatter);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Multirate buffer RHS */
PetscErrorCode FVNetRHS_Buffer(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       i,j,k,m,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;

  PetscFunctionBeginUser;
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all marked vertices (including ghosts) and compute the flux/reconstruction data for the vertex. */
  ierr = ISGetLocalSize(rhsctx->vtxlist,&nv);CHKERRQ(ierr);
  ierr = ISGetIndices(rhsctx->vtxlist,&vtxlist);CHKERRQ(ierr);
  for (k=0; k<nv; k++) {
    v    = vtxlist[k];
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        /* Reconstruct all local edge data points */
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          if (v == vfrom) {
            for (j=0; j<dof; j++) {
              f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]);
            }
          } else if (v == vto) {
            for (j=0; j<dof; j++) {
              nnodes = fvedge->nnodes;
              f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
            }
          }
        }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr  = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now All processors have the reconstruction data to compute the coupling flux */
  for (k=0; k<nv; k++) {
    v    = vtxlist[k];
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
       /* Now compute the coupling flux */
        junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
        for (i=0; i<junction->numedges; i++) {
          for (j=0; j<dof; j++) {
            f[offsetf+i*dof+j] = junction->flux[i*dof+j];
          }
        }
        break;
      case OUTFLOW: 
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through buffer vertices and update the buffer edges connected to it. */
  for (k=0; k<nv; k++) {
    v    = vtxlist[k]; 
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (m=0; m<nedges; m++) {
      /* Update the buffer regions */
      e      = edges[m];
      ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
      ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom  = cone[0];
      vto    = cone[1];
      ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
      h      = fvedge->h;
      nnodes = fvedge->nnodes;
      if (v==vfrom && fvedge->frombufferlvl) {
        /* Update the from buffer region of edge e */
        for (j=0; j<dof; j++) {
          f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
        }
        switch (junction->type) {
          case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            for (j=0; j<dof; j++) {
              fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
            }
            break;
          case OUTFLOW: 
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            uR = fvnet->uLR+dof; 
            for (j=0; j<dof; j++) {
              fvnet->uPlus[j] = uR[j]; 
            }
            break;
          default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        /* Iterate through the interior interfaces of the fvedge up to the buffer width*/
        for (i=1; i<(bufferwidth); i++) {
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
          ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
          for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
          }
          fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
          for (j=0; j<dof; j++) {
            f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
            f[offset+i*dof+j]     += fvnet->flux[j]/h;
          }
        }
        /* Manage the last cell in the buffer region */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof],&xarr[offset+(bufferwidth+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
        }
      } else if (v==vto && fvedge->tobufferlvl) {
        /* ||Update to buffer region of edge e||*/
         /* Compute reconstruction to the left of the to buffer interface */ 
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-2)*dof],&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof]);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        /* Now we manage the cell to the right of the to buffer interface. This assumes bufferwidth >= 2 (reasonable assumption) */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof],&xarr[offset+(nnodes-bufferwidth+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(nnodes-bufferwidth)*dof+j] += fvnet->flux[j]/h;
        } 
        /* Iterate through the remaining cells */
        for (i=nnodes-bufferwidth+1; i<(nnodes-1); i++) {
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
          ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
          for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
          }
          fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
          for (j=0; j<dof; j++) {
            f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
            f[offset+i*dof+j]     += fvnet->flux[j]/h;
          }
        }
        /* Now reconstruct the value at the 2nd to last interface. */
        switch(junction->type) {
          case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            for (j=0; j<dof; j++) {
              fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
            }
            break;
          case OUTFLOW: 
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
            break;
          default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
          f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
        }
        /* Update the vfrom vertex flux for this edge */
        for (j=0; j<dof; j++) {
          f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
        }
      }    
    }
    /* We have now updated the rhs for all data for this edge. */    
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  /* Move Data into the expected format from the multirate ode */
  if(!rhsctx->scatter) {
    /* build the scatter */
    ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&rhsctx->scatter);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Multirate Non-buffer RHS */
PetscErrorCode FVNetRHS_Multirate_SingleCoupleEval(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       i,j,k,ne,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist,*edgelist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;
  PetscBool      isghostv; 

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rhsctx->edgelist,&ne);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rhsctx->vtxlist,&nv);CHKERRQ(ierr);
  ierr = ISGetIndices(rhsctx->edgelist,&edgelist);CHKERRQ(ierr); 
  ierr = ISGetIndices(rhsctx->vtxlist,&vtxlist);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);

  /* Iterate through all marked vertices (including ghosts) and compute the flux/reconstruction data for the vertex. */
  for (k=0; k<nv; k++) {
    v = vtxlist[k];
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          if (v == vfrom) {
            for (j=0; j<dof; j++) {
              f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]); 
            }
          } else if(v == vto){
            for (j=0; j<dof; j++) {
              nnodes = fvedge->nnodes;
              f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
            }
          }
        }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Either take the data from the beginning of the edge or the end */
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now All processors have the reconstruction data to compute the coupling flux */
  /* Only compute the flux data on non-ghost vertices, this ensures that the flux computation is unique for each 
    vertex and only one vertex has to call the coupling flux function */
  for (k=0; k<nv; k++) {
    v = vtxlist[k];
    ierr = DMNetworkIsGhostVertex(fvnet->network,v,&isghostv);CHKERRQ(ierr);
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        if(!isghostv) {
          /* Now compute the coupling flux */
          junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = junction->flux[i*dof+j];
            }
          }
        } else {
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = 0.0;
            }
          }
        }
        break;
      case OUTFLOW: 
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    }
  }
    /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through the edges and update the cell data belonging to that edge */
  for (k=0; k<ne; k++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
    4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    e      = edgelist[k];
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h      = fvedge->h;
    nnodes = fvedge->nnodes;
    if (!fvedge->frombufferlvl) { /* Evaluate from buffer data */
      /* Update the vfrom vertex flux for this edge */
      for (j=0; j<dof; j++) {
        f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
      }
      /* Now reconstruct the value at the left cell of the 1/2 interface. */
      switch (junction->type) {
        case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
          for(j=0; j<dof; j++){
              fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
          }
          break;
      case OUTFLOW: 
          /* Ghost cell Reconstruction Technique: Repeat the last cell. */
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
          uR = fvnet->uLR+dof; 
          for (j=0; j<dof; j++){
            fvnet->uPlus[j] = uR[j]; 
          }
          break;
        default:
          SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
      } 
      /* Iterate through the interior interfaces of the fvedge up to the buffer width*/
      for (i=1; i<(bufferwidth-1); i++) {
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
          f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
          f[offset+i*dof+j]     += fvnet->flux[j]/h;
        }
      }
    }
    /* Manage the two cells next to the buffer interface */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-2)*dof],&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof]);CHKERRQ(ierr);
    if (!fvedge->frombufferlvl) {
      ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+(bufferwidth-2)*dof+j] -= fvnet->flux[j]/h;
        f[offset+(bufferwidth-1)*dof+j] += fvnet->flux[j]/h;
      }
    }
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    /* Here I assume that the nnodes > 2*bufferwidth */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof],&xarr[offset+(bufferwidth+1)*dof]);CHKERRQ(ierr);
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      if (!fvedge->frombufferlvl) f[offset+(bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
      f[offset+(bufferwidth)*dof+j] += fvnet->flux[j]/h;
    }
    /* Iterate through the interior interfaces of the fvedge that don't interact with 'buffer' data. This assumes bufferwidth>=1  */
    for (i=bufferwidth+1; i<(nnodes-bufferwidth); i++) {
      ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
      ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      for (j=0; j<dof; j++) {
        fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
      }
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
        f[offset+i*dof+j]     += fvnet->flux[j]/h;
      }
    }
    /* Now we manage the cell to the right of the to buffer interface. This assumes bufferwidth >= 2 (reasonable assumption) */
    ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof],&xarr[offset+(nnodes-bufferwidth+1)*dof]);CHKERRQ(ierr);
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
    }
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      f[offset+(nnodes-bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
      if (!fvedge->tobufferlvl) f[offset+(nnodes-bufferwidth)*dof+j] += fvnet->flux[j]/h;
    }
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    /* Iterate through the remaining cells */
    if (!fvedge->tobufferlvl) {
      for (i=nnodes-bufferwidth+1; i<(nnodes-1); i++) {
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
          f[offset+i*dof+j]     += fvnet->flux[j]/h;
        }
      }
      /* Now reconstruct the value at the 2nd to last interface */
      switch (junction->type) {
        case JUNCT:
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          for (j=0; j<dof; j++) {
            fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
          }
          break;
        case OUTFLOW: 
          /* Ghost cell Reconstruction Technique: Repeat the last cell. */
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
          break;
        default:
          SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
      } 
      ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for (j=0; j<dof; j++) {
        f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
        f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
      }
      /* Update the vfrom vertex flux for this edge */
      for (j=0; j<dof; j++) {
        f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
      }
    }  
    /* We have now updated the rhs for all data for this edge. */    
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  /* Move Data into the expected format from the multirate ode  */
  if(!rhsctx->scatter) {
    /* build the scatter */
    ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&rhsctx->scatter);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Multirate buffer RHS */
PetscErrorCode FVNetRHS_Buffer_SingleCoupleEval(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       i,j,k,m,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;
  PetscBool      isghostv; 

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all marked vertices (including ghosts) and compute the flux/reconstruction data for the vertex. */
  ierr = ISGetLocalSize(rhsctx->vtxlist,&nv);CHKERRQ(ierr);
  ierr = ISGetIndices(rhsctx->vtxlist,&vtxlist);CHKERRQ(ierr);
  for (k=0; k<nv; k++) {
    v    = vtxlist[k];
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        /* Reconstruct all local edge data points */
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          /* Hard coded 2 cell one-side reconstruction. To be improved */
          if (v == vfrom) {
            for (j=0; j<dof; j++) {
              f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]);
            }
          } else if (v == vto) {
            for (j=0; j<dof; j++) {
              nnodes = fvedge->nnodes;
              f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
            }
          }
        }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr  = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            /* place flux in the flux component */
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now All processors have the reconstruction data to compute the coupling flux */
  for (k=0; k<nv; k++) {
    v    = vtxlist[k];
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    ierr = DMNetworkIsGhostVertex(fvnet->network,v,&isghostv);CHKERRQ(ierr);
    
    switch (junction->type) {
      case JUNCT:
        if (!isghostv) {
        /* Now compute the coupling flux */
          junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = junction->flux[i*dof+j];
            }
          }
        } else {
           for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = 0.0;
            }
          }
        }
        break;
      case OUTFLOW: 
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
   /* Now communicate the flux data to all processors */
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through buffer vertices and update the buffer edges connected to it. */
  for (k=0; k<nv; k++) {
    v    = vtxlist[k]; 
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (m=0; m<nedges; m++) {
      /* Update the buffer regions */
      e      = edges[m];
      ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
      ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom  = cone[0];
      vto    = cone[1];
      ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
      h      = fvedge->h;
      nnodes = fvedge->nnodes;
      if (v==vfrom && fvedge->frombufferlvl) {
        /* Update the from buffer region of edge e */
        for (j=0; j<dof; j++) {
          f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
        }
        switch (junction->type) {
          case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            for (j=0; j<dof; j++) {
              fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
            }
            break;
          case OUTFLOW: 
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            uR = fvnet->uLR+dof; 
            for (j=0; j<dof; j++) {
              fvnet->uPlus[j] = uR[j]; 
            }
            break;
          default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        /* Iterate through the interior interfaces of the fvedge up to the buffer width*/
        for (i=1; i<(bufferwidth); i++) {
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
          ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
          for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
          }
          fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
          for (j=0; j<dof; j++) {
            f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
            f[offset+i*dof+j]     += fvnet->flux[j]/h;
          }
        }
        /* Manage the last cell in the buffer region */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(bufferwidth-1)*dof],&xarr[offset+(bufferwidth)*dof],&xarr[offset+(bufferwidth+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
        }
      } else if (v==vto && fvedge->tobufferlvl) {
        /* ||Update to buffer region of edge e||*/
         /* Compute reconstruction to the left of the to buffer interface */ 
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-2)*dof],&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof]);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        /* Now we manage the cell to the right of the to buffer interface. This assumes bufferwidth >= 2 (reasonable assumption) */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-bufferwidth-1)*dof],&xarr[offset+(nnodes-bufferwidth)*dof],&xarr[offset+(nnodes-bufferwidth+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(nnodes-bufferwidth)*dof+j] += fvnet->flux[j]/h;
        } 
        /* Iterate through the remaining cells */
        for (i=nnodes-bufferwidth+1; i<(nnodes-1); i++) {
          ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
          ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
          for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
          }
          fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
          for (j=0; j<dof; j++) {
            f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
            f[offset+i*dof+j]     += fvnet->flux[j]/h;
          }
        }
        /* Now reconstruct the value at the 2nd to last interface. */
        switch(junction->type) {
          case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            for (j=0; j<dof; j++) {
              fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
            }
            break;
          case OUTFLOW: 
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
            break;
          default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        ierr    = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
          f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
        }
        /* Update the vfrom vertex flux for this edge */
        for (j=0; j<dof; j++) {
          f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
        }
      }    
    }
    /* We have now updated the rhs for all data for this edge. */    
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  /* Move Data into the expected format from the multirate ode */
  if(!rhsctx->scatter) {
    /* build the scatter */
    ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&rhsctx->scatter);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rhsctx->scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetRHS_SingleCoupleEval(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  FVNetwork      fvnet = (FVNetwork)ctx;    
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       v,e,vStart,vEnd,eStart,eEnd,vfrom,vto;
  PetscInt       offsetf,offset,nedges,nnodes,i,j,dof = fvnet->physics.dof;;
  const PetscInt *cone,*edges;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;
  PetscBool      isghostv;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points (NOTE: This routine (and the others done elsewhere) need to be refactored) */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          for (i=0; i<nedges; i++) {
            e     = edges[i];
            ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
            ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
            ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
            vfrom = cone[0];
            vto   = cone[1];
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            if (v == vfrom) {
              for (j=0; j<dof; j++) {
                f[offsetf+fvedge->offset_vfrom+j] = 0.5*(3*xarr[offset+j] - xarr[offset+dof+j]); 
              }
            } else if (v == vto) {
              for (j=0; j<dof; j++) {
                nnodes = fvedge->nnodes;
                f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
              }
            }
          }
        break;
      case OUTFLOW: 
        if (junction->numedges != 1) {
            SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
        } else {
          ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          e     = edges[0];
          ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
          vfrom = cone[0];
          vto   = cone[1];
          if (v == vfrom) {
            /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for (j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          } else if (v == vto) {
            nnodes = fvedge->nnodes;
              /* Ghost cell Reconstruction Technique: Repeat the last cell. */
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-1)*dof],&xarr[offset+(nnodes-2)*dof]);CHKERRQ(ierr);
            /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
              is just the cell average to the right of the boundary. */
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset+(nnodes-1)*dof],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for(j=0; j<dof; j++) {
              f[offsetf+j] = fvnet->flux[j];
            }
          }
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now ALL processors have the reconstruction data to compute the coupling flux */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkIsGhostVertex(fvnet->network,v,&isghostv);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        if(!isghostv) {
          /* Now compute the coupling flux */
          junction->couplingflux(fvnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = junction->flux[i*dof+j];
            }
          }
        } else {
           for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = 0.0;
            }
          }
        }
        break;
      case OUTFLOW: 
          /* Requires no new computation */
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
     /* Now communicate the flux data to all processors */
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
       4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h      = fvedge->h;
    nnodes = fvedge->nnodes;
    /* Update the vfrom vertex flux for this edge */
    for (j=0; j<dof; j++) {
      f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
    }
    /* Now reconstruct the value at the left cell of the 1/2 interface. */
    switch (junction->type) {
      case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
        for (j=0; j<dof; j++) {
            fvnet->uPlus[j] = 0.5*(xarr[offset+j] + xarr[offset+dof+j]);
        }
        break;
      case OUTFLOW: 
        /* Ghost cell Reconstruction Technique: Repeat the last cell. */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
        uR = fvnet->uLR+dof; 
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = uR[j]; 
        }
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
    /* Iterate through the interior interfaces of the fvedge */
    for (i=1; i<(nnodes - 1); i++) { 
      ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
      ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
      for (j=0; j<dof; j++) {
        fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
      }
      fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for( j=0; j<dof; j++) {
        f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
        f[offset+i*dof+j]     += fvnet->flux[j]/h;
      }
    }
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    /* Now reconstruct the value at the 2nd to last interface. */
    switch (junction->type) {
      case JUNCT:
        /* Hard coded 2 cell one-side reconstruction. To be improved */
        for (j=0; j<dof; j++) {
          fvnet->uLR[j] = 0.5*(xarr[offset+(nnodes-1)*dof+j] + xarr[offset+(nnodes-2)*dof+j]);
        }
        break;
      case OUTFLOW: 
        /* Ghost cell Reconstruction Technique: Repeat the last cell. */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+dof*(nnodes-2)],&xarr[offset+dof*(nnodes-1)],&xarr[offset+dof*(nnodes-1)]);CHKERRQ(ierr);
        break;
      default:
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
    ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
    fvedge->cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(fvedge->cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      f[offset+dof*(nnodes-2)+j] -= fvnet->flux[j]/h; 
      f[offset+dof*(nnodes-1)+j] += fvnet->flux[j]/h;
    }
    /* Update the vfrom vertex flux for this edge */
    for (j=0; j<dof; j++) {
      f[offset+dof*(nnodes-1)+j] -= f[fvedge->offset_vto+j+offsetf]/h;
    }
    /* We have now updated the rhs for all data for this edge. */
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}