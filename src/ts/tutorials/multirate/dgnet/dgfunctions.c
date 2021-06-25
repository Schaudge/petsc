#include "dgnet.h"

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
PetscReal innerprod_internal(PetscInt len,PetscReal *x  ,PetscReal *y) {
  PetscInt i; 
  PetscReal sum = 0.; 

  for(i=0; i<len; i++) {
    sum += x[i]*y[i];
  }
  return sum;
}

PetscErrorCode DGNetRHS(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  DGNetwork      fvnet = (DGNetwork)ctx;    
  PetscReal      h,maxspeed;
  PetscScalar    *f,*uR,*xarr;
  PetscInt       v,e,vStart,vEnd,eStart,eEnd,vfrom,vto, m = fvnet->basisorder+1;
  PetscInt       offsetf,offset,nedges,nnodes,i,j,dof = fvnet->physics.dof;;
  const PetscInt *cone,*edges;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  EdgeFE         fvedge; 
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
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
      for (i=0; i<nedges; i++) {
        e     = edges[i];
        ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
        ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        vfrom = cone[0];
        vto   = cone[1];
        if (v == vfrom) {
          /* left eval */
          for (j=0; j<dof; j++) {
            f[offsetf+fvedge->offset_vfrom+j] = innerprod_internal(m,)
          }
        } else if (v == vto) {
          for (j=0; j<dof; j++) {
            nnodes = fvedge->nnodes;
            f[offsetf+fvedge->offset_vto+j] = 0.5*(3*xarr[offset+(nnodes-1)*dof+j] - xarr[offset+(nnodes-2)*dof+j]); 
          }
        }
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
