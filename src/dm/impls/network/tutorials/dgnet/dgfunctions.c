#include "dgnet.h"
#include <stdio.h>

PetscErrorCode PhysicsDestroy_SimpleFree_Net(void *vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscReal evalboundary_internal(DGNetwork dgnet, PetscInt field,PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEvaL_bdry[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
static PetscReal evalquad_internal(DGNetwork dgnet, PetscInt field, PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEval[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
static PetscReal evalquadDer_internal(DGNetwork dgnet, PetscInt field, PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEvalD[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}

PetscErrorCode DGNetRHS_NETRSVERSION(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  DGNetwork      dgnet = (DGNetwork)ctx;    
  PetscReal      maxspeed,detJ,J,invJ,*numflux,*netflux,*errorest; 
  PetscScalar    *f,*xarr,*coeff; 
  PetscInt       v,e,c,vStart,vEnd,eStart,eEnd,vfrom,vto,cStart,cEnd,q,deg,ndeg,quadsize,tab,face,fStart,fEnd;
  PetscInt       offsetf,offset,nedges,i,j,dof = dgnet->physics.dof,field,fieldoff;
  const PetscInt *cone,*edges,*supp;
  Vec            localX = dgnet->localX,localF = dgnet->localF,Ftmp = dgnet->Ftmp; 
  EdgeFE         edgefe; 
  Junction       junction;
  PetscSection   section;
  const PetscReal *qweight;
  RiemannSolver   rs = dgnet->physics.rs; 
  FILE           *file; /* remove */
  char            filename[128];
  PetscBool       adaption;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(dgnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points (NOTE: This routine (and the others done elsewhere) need to be refactored) */
    ierr = DMNetworkGetLocalVecOffset(dgnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(dgnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    ierr = DMNetworkGetSupportingEdges(dgnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    for (i=0; i<nedges; i++) {
      e     = edges[i];
      ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
      ierr  = DMNetworkGetConnectedVertices(dgnet->network,e,&cone);CHKERRQ(ierr);
      ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
      /* DMPlex stuff here, get cell chart */
      ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
      ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];
      if (v == vfrom) {
        /* left eval */
        for (field=0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,cStart,field,&fieldoff);CHKERRQ(ierr);
          f[offsetf+edgefe->offset_vfrom+field] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
        }
      } else if (v == vto) {
        for (field=0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff);CHKERRQ(ierr);
          f[offsetf+edgefe->offset_vto+field] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        }
      }
    }
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dgnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dgnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now ALL processors have the evaluation data to compute the coupling flux */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(dgnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(dgnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    /* compute the coupling flux */
    ierr = NetRSEvaluate(junction->netrs,f+offsetf,junction->dir,&netflux,&errorest,&adaption);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(dgnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    /* move the following to a viewer routine for netrs */
    for (i=0; i<nedges; i++) {
      e     = edges[i];
      ierr = PetscSNPrintf(filename,128,"./output/v%ie%i.txt",v,e);CHKERRQ(ierr); 
      ierr = PetscFOpen(PETSC_COMM_SELF,filename,"a",&file);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"%e, %e ,%i \n",time,errorest[i],adaption);CHKERRQ(ierr);
      ierr = PetscFClose(PETSC_COMM_SELF,file);CHKERRQ(ierr);
    }

    for (i=0; i<junction->numedges; i++) {
      for (j=0; j<dof; j++) {
          f[offsetf+i*dof+j] = netflux[i*dof+j];
      }
    }
  }
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight);CHKERRQ(ierr);
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      for(q=0; q<quadsize; q++) {
        for(field = 0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
          coeff = xarr+offset+fieldoff;
          dgnet->pteval[field] = evalquad_internal(dgnet,field,q,coeff);
        }
        dgnet->physics.flux((void*)dgnet->physics.user,dgnet->pteval,dgnet->fluxeval+q*dof);
      }
      /* Now we can compute quadrature for each integral for each field */
      for(field = 0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          for (q = 0; q<quadsize; q++) {
            *coeff += qweight[q]*dgnet->fluxeval[q*dof+field]*dgnet->LegEvalD[tab][ndeg*q+deg]; 
          }
        }
      }
    }
    /* Flux Time !!! :) */ 
    /* update the boundary cells first, (cstart,cEnd) as their fluxes are coupling fluxes */
    ierr   = DMNetworkGetConnectedVertices(dgnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    /*cStart cell */
    ierr   = DMNetworkGetComponent(dgnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(dgnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    /* Update the vfrom vertex flux for this edge */
    for (field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cStart,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff += f[edgefe->offset_vfrom+field+offsetf]*dgnet->LegEvaL_bdry[tab][deg];
      }
    }
    /* cEnd cell */
    ierr   = DMNetworkGetComponent(dgnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(dgnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    /* Update the vfrom vertex flux for this edge */
    for (field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff -= f[edgefe->offset_vto+field+offsetf]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
      }
    }
    /* 2) Then iterate through the flux updates */
    /* we iterate through the 1 codim cells (faces) skipping the first and last to compute the numerical fluxes and update the resulting cells coefficients */
    ierr  = DMPlexGetHeightStratum(edgefe->dm,1,&fStart,&fEnd);CHKERRQ(ierr);
    for(face=fStart+1; face<fEnd-1; face++) {
      /* WE ASSUME 1D HERE WITH SUPPORT SIZE OF 2 !!!! */
      ierr = DMPlexGetSupport(edgefe->dm,face,&supp);CHKERRQ(ierr);
      /* evaluate at the face */
      for(field = 0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff);CHKERRQ(ierr);
        dgnet->uLR[field] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        ierr = PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff);CHKERRQ(ierr);
        dgnet->uLR[field+dof] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
      }
      ierr = RiemannSolverEvaluate(rs,dgnet->uLR,dgnet->uLR+dof,&numflux,&maxspeed);CHKERRQ(ierr);
      /* Update coefficents with the numerical flux */
      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff -= numflux[field]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
        }
      }
      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff += numflux[field]*dgnet->LegEvaL_bdry[tab][deg];
        }
      }
    }
    /* Normalization loop */
    for (c=cStart; c<cEnd; c++) {
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      for(field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff  = f+offset+fieldoff+deg;
          *coeff *= dgnet->Leg_L2[tab][deg]/detJ; /* Inverting the Mass matrix. To be refactored later 
          with arbitrary basis */
        }
      }
    }
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dgnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DGNetworkProject(DGNetwork dgnet,Vec X0,PetscReal t) 
{
  PetscErrorCode ierr;
  PetscInt       type,offset,e,eStart,eEnd,dof = dgnet->physics.dof;
  PetscInt       c,cStart,cEnd,field,edgeid,deg,ndeg,tab,fieldoff,quadsize,q;
  PetscScalar    *xarr,*coeff;
  EdgeFE         edgefe;
  Vec            localX = dgnet->localX;
  PetscReal      J,invJ,detJ,v0;
  const PetscReal *qpoint,*qweight;
  PetscSection   section;
  
  PetscFunctionBegin;
  ierr = VecZeroEntries(localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,&qpoint,&qweight);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,&type,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr  = DMNetworkGetGlobalEdgeIndex(dgnet->network,e,&edgeid);CHKERRQ(ierr);
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    for (c=cStart; c<cEnd; c++) {
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,&v0,&J,&invJ,&detJ);CHKERRQ(ierr);
      /* We can compute points in real space by Jx + v0, the affine transformation */
      for(field=0; field<dof; field++){
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff  = xarr+offset+fieldoff+deg;
          for (q=0; q< quadsize; q++) {
            /* Evaluate the sample function at the quadrature point */
            dgnet->physics.samplenetwork((void*)dgnet->physics.user,dgnet->initial,t,qpoint[q]*J+v0,dgnet->pteval,edgeid);
            *coeff += qweight[q]*dgnet->pteval[field]*dgnet->LegEval[tab][ndeg*q+deg]; 
          }
          *coeff *= dgnet->Leg_L2[tab][deg];
        }
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  /* Can use insert as each edge belongs to a single processor and vertex data is only for temporary computation and holds no 'real' data. */
  ierr = DMLocalToGlobalBegin(dgnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
/* Compute the L2 Norm of the Vector X associated with the DGNetwork dgnet */
PetscErrorCode DGNetworkNormL2(DGNetwork dgnet, Vec X,PetscReal *norm) 
{
  PetscErrorCode     ierr;
  PetscInt           field,offset,e,eStart,eEnd,c,cStart,cEnd,dof = dgnet->physics.dof,quadsize,q,fieldoff;
  const PetscScalar  *xarr,*coeff;
  EdgeFE             edgefe;
  Vec                localX = dgnet->localX;
  PetscSection       section;
  PetscReal          J,invJ,detJ,qeval,*cellint;
  const PetscReal    *qweight;
  
  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for (field=0;field<dof;field++) {
    norm[field] = 0.0; 
  }
  ierr = PetscMalloc1(dof,&cellint);CHKERRQ(ierr);
  for (e=eStart; e<eEnd-1; e++) {
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    ierr = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight);CHKERRQ(ierr);
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      for(field = 0; field<dof; field++) cellint[field] = 0; 
      for(q=0; q<quadsize; q++) {
        for(field = 0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
          coeff = xarr+offset+fieldoff;
          qeval = evalquad_internal(dgnet,field,q,(PetscReal*)coeff);
          cellint[field] += qweight[q]*PetscPowReal(qeval,2);  
        }
      }
      /* Now we can compute quadrature for each integral for each field */
      for(field = 0; field<dof; field++) {
        norm[field] += detJ*cellint[field];
      }
    }
    for(field = 0; field<dof; field++) {
      norm[field] = PetscSqrtReal(norm[field]);
    }
  }
  ierr = PetscFree(cellint);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&norm,&norm,dof,MPIU_REAL,MPIU_SUM,dgnet->comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal MinMod3(PetscReal a,PetscReal b, PetscReal c) { return (PetscSign(a) == PetscSign(b) && PetscSign(b) == PetscSign(c)) ? PetscSign(a)*PetscMin(PetscMin(PetscAbs(a),PetscAbs(b)),PetscAbs(c)) : 0; }

/* Make a limiter class and redo this in that class */ 

/* Apply a TVD limiter to the DG solution in characteristic variables */

/*

  input *uL *uM and *uR are the cell averages  of the left, center and right element DG solutions respectively 
  ordered by field (requires copying of arrays)

  input 
  
  */
PetscErrorCode TVDLimit_1D(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM,const PetscScalar *uR, PetscScalar *ubdryL, PetscScalar *ubdryR, PetscReal *uCoeff, PetscSection sec, PetscInt c)
{
  PetscErrorCode ierr; 
  PetscScalar    jmpL,jmpR,*cjmpL,*cjmpR,*uLtmp,*uRtmp;
  PetscInt       field,j,k,dof = dgnet->physics.dof,secdof;
  PetscBool      limiteractivated = PETSC_FALSE;
  PetscReal      slope; 
  PetscInt       deg,fieldoff,fielddeg; 

  PetscFunctionBegin;
  /* Create characteristic jumps */
  ierr  = (*dgnet->physics.characteristic)(dgnet->physics.user,dof,uM,dgnet->R,dgnet->Rinv,dgnet->speeds);CHKERRQ(ierr);
  ierr  = PetscArrayzero(dgnet->cjmpLR,2*dof);CHKERRQ(ierr);
  cjmpL = &dgnet->cjmpLR[0];
  cjmpR = &dgnet->cjmpLR[dof];
  for (j=0; j<dof; j++) {
    jmpL = uM[j]-uL[j]; /* Conservative Jumps */
    jmpR = uR[j]-uM[j];
    for (k=0; k<dof; k++) {
      cjmpL[k] += dgnet->Rinv[k+j*dof]*jmpL;
      cjmpR[k] += dgnet->Rinv[k+j*dof]*jmpR;
    }
  }
  /* now the jumps are in the characteristic variables */
  /* write the bdry evals and center cell avg in characteristic variables */
  ierr  = PetscArrayzero(dgnet->cbdryeval_L,dof);CHKERRQ(ierr);
  ierr  = PetscArrayzero(dgnet->cbdryeval_R,dof);CHKERRQ(ierr);
  ierr  = PetscArrayzero(dgnet->cuAvg,dof);CHKERRQ(ierr);

  for(field=0; field<dof; field++) {
    for (k=0; k<dof; k++) {
      dgnet->cbdryeval_L[k]  += dgnet->Rinv[k+field*dof]*ubdryL[field];
      dgnet->cbdryeval_R[k]  += dgnet->Rinv[k+field*dof]*ubdryR[field];
      dgnet->cuAvg[k]        += dgnet->Rinv[k+field*dof]*uM[field];
    }
  }

  uLtmp = dgnet->uLR; 
  uRtmp = &dgnet->uLR[dof];

  /* we apply the limiter detecter */
  for (j=0; j<dof; j++) {
    slope    = MinMod3(cjmpL[j],cjmpR[j],dgnet->cuAvg[j]- dgnet->cbdryeval_L[j]);
    uLtmp[j] = dgnet->cuAvg[j] - slope; 
    slope    = MinMod3(cjmpL[j],cjmpR[j], dgnet->cbdryeval_R[j]-dgnet->cuAvg[j]);
    uRtmp[j] = dgnet->cuAvg[j] + slope;

    dgnet->limitactive[j] = (PetscAbs(uRtmp[j] - dgnet->cbdryeval_R[j]) > 1e-10 || PetscAbs(uLtmp[j] - dgnet->cbdryeval_L[j]) > 1e-10); 
    
    if (dgnet->limitactive[j]) {
      limiteractivated = PETSC_TRUE;
    }
  }

  if (limiteractivated) {
    /* evaluate the coeffients of the center cell in the characteristic coordinates */

    /* Note that we need to expand each basis the the largest DG basis for this to make sense. Thank god 
    the legendre basis is hierarchical (and orthogonal), making this way way easier */ 

    ierr = PetscArrayzero(dgnet->charcoeff,dgnet->physics.maxorder+1*dof);CHKERRQ(ierr);
    for(field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldDof(sec,c,field,&fielddeg);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(sec,c,field,&fieldoff);CHKERRQ(ierr);
      for (deg=0;deg<fielddeg;deg++) {
        for (k=0; k<dof; k++) {
          dgnet->charcoeff[k*(dgnet->physics.maxorder+1)+deg]  += dgnet->Rinv[k+field*dof]*uCoeff[fieldoff+deg];
        }
      }
    }
    /* Now the coeffients are in then characterstic variables. Now apply the P1 MUSCL projection 
        limiter on the detected characteristic variables */ 

    for(j=0; j<dof; j++) {
      if (dgnet->limitactive[j]) {
        ierr = PetscArrayzero(dgnet->charcoeff+j*(dgnet->physics.maxorder+1),dgnet->physics.maxorder+1);CHKERRQ(ierr);
        dgnet->charcoeff[j*(dgnet->physics.maxorder+1)] = dgnet->cuAvg[j]; 
        if (dgnet->physics.maxorder >=1) dgnet->charcoeff[j*(dgnet->physics.maxorder+1)+1] = (uRtmp[j]-uLtmp[j])/2.;
        if (dgnet->physics.maxorder >=2) dgnet->charcoeff[j*(dgnet->physics.maxorder+1)+2] = (uRtmp[j]+uLtmp[j])/2. - dgnet->cuAvg[j];
      }
    } 
    /* Now put the coefficients back into conservative form. Note that 
        as we expanded the DG basis to the maximum order among all field, this 
        technically requires a projection, however the legendre basis 
        is orthogonal and hierarchical, and thus this amounts to simply ignoring higher order terms. 
        
        this does not mess with conservation as the cell averages are unchanged */ 
    ierr = PetscSectionGetDof(sec,c,&secdof);CHKERRQ(ierr);
    for(field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldDof(sec,c,field,&fielddeg);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(sec,c,field,&fieldoff);CHKERRQ(ierr);
      for (deg=0;deg<fielddeg;deg++) {
        uCoeff[fieldoff+deg] = 0; 
        for (k=0; k<dof; k++) {
          uCoeff[fieldoff+deg] += dgnet->R[field+k*dof]*dgnet->charcoeff[k*(dgnet->physics.maxorder+1)+deg];
        }
      }
    }
  } 
  /* the uCoeff now contains the limited coefficients */
  PetscFunctionReturn(0);
}
/*

  input *uL *uM and *uR are the cell averages  of the left, center and right element DG solutions respectively 
  ordered by field (requires copying of arrays)

  input 
  
  */
PetscErrorCode TVDLimit_1D_2(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM,const PetscScalar *uR, PetscScalar *ubdryL, PetscScalar *ubdryR, PetscReal *uCoeff, PetscSection sec, PetscInt c)
{
  PetscErrorCode ierr; 
  PetscScalar    *cjmpL,*cjmpR,*uLtmp,*uRtmp,*cuLtmp,*cuRtmp; 
  PetscInt       j,dof = dgnet->physics.dof;
  PetscBool      limiteractivated = PETSC_FALSE;
  PetscReal      slope; 

  PetscFunctionBegin;
  /* Do limiter detection in the conservative variables */
  uLtmp  = dgnet->cbdryeval_L; 
  uRtmp  = dgnet->cbdryeval_R;
  cuLtmp = dgnet->cuLR; 
  cuRtmp = dgnet->cuLR+dof;
  cjmpL = &dgnet->cjmpLR[0];
  cjmpR = &dgnet->cjmpLR[dof];
  /* Compute the conservative jumps */
  for(j=0;j<dof;j++) {
    cjmpL[j] = uM[j] - uL[j];
    cjmpR[j] = uR[j] - uM[j];
  }
  /* we apply the limiter detecter */
  for (j=0; j<dof; j++) {
    slope    = MinMod3(cjmpL[j],cjmpR[j],uM[j]- ubdryL[j]);
    uLtmp[j] = uM[j] - slope; 
    //ierr = PetscPrintf(dgnet->comm,"uL -   jmpL: %e jmpR: %e bdryL: %e  minmod: %e uLtmp: %e diff: %e   \n",cjmpL[j],cjmpR[j],uM[j]-ubdryL[j], slope,uLtmp[j],uLtmp[j] - ubdryL[j]);CHKERRQ(ierr);
    slope    = MinMod3(cjmpL[j],cjmpR[j], ubdryR[j]-uM[j]);
    uRtmp[j] = uM[j] + slope;
    dgnet->limitactive[j] = (PetscAbs(uRtmp[j] - ubdryR[j]) > 1e-12 || PetscAbs(uLtmp[j] - ubdryL[j]) > 1e-12); 
    if (dgnet->limitactive[j]) {
      limiteractivated = PETSC_TRUE;
    }
  }
  if (limiteractivated) {
    ierr = TVDLimit_1D(dgnet,uL,uM,uR,ubdryL,ubdryR,uCoeff,sec,c);CHKERRQ(ierr);
  } 
  /* the uCoeff now contains the limited coefficients */
  PetscFunctionReturn(0);
}
/* basis one-sided limiter, super lame and not robust, detects only if the cell averages of the neighbors are "large" */
PetscErrorCode Limit_1D_onesided(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM, PetscReal *uCoeff, PetscSection sec, PetscInt c, PetscReal jumptol)
{
  PetscErrorCode ierr; 
  PetscInt       field,j,k,dof = dgnet->physics.dof,secdof;
  PetscBool      limiteractivated = PETSC_FALSE;
  PetscInt       deg,fieldoff,fielddeg; 

  PetscFunctionBegin;
   
    /* we apply the limiter detecter in conservative variables */
    for (j=0; j<dof; j++) {
      dgnet->limitactive[j] = (PetscAbs((uM[j]-uL[j])/uM[j])>jumptol); 
      if (dgnet->limitactive[j]) limiteractivated = PETSC_TRUE; 
    }

    if (limiteractivated) {
      /* evaluate the coeffients of the center cell in the characteristic coordinates */
          /* now the jumps are in the characteristic variables */
      /* write the bdry evals and center cell avg in characteristic variables */
      ierr  = (*dgnet->physics.characteristic)(dgnet->physics.user,dof,uM,dgnet->R,dgnet->Rinv,dgnet->speeds);CHKERRQ(ierr);
      ierr  = PetscArrayzero(dgnet->cuAvg,dof);CHKERRQ(ierr);

      for(field=0; field<dof; field++) {
        for (k=0; k<dof; k++) {
          dgnet->cuAvg[k]        += dgnet->Rinv[k+field*dof]*uM[field];
        }
      }

      /* Note that we need to expand each basis the the largest DG basis for this to make sense. Thank god 
      the legendre basis is hierarchical (and orthogonal), making this way way easier */ 

      ierr = PetscArrayzero(dgnet->charcoeff,dgnet->physics.maxorder+1*dof);CHKERRQ(ierr);
      for(field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldDof(sec,c,field,&fielddeg);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(sec,c,field,&fieldoff);CHKERRQ(ierr);
        for (deg=0;deg<fielddeg;deg++) {
          for (k=0; k<dof; k++) {
            dgnet->charcoeff[k*(dgnet->physics.maxorder+1)+deg]  += dgnet->Rinv[k+field*dof]*uCoeff[fieldoff+deg];
          }
        }
      }
      /* Now the coeffients are in then characterstic variables. Now apply the P0 projection */

      for(j=0; j<dof; j++) {
        if (dgnet->limitactive[j]) {
          ierr = PetscArrayzero(dgnet->charcoeff+j*(dgnet->physics.maxorder+1),dgnet->physics.maxorder+1);CHKERRQ(ierr);
          dgnet->charcoeff[j*(dgnet->physics.maxorder+1)] = dgnet->cuAvg[j];  
        }
      } 
      /* Now put the coefficients back into conservative form. Note that 
         as we expanded the DG basis to the maximum order among all field, this 
         technically requires a projection, however the legendre basis 
         is orthogonal and hierarchical, and thus this amounts to simply ignoring higher order terms. 
         
         this does not mess with conservation as the cell averages are unchanged */ 
      ierr = PetscSectionGetDof(sec,c,&secdof);CHKERRQ(ierr);
      for(field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldDof(sec,c,field,&fielddeg);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(sec,c,field,&fieldoff);CHKERRQ(ierr);
        for (deg=0;deg<fielddeg;deg++) {
          uCoeff[fieldoff+deg] = 0; 
          for (k=0; k<dof; k++) {
            uCoeff[fieldoff+deg] += dgnet->R[field+k*dof]*dgnet->charcoeff[k*(dgnet->physics.maxorder+1)+deg];
          }
        }
      }
    } 
    /* the uCoeff now contains the limited coefficients */
  PetscFunctionReturn(0);
} 

/* Version of DGNetlimit that has the function pattern of a rhs function. Necessary for the nested version 
as I will call this with diffferent ctx within a another post-stage function in the nested case. The alternative 
was dummy ts objects just to store the ctx */

PetscErrorCode DGNetlimiter_ctx(Vec Y,void* ctx) {
  PetscErrorCode ierr; 
  DGNetwork      dgnet = (DGNetwork)ctx; 
  PetscScalar    *xarr;
  PetscInt       e,c,eStart,eEnd,cStart,cEnd;
  PetscInt       offset,dof,field,fieldoff;
  PetscReal      detJ;
  Vec            localX;
  EdgeFE         edgefe; 
  PetscSection   section;

  PetscFunctionBeginUser;
  dof  = dgnet->physics.dof; localX = dgnet->localX;
  ierr = DMGlobalToLocalBegin(dgnet->network,Y,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,Y,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  /* Iterate through the edges of the network and apply the limiter to each mesh on the edge */
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {  
    /* Also the update pattern is probably not ideal but I don't care for now */
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    for(c=cStart+1; c<cEnd-1; c++) {
      /* make the cell avg arrays and bdry evaluations */
      for(field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        dgnet->uLR[field]         = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
        dgnet->uLR[field+dof]     = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        dgnet->uavgs[field+dof]   = xarr[offset+fieldoff]; 
        ierr = PetscSectionGetFieldOffset(section,c-1,field,&fieldoff);CHKERRQ(ierr);
        dgnet->uavgs[field]       = xarr[offset+fieldoff];
        ierr = PetscSectionGetFieldOffset(section,c+1,field,&fieldoff);CHKERRQ(ierr);
        dgnet->uavgs[field+2*dof] = xarr[offset+fieldoff]; 
      }
      ierr = TVDLimit_1D_2(dgnet,dgnet->uavgs, dgnet->uavgs+dof,dgnet->uavgs+2*dof,dgnet->uLR,dgnet->uLR+dof,xarr+offset,section,c);CHKERRQ(ierr);
      /* 
        TODO : Could print out the limited cells here 
      */ 
    }
    /* Now we limit the bdry cells */
    for(field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cStart,field,&fieldoff);CHKERRQ(ierr);
      dgnet->uavgs[field+dof]   = xarr[offset+fieldoff]; 
      ierr = PetscSectionGetFieldOffset(section,cStart+1,field,&fieldoff);CHKERRQ(ierr);
      dgnet->uavgs[field]   = xarr[offset+fieldoff];
    }
    ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,cStart,NULL,NULL,NULL,&detJ);CHKERRQ(ierr);
    ierr = Limit_1D_onesided(dgnet,dgnet->uavgs, dgnet->uavgs+dof,xarr+offset,section,cStart,dgnet->jumptol/detJ);CHKERRQ(ierr);

    for(field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff);CHKERRQ(ierr);
      dgnet->uavgs[field]   = xarr[offset+fieldoff]; 
      ierr = PetscSectionGetFieldOffset(section,cEnd-2,field,&fieldoff);CHKERRQ(ierr);
      dgnet->uavgs[field+dof]   = xarr[offset+fieldoff];
    }
    ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,cEnd,NULL,NULL,NULL,&detJ);CHKERRQ(ierr);
    ierr = Limit_1D_onesided(dgnet,dgnet->uavgs+dof, dgnet->uavgs,xarr+offset,section,cEnd-1,dgnet->jumptol/detJ);CHKERRQ(ierr);
  }

  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dgnet->network,localX,INSERT_VALUES,Y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localX,INSERT_VALUES,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* All this does is call the ctx version with a the ts ctx, and limit the current stage vector */ 
PetscErrorCode DGNetlimiter(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y) {
  PetscErrorCode ierr; 
  DGNetwork      dgnet;

  PetscFunctionBeginUser;
  ierr = TSGetApplicationContext(ts,&dgnet);CHKERRQ(ierr);
  ierr = DGNetlimiter_ctx(Y[stageindex],dgnet);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Nested Version of Limiters. WIP interface to allow multiple simulations to run alongside eachother */ 
PetscErrorCode DGNetlimiter_Nested(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y) {
  PetscErrorCode  ierr;
  DGNetwork_Nest  dgnet_nest;
  PetscInt        i,nestsize,numsim;
  MPI_Comm        comm; 
  Vec             Ysub; 
  PetscBool       isequal; 

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts,&dgnet_nest);CHKERRQ(ierr);
  numsim = dgnet_nest->numsimulations;
  /* This routine only works if X,F are VecNest, sanity check here */
  ierr = PetscObjectTypeCompare((PetscObject)Y[stageindex],VECNEST,&isequal);CHKERRQ(ierr);
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec Y[stageindex] must be of type VecNest. \n");}
  /* Small sanity check */
  ierr = VecNestGetSize(Y[stageindex],&nestsize);CHKERRQ(ierr);
  if (nestsize < numsim) {SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Vec X must have at least 1 vector per dgnet simulation.\n  \
  Num Simulation: %i \n Num Vectors %i \n ",numsim,nestsize);}

  /* For each dgnet simulation in dgnet_nest run the limiter*/
  for(i=0; i<numsim; i++) {
    ierr = VecNestGetSubVec(Y[stageindex],i,&Ysub);CHKERRQ(ierr); /* Doesn't need to be returned */
    ierr = DGNetlimiter_ctx(Ysub,dgnet_nest->dgnets[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* For Running multiple netrs rhs tests*/
PetscErrorCode DGNetRHS_NETRS_Nested(TS ts,PetscReal time,Vec X,Vec F,void *ctx) 
{
  PetscErrorCode  ierr;
  DGNetwork_Nest  dgnet_nest = (DGNetwork_Nest)ctx;
  PetscInt        i,nestsize,numsim = dgnet_nest->numsimulations;
  MPI_Comm        comm; 
  Vec             Xsub,Fsub;  
  PetscBool       isequal; 

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  /* This routine only works if X,F are VecNest, sanity check here */
  ierr = PetscObjectTypeCompare((PetscObject)X,VECNEST,&isequal);CHKERRQ(ierr);
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec X must be of type VecNest.");}
  ierr = PetscObjectTypeCompare((PetscObject)F,VECNEST,&isequal);CHKERRQ(ierr);
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec F must be of type VecNest.");}
  /* Small sanity check */
  ierr = VecNestGetSize(X,&nestsize);CHKERRQ(ierr);
  if (nestsize < numsim) {SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Vec X must have at least 1 vector per dgnet simulation.\n \
Num Simulation: %i \n Num Vectors %i \n ",numsim,nestsize);}
  /* For each dgnet simulation in dgnet_nest run the DG kernel */
  for(i=0; i<numsim; i++) {
    ierr = VecNestGetSubVec(X,i,&Xsub);CHKERRQ(ierr); /* Don't need to be returned */
    ierr = VecNestGetSubVec(F,i,&Fsub);CHKERRQ(ierr);
    ierr = DGNetRHS_NETRSVERSION(ts,time,Xsub,Fsub,dgnet_nest->dgnets[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DGNetRHS_NETRSVERSION2(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  DGNetwork      dgnet = (DGNetwork)ctx;    
  PetscReal      maxspeed,detJ,J,invJ,*numflux,*netflux,*errorest; 
  PetscScalar    *f,*xarr,*coeff; 
  PetscInt       v,e,c,vStart,vEnd,eStart,eEnd,vfrom,vto,cStart,cEnd,q,deg,ndeg,quadsize,tab,face,fStart,fEnd;
  PetscInt       offsetf,offset,nedges,i,j,dof = dgnet->physics.dof,field,fieldoff;
  const PetscInt *cone,*edges,*supp;
  Vec            localX = dgnet->localX,localF = dgnet->localF,Ftmp = dgnet->Ftmp; 
  EdgeFE         edgefe; 
  Junction       junction;
  PetscSection   section;
  const PetscReal *qweight;
  RiemannSolver   rs = dgnet->physics.rs; 
  PetscBool       adaption;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(dgnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points (NOTE: This routine (and the others done elsewhere) need to be refactored) */
    ierr = DMNetworkGetLocalVecOffset(dgnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(dgnet->network,v,JUNCTION,NULL,(void**)&junction,NULL); 
    ierr = DMNetworkGetSupportingEdges(dgnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    for (i=0; i<nedges; i++) {
      e     = edges[i];
      ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
      ierr  = DMNetworkGetConnectedVertices(dgnet->network,e,&cone);CHKERRQ(ierr);
      ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
      /* DMPlex stuff here, get cell chart */
      ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
      ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];
      if (v == vfrom) {
        /* left eval */
        for (field=0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,cStart,field,&fieldoff);CHKERRQ(ierr);
          f[offsetf+edgefe->offset_vfrom+field] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
        }
      } else if (v == vto) {
        for (field=0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff);CHKERRQ(ierr);
          f[offsetf+edgefe->offset_vto+field] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        }
      }
    }
  }
  /* Now communicate the flux/reconstruction data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dgnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dgnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(dgnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Now ALL processors have the evaluation data to compute the coupling flux */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(dgnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(dgnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    /* compute the coupling flux */
    ierr = NetRSEvaluate(junction->netrs,f+offsetf,junction->dir,&netflux,&errorest,&adaption);CHKERRQ(ierr);
  
    for (i=0; i<junction->numedges; i++) {
      for (j=0; j<dof; j++) {
          f[offsetf+i*dof+j] = netflux[i*dof+j];
      }
    }
  }
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight);CHKERRQ(ierr);
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      for(q=0; q<quadsize; q++) {
        for(field = 0; field<dof; field++) {
          ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
          coeff = xarr+offset+fieldoff;
          dgnet->pteval[field] = evalquad_internal(dgnet,field,q,coeff);
        }
        dgnet->physics.flux((void*)dgnet->physics.user,dgnet->pteval,dgnet->fluxeval+q*dof);
      }
      /* Now we can compute quadrature for each integral for each field */
      for(field = 0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          for (q = 0; q<quadsize; q++) {
            *coeff += qweight[q]*dgnet->fluxeval[q*dof+field]*dgnet->LegEvalD[tab][ndeg*q+deg]; 
          }
        }
      }
    }
    /* Flux Time !!! :) */ 
    /* update the boundary cells first, (cstart,cEnd) as their fluxes are coupling fluxes */
    ierr   = DMNetworkGetConnectedVertices(dgnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    /*cStart cell */
    ierr   = DMNetworkGetComponent(dgnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(dgnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    /* Update the vfrom vertex flux for this edge */
    for (field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cStart,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff += f[edgefe->offset_vfrom+field+offsetf]*dgnet->LegEvaL_bdry[tab][deg];
      }
    }
    /* cEnd cell */
    ierr   = DMNetworkGetComponent(dgnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr); 
    ierr   = DMNetworkGetLocalVecOffset(dgnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    /* Update the vfrom vertex flux for this edge */
    for (field=0; field<dof; field++) {
      ierr = PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff -= f[edgefe->offset_vto+field+offsetf]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
      }
    }
    /* 2) Then iterate through the flux updates */
    /* we iterate through the 1 codim cells (faces) skipping the first and last to compute the numerical fluxes and update the resulting cells coefficients */
    ierr  = DMPlexGetHeightStratum(edgefe->dm,1,&fStart,&fEnd);CHKERRQ(ierr);
    for(face=fStart+1; face<fEnd-1; face++) {
      /* WE ASSUME 1D HERE WITH SUPPORT SIZE OF 2 !!!! */
      ierr = DMPlexGetSupport(edgefe->dm,face,&supp);CHKERRQ(ierr);
      /* evaluate at the face */
      for(field = 0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff);CHKERRQ(ierr);
        dgnet->uLR[field] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        ierr = PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff);CHKERRQ(ierr);
        dgnet->uLR[field+dof] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
      }
      ierr = RiemannSolverEvaluate(rs,dgnet->uLR,dgnet->uLR+dof,&numflux,&maxspeed);CHKERRQ(ierr);
      /* Update coefficents with the numerical flux */
      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff -= numflux[field]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
        }
      }
      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff += numflux[field]*dgnet->LegEvaL_bdry[tab][deg];
        }
      }
    }
    /* Normalization loop */
    for (c=cStart; c<cEnd; c++) {
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      for(field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff  = f+offset+fieldoff+deg;
          *coeff *= dgnet->Leg_L2[tab][deg]/detJ; /* Inverting the Mass matrix. To be refactored later 
          with arbitrary basis */
        }
      }
    }
  }
  /* Data Cleanup */
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dgnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dgnet->network,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}