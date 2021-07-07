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

PetscReal innerprod_internal(PetscInt len,PetscReal *x  ,PetscReal *y) {
  PetscInt i; 
  PetscReal sum = 0.; 

  for(i=0; i<len; i++) {
    sum += x[i]*y[i];
  }
  return sum;
}

PetscReal evalboundary_internal(DGNetwork dgnet, PetscInt field,PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEvaL_bdry[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
PetscReal evalquad_internal(DGNetwork dgnet, PetscInt field, PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEval[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
PetscReal evalquadDer_internal(DGNetwork dgnet, PetscInt field, PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEvalD[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
PetscErrorCode DGNetRHS(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  DGNetwork      dgnet = (DGNetwork)ctx;    
  PetscReal      maxspeed,detJ,J,invJ;
  PetscScalar    *f,*xarr,*coeff;
  PetscInt       v,e,c,vStart,vEnd,eStart,eEnd,vfrom,vto,cStart,cEnd,q,deg,ndeg,quadsize,tab,face,fStart,fEnd;
  PetscInt       offsetf,offset,nedges,i,j,dof = dgnet->physics.dof,field,fieldoff;
  const PetscInt *cone,*edges,*supp;
  Vec            localX = dgnet->localX,localF = dgnet->localF,Ftmp = dgnet->Ftmp; 
  EdgeFE         edgefe; 
  Junction       junction;
  PetscSection   section;
  const PetscReal *qweight;

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
        /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
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

  /*
    TODO : Think about how to implement INFLOW/OUTFLOW Boundary conditions in this framework (or devise a better one 
    Too tired to think of how to do boundary conditions with dg in an abstract way )
  */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetLocalVecOffset(dgnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(dgnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
      /* compute the coupling flux */
    junction->couplingflux(dgnet,f+offsetf,junction->dir,junction->flux,&maxspeed,junction);
    for (i=0; i<junction->numedges; i++) {
      for (j=0; j<dof; j++) {
          f[offsetf+i*dof+j] = junction->flux[i*dof+j];
        }
      }
  }
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    /* We do a DG Update here, the main kernel of the algorithm. Note that this can be written much better 
    than is currently done, by seperating out the mesh level dg kernel from the iteration through the edges.
    That is the should be a seperate function and should be an abstract function for the discretization (different implementations 
    could each have their own? Maybe... It works for now though */

    /* Also the update pattern is probably not ideal but I don't care for now */
    ierr  = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ);CHKERRQ(ierr);
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight);CHKERRQ(ierr);
      
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
      ierr = (*dgnet->physics.riemann)(dgnet->physics.user,dof,dgnet->uLR,dgnet->uLR+dof,dgnet->flux,&maxspeed);CHKERRQ(ierr);
      /* Update coefficents with the numerical flux */
      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff -= dgnet->flux[field]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
        }
      }

      for (field=0; field<dof; field++) {
        ierr = PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff);CHKERRQ(ierr);
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff += dgnet->flux[field]*dgnet->LegEvaL_bdry[tab][deg];
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
            dgnet->physics.samplenetwork((void*)&dgnet->physics.user,dgnet->initial,t,qpoint[q]*J+v0,dgnet->pteval,edgeid);
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
