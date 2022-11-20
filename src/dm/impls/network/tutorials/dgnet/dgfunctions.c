#include "dgnet.h"
#include <stdio.h>


PetscLogEvent DGNET_Limiter;
PetscLogEvent DGNET_Edge_RHS;
PetscLogEvent DGNET_RHS_COMM;
PetscLogEvent DGNET_RHS_Vert;

PetscErrorCode PhysicsDestroy_SimpleFree_Net(void *vctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(vctx));
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

/*
static PetscReal evalquadDer_internal(DGNetwork dgnet, PetscInt field, PetscInt qpoint, PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEvalD[tab][qpoint*(ndegree+1)+deg];
  }
  return eval; 
}
*/
PetscErrorCode DGNetworkProject(DGNetwork dgnet,Vec X0,PetscReal t) 
{
  PetscInt       type,offset,e,eStart,eEnd,dof = dgnet->physics.dof;
  PetscInt       c,cStart,cEnd,field,edgeid,deg,ndeg,tab,fieldoff,quadsize,q;
  PetscScalar    *xarr,*coeff;
  EdgeFE         edgefe;
  Vec            localX = dgnet->localX;
  PetscReal      J,invJ,detJ,v0;
  const PetscReal *qpoint,*qweight;
  PetscSection   section;
  
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(localX));
  PetscCall(VecGetArray(localX,&xarr));
  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd));
  PetscCall(PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,&qpoint,&qweight)); 
  for (e=eStart; e<eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network,e,FVEDGE,&type,(void**)&edgefe,NULL));
    PetscCall(DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd));
    PetscCall(DMNetworkGetGlobalEdgeIndex(dgnet->network,e,&edgeid));
    PetscCall(DMGetSection(edgefe->dm,&section));
    for (c=cStart; c<cEnd; c++) {
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,&v0,&J,&invJ,&detJ));
      /* We can compute points in real space by Jx + v0, the affine transformation */
      for(field=0; field<dof; field++){
        PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
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
  PetscCall(VecRestoreArray(localX,&xarr));
  /* Can use insert as each edge belongs to a single processor and vertex data is only for temporary computation and holds no 'real' data. */
  PetscCall(DMLocalToGlobalBegin(dgnet->network,localX,INSERT_VALUES,X0));
  PetscCall(DMLocalToGlobalEnd(dgnet->network,localX,INSERT_VALUES,X0)); 
  PetscFunctionReturn(0);
}
/* Project between a coarse and fine dgnet network. Assume same network, but different meshes for each edge */


/* only will work for implicit fem geometry and assumes that the physical geometry has dimension 1 */
/* very hacked together, thrown out and replace as I implement a better hieracherical dm format */ 
PetscErrorCode DGNetworkProject_Coarse_To_Fine(DGNetwork dgnet_fine,DGNetwork dgnet_coarse, Vec Coarse, Vec projection) 
{
  PetscInt       type,offset_fine,offset_coarse,e,eStart,eEnd,dof = dgnet_fine->physics.dof;
  PetscInt       i,j,c_fine,cStart_fine,cEnd_fine,field,deg_fine,ndeg_fine,tab,tab_coarse,fieldoff_fine,quadsize,q,order_fine,order_coarse;
  PetscInt       ndeg_coarse,fieldoff_coarse,deg_coarse,c_coarse,cStart_coarse,cEnd_coarse;
  PetscInt       **degs_coarse; 
  PetscScalar    *xarr_proj,*xarr_coarse,*coeff,*coeff_coarse,qeval;
  PetscScalar    *coords_coarse= NULL;
  EdgeFE         edgefe_fine, edgefe_coarse;
  Vec            coord_coarse;
  PetscReal      J_fine,invJ_fine,detJ_fine,v0_fine,xend,q_geom,q_coarse,J_coarse,invJ_coarse,detJ_coarse,v0_coarse;
  PetscReal       **q_coarse_Eval;
  const PetscReal *qpoint,*qweight;
  PetscSection   section_fine,section_coarse,sec_coord_coarse;
  DM             geom_coarse; 
  
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(projection));
  PetscCall(VecGetArray(projection,&xarr_proj));
  PetscCall(VecGetArray(Coarse,&xarr_coarse));
  PetscCall(DMNetworkGetEdgeRange(dgnet_fine->network,&eStart,&eEnd)); /* assume same for coarse */ 

  /* pick the finer quadrature for the evaluation */
  PetscCall(PetscQuadratureGetOrder(dgnet_fine->quad,&order_fine)); 
  PetscCall(PetscQuadratureGetOrder(dgnet_fine->quad,&order_coarse)); 
  if (order_fine >= order_coarse) 
  {
    PetscCall(PetscQuadratureGetData(dgnet_fine->quad,NULL,NULL,&quadsize,&qpoint,&qweight)); 
  } else {
    PetscCall(PetscQuadratureGetData(dgnet_coarse->quad,NULL,NULL,&quadsize,&qpoint,&qweight)); 
  }

  /* allocate array for legendre evaluations at the fine qaudarture points */ 
  PetscCall(PetscMalloc2(dgnet_coarse->tabordersize,&q_coarse_Eval,dgnet_coarse->tabordersize,&degs_coarse));
  for(i=0;i<dgnet_coarse->tabordersize;i++) {
    PetscCall(PetscMalloc2(dgnet_coarse->taborder[i]+1,&q_coarse_Eval[i],dgnet_coarse->taborder[i]+1,&degs_coarse[i]));
    for(j=0; j<=dgnet_coarse->taborder[i]; j++) { degs_coarse[i][j] = j; }
  }
  /* project the mesh on each edge */
  for (e=eStart; e<eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet_fine->network,e,FVEDGE,&type,(void**)&edgefe_fine,NULL));
    PetscCall(DMNetworkGetComponent(dgnet_coarse->network,e,FVEDGE,&type,(void**)&edgefe_coarse,NULL));

    PetscCall(DMNetworkGetLocalVecOffset(dgnet_fine->network,e,FVEDGE,&offset_fine));
    PetscCall(DMPlexGetHeightStratum(edgefe_fine->dm,0,&cStart_fine,&cEnd_fine));
    PetscCall(DMGetSection(edgefe_fine->dm,&section_fine));

    PetscCall(DMNetworkGetLocalVecOffset(dgnet_coarse->network,e,FVEDGE,&offset_coarse));
    PetscCall(DMPlexGetHeightStratum(edgefe_coarse->dm,0,&cStart_coarse,&cEnd_coarse));
    PetscCall(DMGetSection(edgefe_coarse->dm,&section_coarse));

    PetscCall(DMGetCoordinateDM(edgefe_coarse->dm,&geom_coarse)); 
    PetscCall(DMGetCoordinatesLocal(edgefe_coarse->dm,&coord_coarse));
    PetscCall(DMGetCoordinateSection(edgefe_coarse->dm,&sec_coord_coarse));

    c_coarse = cStart_coarse;
    /* asssuming implicit geometry and 1d meshes henceforth */
    /* assume numcoords == 2 */
    PetscCall(DMPlexVecGetClosure(edgefe_coarse->dm, sec_coord_coarse, coord_coarse, c_coarse, NULL, &coords_coarse));
    xend = coords_coarse[1]; 
    PetscCall(DMPlexVecRestoreClosure(edgefe_coarse->dm, sec_coord_coarse, coord_coarse, c_coarse, NULL, &coords_coarse));
    PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe_coarse->dm,c_coarse,&v0_coarse,&J_coarse,&invJ_coarse,&detJ_coarse));
    for (c_fine=cStart_fine; c_fine<cEnd_fine; c_fine++) {
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe_fine->dm,c_fine,&v0_fine,&J_fine,&invJ_fine,&detJ_fine));
      /* We can compute points in real space by Jx + v0, the affine transformation */
      for (q=0; q< quadsize; q++) {
        q_geom = J_fine*qpoint[q]+v0_fine;
      /* Find the cell on the coarse mesh the q_geom point belongs to. Simple Search as we are in 1D */
        while(q_geom > xend) {
          c_coarse++;
          PetscCall(DMPlexVecGetClosure(edgefe_coarse->dm, sec_coord_coarse, coord_coarse, c_coarse, NULL, &coords_coarse));
          xend = coords_coarse[1]; 
          PetscCall(DMPlexVecRestoreClosure(edgefe_coarse->dm, sec_coord_coarse, coord_coarse, c_coarse, NULL, &coords_coarse));
          PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe_coarse->dm,c_coarse,&v0_coarse,&J_coarse,&invJ_coarse,&detJ_coarse));
        }
      /* Find the q_geom point on the coarse reference cell [-1,1] and evaluate the coarse legendre basis at that point */
        q_coarse = invJ_coarse*(q_geom-v0_coarse);
        for(i=0;i<dgnet_coarse->tabordersize;i++) {
          PetscCall(PetscDTLegendreEval(1,&q_coarse,dgnet_coarse->taborder[i]+1,degs_coarse[i],q_coarse_Eval[i],NULL,NULL));
        }


        for(field=0; field<dof; field++){
          PetscCall(PetscSectionGetFieldOffset(section_fine,c_fine,field,&fieldoff_fine));
          PetscCall(PetscSectionGetFieldOffset(section_coarse,c_coarse,field,&fieldoff_coarse));
          tab = dgnet_fine->fieldtotab[field];
          tab_coarse = dgnet_coarse->fieldtotab[field];
          ndeg_fine = dgnet_fine->taborder[tab]+1;
          ndeg_coarse = dgnet_coarse->taborder[tab_coarse]+1;
          coeff_coarse = xarr_coarse+offset_coarse+fieldoff_coarse;
          qeval = 0.0;
          for(deg_coarse = 0; deg_coarse<ndeg_coarse; deg_coarse++) {
            qeval += q_coarse_Eval[tab_coarse][deg_coarse]*coeff_coarse[deg_coarse];
          }
          /* projection for all fields at the quadrature point */ 
          for (deg_fine = 0; deg_fine<ndeg_fine; deg_fine++) {
            coeff  = xarr_proj+offset_fine+fieldoff_fine+deg_fine;
            *coeff += qweight[q]*qeval*dgnet_fine->LegEval[tab][ndeg_fine*q+deg_fine]; 
          }
        }
      }
      /* normalization */
      for(field=0; field<dof; field++) {
        tab = dgnet_fine->fieldtotab[field];
        ndeg_fine = dgnet_fine->taborder[tab]+1;
        PetscCall(PetscSectionGetFieldOffset(section_fine,c_fine,field,&fieldoff_fine));
        for (deg_fine = 0; deg_fine<ndeg_fine; deg_fine++) {
              coeff  = xarr_proj+offset_fine+fieldoff_fine+deg_fine;
              *coeff *= dgnet_fine->Leg_L2[tab][deg_fine];
        }
      }
    }
  }
  PetscCall(VecRestoreArray(projection,&xarr_proj));
  PetscCall(VecRestoreArray(Coarse,&xarr_coarse));

  /* free allocated memory */ 
  for(i=0;i<dgnet_coarse->tabordersize;i++) {
   PetscCall(PetscFree2(q_coarse_Eval[i],degs_coarse[i]));
  } 
  PetscCall(PetscFree2(q_coarse_Eval,degs_coarse));
  PetscFunctionReturn(0);
}
/* Compute the L2 Norm of the Vector X associated with the DGNetwork dgnet */
PetscErrorCode DGNetworkNormL2(DGNetwork dgnet, Vec X,PetscReal *norm) 
{
  PetscInt           field,offset,e,eStart,eEnd,c,cStart,cEnd,dof = dgnet->physics.dof,quadsize,q,fieldoff;
  const PetscScalar  *xarr,*coeff;
  EdgeFE             edgefe;
  Vec                localX = dgnet->localX;
  PetscSection       section;
  PetscReal          J,invJ,detJ,qeval,*cellint,*norm_wrk;
  const PetscReal    *qweight;
  
  PetscFunctionBegin;
  PetscCall(DMGlobalToLocalBegin(dgnet->network,X,INSERT_VALUES,localX)); 
  PetscCall(DMGlobalToLocalEnd(dgnet->network,X,INSERT_VALUES,localX));
  PetscCall(VecGetArrayRead(localX,&xarr));
  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd));
  for (field=0;field<dof;field++) {
    norm[field] = 0.0; 
  }
  PetscCall(PetscMalloc2(dof,&cellint,dof,&norm_wrk));
  for (e=eStart; e<eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL));
    PetscCall(DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd));
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    PetscCall(DMGetSection(edgefe->dm,&section));
    PetscCall(PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight));
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ));
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      for(field = 0; field<dof; field++) cellint[field] = 0; 
      for(q=0; q<quadsize; q++) {
        for(field = 0; field<dof; field++) {
          PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
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
  }
  PetscCall(VecRestoreArrayRead(localX,&xarr));
  PetscCallMPI(MPI_Allreduce(norm,norm_wrk,dof,MPIU_REAL,MPIU_SUM,dgnet->comm));
  for(field = 0; field<dof; field++) {
      norm[field] = PetscSqrtReal(norm_wrk[field]);
  }
  PetscCall(PetscFree2(cellint,norm_wrk));
  PetscFunctionReturn(0);
}

static inline PetscReal MinMod3(PetscReal a,PetscReal b, PetscReal c) { return (PetscSign(a) == PetscSign(b) && PetscSign(b) == PetscSign(c)) ? PetscSign(a)*PetscMin(PetscMin(PetscAbs(a),PetscAbs(b)),PetscAbs(c)) : 0; }

/* Make a limiter class and redo this in that class */ 

/* Apply a TVD limiter to the DG solution in characteristic variables */

/*

  input *uL *uM and *uR are the cell averages  of the left, center and right element DG solutions respectively 
  ordered by field (requires copying of arrays)

  input 
  
  */
PetscErrorCode TVDLimit_1D(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM,const PetscScalar *uR, PetscScalar *ubdryL, PetscScalar *ubdryR, PetscReal *uCoeff, PetscSection sec, PetscInt c)
{
  PetscScalar    jmpL,jmpR,*cjmpL,*cjmpR,*uLtmp,*uRtmp;
  PetscInt       field,j,k,dof = dgnet->physics.dof,secdof;
  PetscBool      limiteractivated = PETSC_FALSE;
  PetscReal      slope; 
  PetscInt       deg,fieldoff,fielddeg; 

  PetscFunctionBegin;
  /* Create characteristic jumps */
  PetscCall((*dgnet->physics.characteristic)(dgnet->physics.user,dof,uM,dgnet->R,dgnet->Rinv,dgnet->speeds));
  PetscCall(PetscArrayzero(dgnet->cjmpLR,2*dof));
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
  PetscCall(PetscArrayzero(dgnet->cbdryeval_L,dof));
  PetscCall(PetscArrayzero(dgnet->cbdryeval_R,dof));
  PetscCall(PetscArrayzero(dgnet->cuAvg,dof));

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

    /* Note that we need to expand each basis to the largest DG basis for this to make sense. Thank god 
    the legendre basis is hierarchical (and orthogonal), making this way way easier */ 

    PetscCall(PetscArrayzero(dgnet->charcoeff,(dgnet->physics.maxorder+1)*dof));
    for(field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldDof(sec,c,field,&fielddeg));
      PetscCall(PetscSectionGetFieldOffset(sec,c,field,&fieldoff));
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
        PetscCall(PetscArrayzero(dgnet->charcoeff+j*(dgnet->physics.maxorder+1),dgnet->physics.maxorder+1));
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
    PetscCall(PetscSectionGetDof(sec,c,&secdof));
    for(field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldDof(sec,c,field,&fielddeg));
      PetscCall(PetscSectionGetFieldOffset(sec,c,field,&fieldoff));
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
PetscErrorCode TVDLimit_1D_2(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM,const PetscScalar *uR, PetscScalar *ubdryL, PetscScalar *ubdryR, PetscReal *uCoeff, PetscSection sec, PetscInt c,PetscReal M)
{
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
    //PetscCall(PetscPrintf(dgnet->comm,"uL -   jmpL: %e jmpR: %e bdryL: %e  minmod: %e uLtmp: %e diff: %e   \n",cjmpL[j],cjmpR[j],uM[j]-ubdryL[j], slope,uLtmp[j],uLtmp[j] - ubdryL[j]));
    slope    = MinMod3(cjmpL[j],cjmpR[j], ubdryR[j]-uM[j]);
    uRtmp[j] = uM[j] + slope;
    dgnet->limitactive[j] = (PetscAbs(uRtmp[j] - ubdryR[j]) > M || PetscAbs(uLtmp[j] - ubdryL[j]) > M); 
    if (dgnet->limitactive[j]) {
      limiteractivated = PETSC_TRUE;
    }
  }
  if (limiteractivated) {
    PetscCall(TVDLimit_1D(dgnet,uL,uM,uR,ubdryL,ubdryR,uCoeff,sec,c));
  } 
  /* the uCoeff now contains the limited coefficients */
  PetscFunctionReturn(0);
}
/* basis one-sided limiter, super lame and not robust, detects only if the cell averages of the neighbors are "large" */
PetscErrorCode Limit_1D_onesided(DGNetwork dgnet,const PetscScalar *uL,const PetscScalar *uM, PetscReal *uCoeff, PetscSection sec, PetscInt c, PetscReal jumptol)
{
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
      PetscCall((*dgnet->physics.characteristic)(dgnet->physics.user,dof,uM,dgnet->R,dgnet->Rinv,dgnet->speeds));
      PetscCall(PetscArrayzero(dgnet->cuAvg,dof));

      for(field=0; field<dof; field++) {
        for (k=0; k<dof; k++) {
          dgnet->cuAvg[k]        += dgnet->Rinv[k+field*dof]*uM[field];
        }
      }

      /* Note that we need to expand each basis the the largest DG basis for this to make sense. Thank god 
      the legendre basis is hierarchical (and orthogonal), making this way way easier */ 

      PetscCall(PetscArrayzero(dgnet->charcoeff,dgnet->physics.maxorder+1*dof));
      for(field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldDof(sec,c,field,&fielddeg));
        PetscCall(PetscSectionGetFieldOffset(sec,c,field,&fieldoff));
        for (deg=0;deg<fielddeg;deg++) {
          for (k=0; k<dof; k++) {
            dgnet->charcoeff[k*(dgnet->physics.maxorder+1)+deg]  += dgnet->Rinv[k+field*dof]*uCoeff[fieldoff+deg];
          }
        }
      }
      /* Now the coeffients are in then characterstic variables. Now apply the P0 projection */

      for(j=0; j<dof; j++) {
        if (dgnet->limitactive[j]) {
          PetscCall(PetscArrayzero(dgnet->charcoeff+j*(dgnet->physics.maxorder+1),dgnet->physics.maxorder+1));
          dgnet->charcoeff[j*(dgnet->physics.maxorder+1)] = dgnet->cuAvg[j];  
        }
      } 
      /* Now put the coefficients back into conservative form. Note that 
         as we expanded the DG basis to the maximum order among all field, this 
         technically requires a projection, however the legendre basis 
         is orthogonal and hierarchical, and thus this amounts to simply ignoring higher order terms. 
         
         this does not mess with conservation as the cell averages are unchanged */ 
      PetscCall(PetscSectionGetDof(sec,c,&secdof));
      for(field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldDof(sec,c,field,&fielddeg));
        PetscCall(PetscSectionGetFieldOffset(sec,c,field,&fieldoff));
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
  DGNetwork      dgnet = (DGNetwork)ctx; 
  PetscScalar    *xarr;
  PetscInt       e,c,eStart,eEnd,cStart,cEnd;
  PetscInt       offset,dof,field,fieldoff;
  PetscReal      detJ,M;
  Vec            localX;
  EdgeFE         edgefe; 
  PetscSection   section;

  PetscFunctionBeginUser;
  dof  = dgnet->physics.dof; localX = dgnet->localX;
  PetscCall(DMGlobalToLocalBegin(dgnet->network,Y,INSERT_VALUES,localX));
  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd)); 
  PetscCall(DMGlobalToLocalEnd(dgnet->network,Y,INSERT_VALUES,localX));
  PetscCall(VecGetArray(localX,&xarr));
  /* Iterate through the edges of the network and apply the limiter to each mesh on the edge */
  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd)); 
  for (e=eStart; e<eEnd; e++) {  
    /* Also the update pattern is probably not ideal but I don't care for now */
    PetscCall(DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL));
    PetscCall(DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd));
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    PetscCall(DMGetSection(edgefe->dm,&section));
    for(c=cStart+1; c<cEnd-1; c++) {
      /* make the cell avg arrays and bdry evaluations */
      for(field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
        dgnet->uLR[field]         = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
        dgnet->uLR[field+dof]     = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        dgnet->uavgs[field+dof]   = xarr[offset+fieldoff]; 
        PetscCall(PetscSectionGetFieldOffset(section,c-1,field,&fieldoff));
        dgnet->uavgs[field]       = xarr[offset+fieldoff];
        PetscCall(PetscSectionGetFieldOffset(section,c+1,field,&fieldoff));
        dgnet->uavgs[field+2*dof] = xarr[offset+fieldoff]; 
      }
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,NULL,NULL,&detJ));
      M = detJ*detJ*dgnet->M;
      PetscCall(TVDLimit_1D_2(dgnet,dgnet->uavgs, dgnet->uavgs+dof,dgnet->uavgs+2*dof,dgnet->uLR,dgnet->uLR+dof,xarr+offset,section,c,M));      
      /* 
        TODO : Could print out the limited cells here 
      */ 
    }
    /* Now we limit the bdry cells */
    for(field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldOffset(section,cStart,field,&fieldoff));
      dgnet->uavgs[field+dof]   = xarr[offset+fieldoff]; 
      PetscCall(PetscSectionGetFieldOffset(section,cStart+1,field,&fieldoff));
      dgnet->uavgs[field]   = xarr[offset+fieldoff];
    }
    PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,cStart,NULL,NULL,NULL,&detJ));
    PetscCall(Limit_1D_onesided(dgnet,dgnet->uavgs, dgnet->uavgs+dof,xarr+offset,section,cStart,dgnet->jumptol/detJ));

    for(field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff));
      dgnet->uavgs[field]   = xarr[offset+fieldoff]; 
      PetscCall(PetscSectionGetFieldOffset(section,cEnd-2,field,&fieldoff));
      dgnet->uavgs[field+dof]   = xarr[offset+fieldoff];
    }
    PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,cEnd,NULL,NULL,NULL,&detJ));
    PetscCall(Limit_1D_onesided(dgnet,dgnet->uavgs+dof, dgnet->uavgs,xarr+offset,section,cEnd-1,dgnet->jumptol/detJ));
  }

  /* Data Cleanup */
  PetscCall(VecRestoreArray(localX,&xarr));
  PetscCall(DMLocalToGlobalBegin(dgnet->network,localX,INSERT_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dgnet->network,localX,INSERT_VALUES,Y));
  PetscFunctionReturn(0);
}

/* All this does is call the ctx version with a the ts ctx, and limit the current stage vector */ 
PetscErrorCode DGNetlimiter(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y) {
  DGNetwork      dgnet;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts,&dgnet));
  PetscLogEventBegin(DGNET_Limiter,0,0,0,0);
  PetscCall(DGNetlimiter_ctx(Y[stageindex],dgnet));
  PetscLogEventEnd(DGNET_Limiter,0,0,0,0);
  PetscFunctionReturn(0);
}

/* Nested Version of Limiters. WIP interface to allow multiple simulations to run alongside eachother */ 
PetscErrorCode DGNetlimiter_Nested(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y) {
  DGNetwork_Nest  dgnet_nest;
  PetscInt        i,nestsize,numsim;
  MPI_Comm        comm; 
  Vec             Ysub; 
  PetscBool       isequal; 

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts,&comm));
  PetscCall(TSGetApplicationContext(ts,&dgnet_nest));
  numsim = dgnet_nest->numsimulations;
  /* This routine only works if X,F are VecNest, sanity check here */
  PetscCall(PetscObjectTypeCompare((PetscObject)Y[stageindex],VECNEST,&isequal));
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec Y[stageindex] must be of type VecNest. \n");}
  /* Small sanity check */
  PetscCall(VecNestGetSize(Y[stageindex],&nestsize));
  if (nestsize < numsim) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec X must have at least 1 vector per dgnet simulation.\n  \
  Num Simulation: %i \n Num Vectors %i \n ",numsim,nestsize);}

  /* For each dgnet simulation in dgnet_nest run the limiter*/
  for(i=0; i<numsim; i++) {
    PetscCall(VecNestGetSubVec(Y[stageindex],i,&Ysub)); /* Doesn't need to be returned */
    PetscCall(DGNetlimiter_ctx(Ysub,dgnet_nest->dgnets[i]));
  }
  PetscFunctionReturn(0);
}

/* For Running multiple netrs rhs tests*/
PetscErrorCode DGNetRHS_NETRS_Nested(TS ts,PetscReal time,Vec X,Vec F,void *ctx) 
{
  DGNetwork_Nest  dgnet_nest = (DGNetwork_Nest)ctx;
  PetscInt        i,nestsize,numsim = dgnet_nest->numsimulations;
  MPI_Comm        comm; 
  Vec             Xsub,Fsub;  
  PetscBool       isequal; 

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts,&comm));
  /* This routine only works if X,F are VecNest, sanity check here */
  PetscCall(PetscObjectTypeCompare((PetscObject)X,VECNEST,&isequal));
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec X must be of type VecNest.");}
  PetscCall(PetscObjectTypeCompare((PetscObject)F,VECNEST,&isequal));
  if (!isequal) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec F must be of type VecNest.");}
  /* Small sanity check */
  PetscCall(VecNestGetSize(X,&nestsize));
  if (nestsize < numsim) {SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec X must have at least 1 vector per dgnet simulation.\n \
Num Simulation: %i \n Num Vectors %i \n ",numsim,nestsize);}
  /* For each dgnet simulation in dgnet_nest run the DG kernel */
  for(i=0; i<numsim; i++) {
    PetscCall(VecNestGetSubVec(X,i,&Xsub)); /* Don't need to be returned */
    PetscCall(VecNestGetSubVec(F,i,&Fsub));
    PetscCall(DGNetRHS(ts,time,Xsub,Fsub,dgnet_nest->dgnets[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DGNetRHS(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr; 
  DGNetwork      dgnet = (DGNetwork)ctx;    
  PetscReal      maxspeed,detJ,J,invJ,*numflux;
  PetscScalar    *f,*xarr,*coeff,*riemanndata,*flux; 
  PetscInt       v,e,c,vStart,vEnd,eStart,eEnd,vfrom,vto,cStart,cEnd,q,deg,ndeg,quadsize,tab,face,fStart,fEnd,voff; 
  PetscInt       offset,nedges,i,dof = dgnet->physics.dof,field,fieldoff;
  const PetscInt *cone,*edges,*supp;
  Vec            localX = dgnet->localX,localF = dgnet->localF; 
  EdgeFE         edgefe; 
  PetscSection   section;
  const PetscReal *qweight;
  RiemannSolver   rs = dgnet->physics.rs; 


  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(localF));
  PetscLogEventBegin(DGNET_RHS_COMM,0,0,0,0);
  PetscCall(DMGlobalToLocalBegin(dgnet->network,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(dgnet->network,X,INSERT_VALUES,localX));
  PetscLogEventEnd(DGNET_RHS_COMM,0,0,0,0);

  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd)); 
  PetscCall(VecGetArray(localX,&xarr));
  PetscCall(VecGetArray(localF,&f));
  PetscLogEventBegin(DGNET_RHS_Vert,0,0,0,0); 
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(dgnet->network,&vStart,&vEnd);
  PetscCall(VecZeroEntries(dgnet->RiemannData));
  PetscCall(VecGetArray(dgnet->RiemannData,&riemanndata));
  for (v=vStart; v<vEnd; v++) {
    /* Evaluate DG solution at the vertices and store for NetRS */
    PetscCall(DMNetworkGetSupportingEdges(dgnet->network,v,&nedges,&edges));
    PetscCall(NetRSGetVertexVecOffset(dgnet->netrs,v,&voff,NULL));
    for (i=0; i<nedges; i++) {
      e     = edges[i];
      PetscCall(DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset));
      PetscCall(DMNetworkGetConnectedVertices(dgnet->network,e,&cone));
      PetscCall(DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL));
      /* DMPlex stuff here, get cell chart */
      PetscCall(DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd));
      PetscCall(DMGetSection(edgefe->dm,&section));
      vfrom = cone[0];
      vto   = cone[1];
      if (v == vfrom) {
        /* left eval */
        for (field=0; field<dof; field++) {
          PetscCall(PetscSectionGetFieldOffset(section,cStart,field,&fieldoff));
          riemanndata[voff+field+dof*i] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
        }
      } else if (v == vto) {
        for (field=0; field<dof; field++) {
          PetscCall(PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff));
          riemanndata[voff+field+dof*i] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        }
      }
    }
  }
  PetscCall(VecRestoreArray(dgnet->RiemannData,&riemanndata));
  /* NetRS solve */
  PetscCall(NetRSSolveFlux(dgnet->netrs,dgnet->RiemannData,dgnet->Flux));
  PetscLogEventEnd(DGNET_RHS_Vert,0,0,0,0); 

  PetscLogEventBegin(DGNET_Edge_RHS,0,0,0,0);
  PetscCall(VecGetArray(dgnet->Flux,&flux)); 
  /* Do DG for each edge */
  PetscCall(DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd)); 
  for (e=eStart; e<eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL));
    PetscCall(DMNetworkGetLocalVecOffset(dgnet->network,e,FVEDGE,&offset));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd));
    /* We will manually use the section for now to deal with indexing offsets etc.. to be redone */
    PetscCall(DMGetSection(edgefe->dm,&section));
    PetscCall(PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,&qweight));
    /* Iterate through the cells of the edge mesh */
    for(c=cStart; c<cEnd; c++) {
      /* Get Geometric Data */
      /* Assumes Affine coordinates for now (And 1D everything!!) (and I think assumes same embedding dimension as topological ) */
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ));
      /* Now we do the main integral \int_K flux(u)\phi_x \dx  on the reference element*/ 
      /* First we evaluate the flux(u) at the quadrature points */
      for(q=0; q<quadsize; q++) {
        for(field = 0; field<dof; field++) {
          PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
          coeff = xarr+offset+fieldoff;
          dgnet->pteval[field] = evalquad_internal(dgnet,field,q,coeff);
        }
        dgnet->physics.flux((void*)dgnet->physics.user,dgnet->pteval,dgnet->fluxeval+q*dof);
      }
      /* Now we can compute quadrature for each integral for each field */
      for(field = 0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
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
    PetscCall(DMNetworkGetConnectedVertices(dgnet->network,e,&cone));
    vfrom  = cone[0];
    vto    = cone[1];
    /*cStart cell */

    /* Need to redo this computation as it doesn't work correclty and recquires a search ... */
    PetscCall(NetRSGetVertexVecOffset(dgnet->netrs,vfrom,&voff,NULL));
    PetscCall(DMNetworkGetSupportingEdges(dgnet->network,vfrom,&nedges,&edges)); 
    i=0; 
    while(edges[i]!=e) i++; 

    /* Update the vfrom vertex flux for this edge */
    for (field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldOffset(section,cStart,field,&fieldoff));
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff += flux[field+i*dof+voff]*dgnet->LegEvaL_bdry[tab][deg];
      }
    }
    /* cEnd cell */
    PetscCall(NetRSGetVertexVecOffset(dgnet->netrs,vto,&voff,NULL));
    PetscCall(DMNetworkGetSupportingEdges(dgnet->network,vto,&nedges,&edges)); 
    i=0; 
    while(edges[i]!=e) i++; 
    /* Update the vto vertex flux for this edge */
    for (field=0; field<dof; field++) {
      PetscCall(PetscSectionGetFieldOffset(section,cEnd-1,field,&fieldoff));
      tab = dgnet->fieldtotab[field];
      ndeg = dgnet->taborder[tab]+1;
      for (deg = 0; deg<ndeg; deg++) {
        coeff = f+offset+fieldoff+deg;
        *coeff -= flux[field+i*dof+voff]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
      }
    }
    /* 2) Then iterate through the flux updates */
    /* we iterate through the 1 codim cells (faces) skipping the first and last to compute the numerical fluxes and update the resulting cells coefficients */
    PetscCall(DMPlexGetHeightStratum(edgefe->dm,1,&fStart,&fEnd));
    for(face=fStart+1; face<fEnd-1; face++) {
      /* WE ASSUME 1D HERE WITH SUPPORT SIZE OF 2 !!!! */
      PetscCall(DMPlexGetSupport(edgefe->dm,face,&supp));
      /* evaluate at the face */
      for(field = 0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff));
        dgnet->uLR[field] = evalboundary_internal(dgnet,field,1,xarr+offset+fieldoff);
        PetscCall(PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff));
        dgnet->uLR[field+dof] = evalboundary_internal(dgnet,field,0,xarr+offset+fieldoff);
      }
      PetscCall(RiemannSolverEvaluate(rs,dgnet->uLR,dgnet->uLR+dof,&numflux,&maxspeed));
      /* Update coefficents with the numerical flux */
      for (field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,supp[0],field,&fieldoff));
        tab = dgnet->fieldtotab[field];
        ndeg = dgnet->taborder[tab]+1;
        for (deg = 0; deg<ndeg; deg++) {
          coeff = f+offset+fieldoff+deg;
          *coeff -= numflux[field]*dgnet->LegEvaL_bdry[tab][ndeg+deg];
        }
      }
      for (field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,supp[1],field,&fieldoff));
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
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&invJ,&detJ));
      for(field=0; field<dof; field++) {
        PetscCall(PetscSectionGetFieldOffset(section,c,field,&fieldoff));
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
  PetscLogEventEnd(DGNET_Edge_RHS,0,0,0,0);
  /* Data Cleanup */
  PetscCall(VecRestoreArray(localX,&xarr));
  PetscCall(VecRestoreArray(dgnet->Flux,&flux));
  PetscCall(VecRestoreArray(localF,&f));
  PetscLogEventBegin(DGNET_RHS_COMM,0,0,0,0);
  PetscCall(DMLocalToGlobalBegin(dgnet->network,localF,INSERT_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dgnet->network,localF,INSERT_VALUES,F));
  PetscLogEventEnd(DGNET_RHS_COMM,0,0,0,0);

  PetscFunctionReturn(0);
}
