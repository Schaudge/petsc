#include "fvnet.h"

PetscErrorCode FVNetworkCreate(FVNetwork fvnet,PetscInt initial,PetscInt Mx)
{
  PetscErrorCode ierr;
  PetscInt       nfvedge;
  PetscMPIInt    rank;
  PetscInt       i,numVertices,numEdges;
  PetscInt       *edgelist;
  Junction       junctions = NULL;
  FVEdge         fvedges = NULL;
  PetscInt       dof = fvnet->physics.dof; 
  
  PetscFunctionBegin;
  fvnet->nnodes_loc = 0;
  ierr              = MPI_Comm_rank(fvnet->comm,&rank);CHKERRQ(ierr);
  numVertices       = 0;
  numEdges          = 0;
  edgelist          = NULL;
  fvnet->initial    = initial; 
  /* proc[0] creates a sequential fvnet and edgelist */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Setup initial %D\n",initial);CHKERRQ(ierr);
  /* Set global number of fvedges, edges, and junctions */
  /*-------------------------------------------------*/
  switch (initial) {
    case 0:
      /* Case 0: */
      /* =================================================
      (OUTFLOW) v0 --E0--> v1--E1--> v2 --E2-->v3 (OUTFLOW)
      ====================================================  */
      nfvedge        = 3;
      fvnet->nedge   = nfvedge;
      fvnet->nvertex = nfvedge + 1;
      /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
      numVertices    = 0;
      numEdges       = 0;
      edgelist       = NULL;
      if (!rank) {
        numVertices = fvnet->nvertex;
        numEdges    = fvnet->nedge;
        ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);

        for (i=0; i<numEdges; i++) {
          edgelist[2*i] = i; 
          edgelist[2*i+1] = i+1;
        }
        /* Add network components */
        /*------------------------*/
        ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
        /* vertex */
        junctions[0].type = OUTFLOW;
        junctions[1].type = JUNCT;
        junctions[2].type = JUNCT;
        junctions[3].type = OUTFLOW;

        for (i=0; i<numVertices; i++) {
          junctions[i].x = i*1.0/3.0*50.0; 
        }
        /* Edge */ 
        fvedges[0].nnodes = Mx; 
        fvedges[1].nnodes = fvnet->hratio*Mx; 
        fvedges[2].nnodes = Mx; 

        for (i=0; i<numEdges;i++) {
          fvedges[i].h = 1.0/3.0/(PetscReal)fvedges[i].nnodes*50.0; 
        }
      }
      break;
    case 1:
      /* Case 1: */
      /* =================================================
      (OUTFLOW) v0 --E0--> v1 (OUTFLOW)
      ====================================================  */
      nfvedge        = 1;
      fvnet->nedge   = nfvedge;
      fvnet->nvertex = nfvedge + 1;
      /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
      numVertices    = 0;
      numEdges       = 0;
      edgelist       = NULL;
      if (!rank) {
        numVertices = fvnet->nvertex;
        numEdges    = fvnet->nedge;
        ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);

        for (i=0; i<numEdges; i++) {
          edgelist[2*i] = i; 
          edgelist[2*i+1] = i+1;
        }
        /* Add network components */
        /*------------------------*/
        ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
        /* vertex */
        junctions[0].type = OUTFLOW;
        junctions[1].type = OUTFLOW;

        for (i=0; i<numVertices; i++) {
          junctions[i].x = i*1.0*50.0; 
        }
        /* Edge */ 
        fvedges[0].nnodes = Mx;
        
        for (i=0; i<numEdges; i++) {
          fvedges[i].h = 1.0/(PetscReal)fvedges[i].nnodes*50.0; 
        }
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }
  /* set edge global id */
  for (i=0; i<numEdges; i++) fvedges[i].id = i;
  /* set junction global id */
  for (i=0; i<numVertices; i++) junctions[i].id = i; 
  fvnet->nedge    = numEdges;
  fvnet->nvertex  = numVertices;
  fvnet->edgelist = edgelist;
  fvnet->junction = junctions;
  fvnet->fvedge   = fvedges;
  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc4(dof*dof,&fvnet->R,dof*dof,&fvnet->Rinv,2*dof,&fvnet->cjmpLR,1*dof,&fvnet->cslope);CHKERRQ(ierr);
  ierr = PetscMalloc4(2*dof,&fvnet->uLR,dof,&fvnet->flux,dof,&fvnet->speeds,dof,&fvnet->uPlus);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkSetComponents(FVNetwork fvnet){
  PetscErrorCode    ierr;  
  PetscInt          i,j,e,v,eStart,eEnd,vStart,vEnd,dof = fvnet->physics.dof;
  PetscInt          KeyEdge,KeyJunction,KeyFlux,vfrom,vto,nedges_tmp,nedges,nvertices; 
  PetscInt          *edgelist = NULL,*edgelists[1];
  FVEdge            fvedge;
  Junction          junction,junctions;
  MPI_Comm          comm = fvnet->comm;
  PetscMPIInt       size,rank;
  PetscReal         length;
  const PetscInt    *cone,*edges;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  nedges      = fvnet->nedge;
  nvertices   = fvnet->nvertex; /* local num of vertices, excluding ghosts */
  edgelist    = fvnet->edgelist;
  junctions   = fvnet->junction;
  fvedge      = fvnet->fvedge;
  /* Set up the network layout */
  ierr = DMNetworkSetSizes(fvnet->network,1,&nvertices,&nedges,0,NULL);CHKERRQ(ierr);
  /* Add local edge connectivity */
  edgelists[0] = edgelist;
  ierr = DMNetworkSetEdgeList(fvnet->network,edgelists,NULL);CHKERRQ(ierr);
  ierr = DMNetworkLayoutSetUp(fvnet->network);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"fvedgestruct",sizeof(struct _p_FVEdge),&KeyEdge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"empty",0,&KeyFlux);CHKERRQ(ierr);
  /* Add FVEdge component to all local edges. Note that as we have 
     yet to distribute the network, all data is on proc[0]. */
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkAddComponent(fvnet->network,e,KeyEdge,&fvedge[e-eStart]);CHKERRQ(ierr);
    ierr = DMNetworkAddNumVariables(fvnet->network,e,dof*fvedge[e-eStart].nnodes);CHKERRQ(ierr);
    /* Add a monitor for every edge in the network, label the data according the user provided physics */
    if (size == 1 && fvnet->monifv) { 
      length = fvedge[e-eStart].h*(fvedge[e-eStart].nnodes+1);
      for (j=0; j<dof; j++) {
         ierr = DMNetworkMonitorAdd(fvnet->monitor,fvnet->physics.fieldname[j],e,fvedge[e-eStart].nnodes,j,dof,0.0,length,fvnet->ymin,fvnet->ymax,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  }
  /* Add Junction component to all local vertices. All data is currently assumed to be on proc[0]. Also add the flux component */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkAddComponent(fvnet->network,v,KeyJunction,&junctions[v-vStart]);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges_tmp,&edges);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(fvnet->network,v,JUNCTION,0);CHKERRQ(ierr);
    /* Add data structure primarily for moving the vertex fluxes around. Is used throughout 
       passing various data between processors. */
    ierr = DMNetworkAddComponent(fvnet->network,v,KeyFlux,NULL);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(fvnet->network,v,FLUX,dof*nedges_tmp);CHKERRQ(ierr);
  }
  ierr = DMSetUp(fvnet->network);CHKERRQ(ierr);
  /* Build the edge offset data to allow for a sensible local ordering of the 
     edges of a vertex. Needed so that the data belonging to a vertex knows
     which edge each piece should interact with. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges_tmp,&edges);CHKERRQ(ierr);
    junction->numedges = nedges_tmp;
    /* Iterate through the connected edges. As we are on a single processor, DMNetworkGetSupportingEdges which returns 
       on processor edges, will be returning ALL connected edges on the graph.  */
    for (i=0; i<nedges_tmp; i++) {
      e     = edges[i];   
      ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
      ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];
      if (v==vto) {
        fvedge->offset_vto = dof*i; 
      } else if (v==vfrom) {
        fvedge->offset_vfrom = dof*i; 
      } else {
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"v %D != vfrom or vto from supporting edge %D",v,e);
      }
    }
  }
  /* Initialize fvedge variables */
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    fvedge->frombufferlvl = 0; 
    fvedge->tobufferlvl   = 0; 
  }
  PetscFunctionReturn(0);
}
/* Now we have a distributed network. It is assumed that localX and Ftmp have been created in fvnet */
PetscErrorCode FVNetworkBuildDynamic(FVNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscInt       e,v,i,nedges,dof = fvnet->physics.dof;
  PetscInt       eStart,eEnd,vStart,vEnd,vfrom,vto,offset;
  const PetscInt *cone,*edges; 
  FVEdge         fvedge; 
  Junction       junction;
  Vec            localX = fvnet->localX; 
  PetscScalar    *xarr; 

  PetscFunctionBegin;
  ierr   = VecSet(fvnet->Ftmp,0.0);CHKERRQ(ierr);
  ierr   = VecSet(localX,0.0);CHKERRQ(ierr);
  ierr   = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr   = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr   = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  /* Build the data so that vertex knows what edges point into it, and which edges point out.
     We temporarily use the flux component to set up this structure. At the end it will be locally 
     stored, but we have to do a message-passing start up to get all of the right 
     information onto the local processors. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offset);CHKERRQ(ierr);
    /* Iterate through the (local) connected edges. Each ghost vertex of a vertex connects to a 
       a non-overlapping set of local edges. This is why we can iterate in this way without 
       potentially conflicting our scatters.*/
    for (i=0; i<nedges; i++) { 
      e     = edges[i];  
      ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void **)&fvedge);CHKERRQ(ierr);
      ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1]; 
      if (v==vto) {
        xarr[offset+fvedge->offset_vto]   = EDGEIN;
      } else if (v==vfrom) {
        xarr[offset+fvedge->offset_vfrom] = EDGEOUT;
      } else {
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D != vfrom or vto from supporting edge %D",v,e);
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr); 
  ierr = DMLocalToGlobalBegin(fvnet->network,localX,ADD_VALUES,fvnet->Ftmp);CHKERRQ(ierr);  
  ierr = DMLocalToGlobalEnd(fvnet->network,localX,ADD_VALUES,fvnet->Ftmp);CHKERRQ(ierr);
  /* Now the flux components hold the edgein/edgeout information for all edges connected to the vertex (not just the local edges) */
  ierr = DMGlobalToLocalBegin(fvnet->network,fvnet->Ftmp,INSERT_VALUES,localX);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,fvnet->Ftmp,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  /* Iterate through all vertices and build the junction component data structure dir. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offset);CHKERRQ(ierr);
    ierr = PetscMalloc1(junction->numedges,&(junction->dir));CHKERRQ(ierr); /* Freed in the final cleanup call*/
    /* Fill in the local dir data */
    for (i=0; i<junction->numedges; i++) { 
      junction->dir[i] = xarr[offset+i*dof];
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}
PetscErrorCode FVNetworkCleanUp(FVNetwork fvnet)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank; 

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fvnet->comm,&rank);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->edgelist);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFree2(fvnet->junction,fvnet->fvedge);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkCreateVectors(FVNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(fvnet->network,&fvnet->X);CHKERRQ(ierr);
  ierr = VecDuplicate(fvnet->X,&fvnet->Ftmp);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localX);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkDestroy(FVNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       i,v,vStart,vEnd;
  Junction       junction;

  PetscFunctionBegin; 
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = PetscFree(junction->dir);CHKERRQ(ierr); /* Free dynamic memory for the junction component */
  }
  ierr = (*fvnet->physics.destroy)(fvnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<fvnet->physics.dof; i++) {
    ierr = PetscFree(fvnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(fvnet->R,fvnet->Rinv,fvnet->cjmpLR,fvnet->cslope);CHKERRQ(ierr);
  ierr = PetscFree4(fvnet->uLR,fvnet->flux,fvnet->speeds,fvnet->uPlus);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->X);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->Ftmp);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->localX);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->localF);CHKERRQ(ierr);
  ierr = ISDestroy(&fvnet->slow_edges);CHKERRQ(ierr);
  ierr = ISDestroy(&fvnet->slow_vert);CHKERRQ(ierr);
  ierr = ISDestroy(&fvnet->buf_slow_vert);CHKERRQ(ierr);
  ierr = ISDestroy(&fvnet->fast_edges);CHKERRQ(ierr);
  ierr = ISDestroy(&fvnet->fast_vert);CHKERRQ(ierr);
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
  PetscReal      h,maxspeed,cfl_idt = 0;
  PetscScalar    *f,*uL,*uR,*xarr;
  PetscInt       v,e,vStart,vEnd,eStart,eEnd,vfrom,vto;
  PetscInt       offsetf,offset,nedges,nnodes,i,j,dof = fvnet->physics.dof;;
  const PetscInt *cone,*edges;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all vertices (including ghosts) and compute the flux/reconstruction data for the vertex.  */
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
          for (i=0; i<nedges; i++) {
            e     = edges[i];
            ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
            ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
            ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
          ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
  /* Now All processors have the reconstruction data to compute the coupling flux */
  for (v=vStart; v<vEnd; v++) {
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        /* Now compute the coupling flux */
        if (junction->numedges > 2) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has more than 2 connected edges. Coupling flux supports only up to 2 edges currently",v);
        } else { /* Note that in creation a junction must have at least 2 connected edges */
          if (junction->dir[0] == EDGEIN) {
            uL = f+offsetf;
            uR = f+offsetf+dof;
          } else { /* EDGEOUT */
            uL = f+offsetf+dof;
            uR = f+offsetf;
          }
          fvnet->couplingflux(fvnet->physics.user,dof,uL,uR,fvnet->flux,&maxspeed);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = fvnet->flux[j];
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
  /* Now all the vertex flux data is available on each processor. */
  /* Iterate through the edges and update the cell data belonging to that edge. */
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for (e=eStart; e<eEnd; e++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
       4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
    ierr   = DMNetworkGetComponentVariableOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
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
      cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
      for( j=0; j<dof; j++) {
        f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
        f[offset+i*dof+j]     += fvnet->flux[j]/h;
      }
    }
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
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
    cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&fvnet->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,fvnet->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Multirate Non-buffer RHS */
PetscErrorCode FVNetRHS_Multirate(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed,cfl_idt = 0;
  PetscScalar    *f,*uL,*uR,*xarr;
  PetscInt       i,j,k,ne,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist,*edgelist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;
  VecScatter     scatter;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(fvnet->network,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  /* Iterate through all marked vertices (including ghosts) and compute the flux/reconstruction data for the vertex. */
  ierr = ISGetLocalSize(rhsctx->edgelist,&ne);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rhsctx->vtxlist,&nv);CHKERRQ(ierr);
  ierr = ISGetIndices(rhsctx->edgelist,&edgelist);CHKERRQ(ierr); 
  ierr = ISGetIndices(rhsctx->vtxlist,&vtxlist);CHKERRQ(ierr);
  for (k=0; k<nv; k++) {
    v = vtxlist[k];
    /* Reconstruct all local edge data points */
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction); 
    switch (junction->type) {
      case JUNCT:
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
          ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction); 
    switch (junction->type) {
      case JUNCT:
        /* Now compute the coupling flux */
        if (junction->numedges > 2) {
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has more than 2 connected edges. Coupling flux supports only up to 2 edges currently",v);
        } else { /* Note that in creation a junction must have at least 2 connected edges */
          if (junction->dir[0] == EDGEIN) {
            uL = f+offsetf;
            uR = f+offsetf+dof;
          } else { /* EDGEOUT */
            uL = f+offsetf+dof;
            uR = f+offsetf;
          }
          fvnet->couplingflux(fvnet->physics.user,dof,uL,uR,fvnet->flux,&maxspeed);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = fvnet->flux[j];
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
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through the edges and update the cell data belonging to that edge */
  for (k=0; k<ne; k++) {
    /* The cells are updated in the order 1) vfrom vertex flux 2) special 2nd flux (requires special reconstruction) 3) interior cells 
    4) 2nd to last flux (requires special reconstruction) 5) vto vertex flux */
    e      = edgelist[k];
    ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom  = cone[0];
    vto    = cone[1];
    ierr   = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
    ierr   = DMNetworkGetComponentVariableOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    ierr   = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h      = fvedge->h;
    nnodes = fvedge->nnodes;
    if (!fvedge->frombufferlvl) { /* Evaluate from buffer data */
      /* Update the vfrom vertex flux for this edge */
      for (j=0; j<dof; j++) {
        f[offset+j] += f[fvedge->offset_vfrom+j+offsetf]/h;
      }
      /* Now reconstruct the value at the left cell of the 1/2 interface. I have to redo code here, should alter 
        how I compute things to avoid this.  */
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
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
      cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
    cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
      cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
    cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
    for (j=0; j<dof; j++) {
      f[offset+(nnodes-bufferwidth-1)*dof+j] -= fvnet->flux[j]/h;
      if (!fvedge->tobufferlvl) f[offset+(nnodes-bufferwidth)*dof+j] += fvnet->flux[j]/h;
    }
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
    /* Iterate through the remaining cells */
    if (!fvedge->tobufferlvl) {
      for (i=nnodes-bufferwidth+1; i<(nnodes-1); i++) {
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for (j=0; j<dof; j++) {
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for (j=0; j<dof; j++) {
          f[offset+(i-1)*dof+j] -= fvnet->flux[j]/h;
          f[offset+i*dof+j]     += fvnet->flux[j]/h;
        }
      }
      /* Now reconstruct the value at the 2nd to last interface . I have to redo code here, should alter 
          how I compute things to avoid this. */
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
      cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
  ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&fvnet->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,fvnet->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Multirate buffer RHS */
PetscErrorCode FVNetRHS_Buffer(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  RhsCtx         *rhsctx = (RhsCtx*)ctx;
  FVNetwork      fvnet = rhsctx->fvnet;
  PetscReal      h,maxspeed,cfl_idt = 0;
  PetscScalar    *f,*uL,*uR,*xarr;
  PetscInt       i,j,k,m,nv,dof = fvnet->physics.dof,bufferwidth = fvnet->bufferwidth;
  PetscInt       v,e,vfrom,vto,offsetf,offset,nedges,nnodes;
  const PetscInt *cone,*edges,*vtxlist;
  Vec            localX = fvnet->localX,localF = fvnet->localF,Ftmp = fvnet->Ftmp; 
  FVEdge         fvedge; 
  Junction       junction;
  VecScatter     scatter;

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
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    switch (junction->type) {
      case JUNCT:
        /* Reconstruct all local edge data points */
        ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
        for (i=0; i<nedges; i++) {
          e     = edges[i];
          ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
          ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
          ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
          ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
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
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction); 
    switch (junction->type) {
      case JUNCT:
        /* Now compute the coupling flux */
        if (junction->numedges > 2){
          SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has more than 2 connected edges. Coupling flux supports only up to 2 edges currently",v);
        } else { /* Note that in creation a junction must have at least 2 connected edges */
          if (junction->dir[0] == EDGEIN) {
            uL = f+offsetf;
            uR = f+offsetf+dof;
          } else { /* EDGEOUT */
            uL = f+offsetf+dof;
            uR = f+offsetf;
          }
          fvnet->couplingflux(fvnet->physics.user,dof,uL,uR,fvnet->flux,&maxspeed);
          for (i=0; i<junction->numedges; i++) {
            for (j=0; j<dof; j++) {
              f[offsetf+i*dof+j] = fvnet->flux[j];
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
  /* Now all the vertex flux data is available on each processor for the multirate partition. */
  /* Iterate through buffer vertices and update the buffer edges connected to it. */
  for (k=0; k<nv; k++) {
    v    = vtxlist[k]; 
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (m=0; m<nedges; m++) {
      /* Update the buffer regions */
      e      = edges[m];
      ierr   = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
      ierr   = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom  = cone[0];
      vto    = cone[1];
      ierr   = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
      h      = fvedge->h;
      nnodes = fvedge->nnodes;
      if (v==vfrom && fvedge->frombufferlvl) {
        /* ||Update the from buffer region of edge e|| */
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
          cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
          cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
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
  /* Move Data into the expected format from the multirate ode  */
  ierr = VecScatterCreate(Ftmp,rhsctx->wheretoputstuff,F,NULL,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,Ftmp,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&fvnet->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,fvnet->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkSetInitial(FVNetwork fvnet,Vec X0) 
{
  PetscErrorCode ierr;
  PetscInt       i,vfrom,vto,type,offset,e,eStart,eEnd,dof = fvnet->physics.dof;
  const PetscInt *cone;
  PetscScalar    *xarr,*u;
  Junction       junction;
  FVEdge         fvedge;
  Vec            localX = fvnet->localX;
  PetscReal      h,xfrom,xto,x;
  
  PetscFunctionBegin;
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,&type,(void**)&fvedge);CHKERRQ(ierr);
    ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h     = fvedge->h;
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    xto   = junction->x;
    ierr  = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    xfrom = junction->x;
    /* This code assumes a geometrically 1d network. To be improved later */
    for (i=0; i<fvedge->nnodes; i++) {
      if (xto>xfrom) {
        x = xfrom+i*h;
      } else {
        x = xfrom-i*h;
      }
      u = xarr+offset+i*dof; 
      switch (fvnet->initial) {
        case 0: 
        case 1: 
          /*Both are networks on [0,1] and so use the same initial conditions. User provided geometrically 1d initial conditions */
          fvnet->physics.sample((void*)&fvnet->physics.user,fvnet->subcase,0.0,x,u);
          break;
        default: 
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  /* Can use insert as each edge belongs to a single processor and vertex data is only for temporary computation and holds no 'real' data. */
  ierr = DMLocalToGlobalBegin(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
/* Specific multirate partitions for test examples. */
PetscErrorCode FVNetworkGenerateMultiratePartition_Preset(FVNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       id,e,eStart,eEnd,slow_edges_count = 0,fast_edges_count = 0,slow_edges_size = 0,fast_edges_size = 0;
  PetscInt       *slow_edges,*fast_edges;
  FVEdge         fvedge;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  switch (fvnet->initial) {
    case 0: /* Mark the boundary edges as slow and the middle edge as fast */
      /* Find the number of slow/fast edges */
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id || id == 2) {
          slow_edges_size++;
        } else if (id == 1) { 
          fast_edges_size++;
        }
      }
      /* Data will be owned and deleted by the IS*/
      ierr = PetscMalloc1(slow_edges_size,&slow_edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(fast_edges_size,&fast_edges);CHKERRQ(ierr);
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id || id == 2) {
          slow_edges[slow_edges_count] = e; 
          slow_edges_count++; 
        } else if (id == 1) { 
          fast_edges[fast_edges_count] = e; 
          fast_edges_count++; 
        }
      }
      /* Generate IS */
      ierr = ISCreateGeneral(MPI_COMM_SELF,slow_edges_size,slow_edges,PETSC_OWN_POINTER,&fvnet->slow_edges);CHKERRQ(ierr);
      ierr = ISCreateGeneral(MPI_COMM_SELF,fast_edges_size,fast_edges,PETSC_OWN_POINTER,&fvnet->fast_edges);CHKERRQ(ierr);
      break; 
    case 1: /* Mark the middle edge as slow */
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id) {
          slow_edges_size++;
        } 
      }
      /* Data will be owned and deleted by the IS*/
      ierr = PetscMalloc1(slow_edges_size,&slow_edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(fast_edges_size,&fast_edges);CHKERRQ(ierr);
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id) {
          slow_edges[slow_edges_count] = e; 
          slow_edges_count++; 
        }
      }
      /* Generate IS */
      ierr = ISCreateGeneral(MPI_COMM_SELF,slow_edges_size,slow_edges,PETSC_OWN_POINTER,&fvnet->slow_edges);CHKERRQ(ierr);
      ierr = ISCreateGeneral(MPI_COMM_SELF,fast_edges_size,fast_edges,PETSC_OWN_POINTER,&fvnet->fast_edges);CHKERRQ(ierr);
      break;
    default: 
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }
  PetscFunctionReturn(0);
}
/* Builds the vertex lists and marks the edges as buffer zones as neccessary. Assumes that fvnet->fast_edges and fvnet->slow_edges 
   have already been built. */
PetscErrorCode FVNetworkFinalizePartition(FVNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       i,vfrom,vto,e,eStart,eEnd,v,vStart,vEnd,offsetf,ne,dof = fvnet->physics.dof;
  PetscInt       *buf_vert,*fast_vert,*slow_vert,*vert_sizes,*vert_count;
  PetscInt       buf_count = 0, buf_size = 0,edgemarking,numlvls = 2; /* Number of multirate levels, assumed two (fast/slow) for now */ 
  const PetscInt *cone,*edges;
  Junction       junction;
  FVEdge         fvedge;
  Vec            Ftmp = fvnet->Ftmp, localF = fvnet->localF; /* Used for passing marking data between processors */
  PetscScalar    *f;
  PetscBool      *hasmarkededge; 

  enum {SLOW=1,FAST=2}; /* for readability */
  
  PetscFunctionBegin;
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  /* We assume fvnet->slow_edges and fvnet->fast_edges are built. We communicate these edge markings into the 
     the junction edgemarkings data. This allows us to mark a junction as fast/slow/buffer */
  /* copy slow edge data */
  ierr = ISGetIndices(fvnet->slow_edges,&edges);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->slow_edges,&ne);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    e     = edges[i];
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vfrom] = SLOW;
    ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vto]   = SLOW; 
  }
  ierr = ISGetIndices(fvnet->fast_edges,&edges);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->fast_edges,&ne);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    e     = edges[i];
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vfrom] = FAST;
    ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vto]   = FAST; 
  }
  /* Now communicate the marking data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Allocate size based on number of marking levels possible. Assume two (fast/slow) for now */
  ierr = PetscMalloc1(numlvls,&hasmarkededge);CHKERRQ(ierr);
  ierr = PetscMalloc2(numlvls,&vert_sizes,numlvls,&vert_count);CHKERRQ(ierr);
  for (i=0; i<numlvls; i++) {
    vert_sizes[i] = 0; 
    vert_count[i] = 0; 
  }
  /* Find the sizes of the vertex lists */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (i=0; i<numlvls; i++) {
      hasmarkededge[i] = PETSC_FALSE; 
    } 
    for (i=0; i<junction->numedges; i++) {
      edgemarking                  = f[offsetf+i*dof];
      hasmarkededge[edgemarking-1] = PETSC_TRUE; 
    }
    /* Add to slow/fast/... lists */
    for (i=0; i<numlvls; i++) {
      if (hasmarkededge[i]) vert_sizes[i]++; 
    }
    /* Add to buffer lists as necessary (assumes numlvl == 2 for now ) */
    if (hasmarkededge[0] && hasmarkededge[1]) {
      buf_size++; 
    }
  }
  /* Data will be owned and deleted by the IS */
  ierr = PetscMalloc1(vert_sizes[0],&slow_vert);CHKERRQ(ierr);
  ierr = PetscMalloc1(vert_sizes[1],&fast_vert);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size,&buf_vert);CHKERRQ(ierr);
  /* Build the vertex lists */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (i=0; i<numlvls; i++) {
      hasmarkededge[i] = PETSC_FALSE; 
    } 
    for (i=0; i<junction->numedges; i++) {
      edgemarking                  = f[offsetf+i*dof];
      hasmarkededge[edgemarking-1] = PETSC_TRUE; 
    }
    /* Add to slow/fast/... lists (assumes numlvl = 2) */
    if (hasmarkededge[0]) {
      slow_vert[vert_count[0]] = v;
      vert_count[0]++; 
    }
    if (hasmarkededge[1]) {
      fast_vert[vert_count[1]] = v; 
      vert_count[1]++; 
    }    
    /* Add to buffer lists as necessary (assumes numlvl == 2 for now ) */
    if (hasmarkededge[0] && hasmarkededge[1]) {
      buf_vert[buf_count] = v; 
      buf_count++;
      ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&ne,&edges);CHKERRQ(ierr);
      for (i=0; i<ne; i++) { /* Mark the connected slow edges as buffer edges */
        e     = edges[i];
        ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        vfrom = cone[0]; 
        vto   = cone[1];
        if (v == vfrom) {
          if (f[offsetf+fvedge->offset_vfrom] == SLOW) fvedge->frombufferlvl = 1;
        } else if (v == vto) {
          if (f[offsetf+fvedge->offset_vto] == SLOW)   fvedge->tobufferlvl   = 1; 
        }
      }
    }
  }
  /* Generate IS */
  ierr = ISCreateGeneral(MPI_COMM_SELF,vert_sizes[0],slow_vert,PETSC_OWN_POINTER,&fvnet->slow_vert);CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,vert_sizes[1],fast_vert,PETSC_OWN_POINTER,&fvnet->fast_vert);CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,buf_size,buf_vert,PETSC_OWN_POINTER,&fvnet->buf_slow_vert);CHKERRQ(ierr);
  /* Free Data */
  ierr = PetscFree2(vert_sizes,vert_count);CHKERRQ(ierr); 
  ierr = PetscFree(hasmarkededge);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Specifically for slow/slowbuffer/fast multirate partitioning. A general function is needed later. */
PetscErrorCode FVNetworkBuildMultirateIS(FVNetwork fvnet, IS *slow, IS *fast, IS *buffer) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,offset,e,dof = fvnet->physics.dof;
  PetscInt       *i_slow,*i_fast,*i_buf;
  const PetscInt *index; 
  PetscInt       slow_size = 0,fast_size = 0,buf_size = 0,size,bufferwidth = fvnet->bufferwidth; 
  PetscInt       slow_count = 0,fast_count = 0, buf_count = 0; 
  FVEdge         fvedge;

  PetscFunctionBegin; 
  /* Iterate through the marked slow edges */
  ierr = ISGetIndices(fvnet->slow_edges,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->slow_edges,&size);CHKERRQ(ierr);
  /* Find the correct sizes for the arrays */
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr); 
    if (!fvedge->tobufferlvl) {
      slow_size += dof*bufferwidth; 
    } else {
      buf_size  += dof*bufferwidth; 
    }
    if (!fvedge->frombufferlvl) {
      slow_size += dof*bufferwidth; 
    } else {
      buf_size  += dof*bufferwidth; 
    }
    slow_size   += dof*fvedge->nnodes - 2*dof*bufferwidth; 
  }
  ierr = PetscMalloc1(slow_size,&i_slow);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size,&i_buf);CHKERRQ(ierr);
  /* Build the set of indices (global data indices) */
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableGlobalOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    if (!fvedge->frombufferlvl) {
      for (j=0; j<bufferwidth; j++) {
        for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        }
      }
    } else {
      for (j=0; j<bufferwidth; j++) {
        for (k=0; k<dof; k++) {
          i_buf[buf_count] = offset+j*dof+k; 
          buf_count++;
        }
      }
    }
    for (j=bufferwidth; j<(fvedge->nnodes-bufferwidth); j++){
      for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        }
    }
    if(!fvedge->tobufferlvl) {
      for (j=fvedge->nnodes-bufferwidth; j<fvedge->nnodes; j++) {
        for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        } 
      }
    } else {
      for (j=fvedge->nnodes-bufferwidth; j<fvedge->nnodes; j++) {
        for (k=0; k<dof; k++) {
          i_buf[buf_count] = offset+j*dof+k; 
          buf_count++;
        }
      } 
    }
  }
  ierr = ISRestoreIndices(fvnet->slow_edges,&index);CHKERRQ(ierr);
  /* Repeat the same procedure for the fast edges. However a fast edge has no buffer data on it so this simplifies a bit. */
  /* Iterate through the marked fast edges */
  ierr = ISGetIndices(fvnet->fast_edges,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->fast_edges,&size);CHKERRQ(ierr);
  /* Find the correct sizes for the arrays */
  for (i=0; i<size; i++) {
    e         = index[i];
    ierr      = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr); 
    fast_size += dof*fvedge->nnodes;
  }
  ierr = PetscMalloc1(fast_size,&i_fast);CHKERRQ(ierr); 
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableGlobalOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    for (j=0; j<fvedge->nnodes; j++) {
      for (k=0; k<dof; k++) {
        i_fast[fast_count] = offset+j*dof+k; 
        fast_count++; 
      }
    }
  }
  ierr = ISRestoreIndices(fvnet->fast_edges,&index);CHKERRQ(ierr);
  /* Now Build the index sets */
  ierr = ISCreateGeneral(fvnet->comm,slow_size,i_slow,PETSC_COPY_VALUES,slow);CHKERRQ(ierr); 
  ierr = ISCreateGeneral(fvnet->comm,fast_size,i_fast,PETSC_COPY_VALUES,fast);CHKERRQ(ierr); 
  ierr = ISCreateGeneral(fvnet->comm,buf_size,i_buf,PETSC_COPY_VALUES,buffer);CHKERRQ(ierr); 
  /* Free Data */
  ierr = PetscFree(i_slow);CHKERRQ(ierr);
  ierr = PetscFree(i_fast);CHKERRQ(ierr);
  ierr = PetscFree(i_buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}