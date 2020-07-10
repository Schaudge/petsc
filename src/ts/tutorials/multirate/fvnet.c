#include "fvnet.h"

PetscErrorCode FVNetworkCreate(PetscInt initial,FVNetwork fvnet, PetscInt Mx)
{
  PetscErrorCode ierr;
  PetscInt       nfvedge;
  PetscMPIInt    rank;
  PetscInt       i,numVertices,numEdges;
  PetscInt       *edgelist;
  Junction       junctions=NULL;
  FVEdge         fvedges=NULL;
  PetscInt       dof=fvnet->physics.dof; 
  
  PetscFunctionBegin;
  fvnet->nnodes_loc = 0;
  ierr = MPI_Comm_rank(fvnet->comm,&rank);CHKERRQ(ierr);
  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;
  fvnet->initial = initial; 

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
    nfvedge = 3;
    fvnet->nedge   = nfvedge;
    fvnet->nvertex = nfvedge + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = fvnet->nvertex;
      numEdges    = fvnet->nedge;

      ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);
      for (i=0; i<numEdges; i++) {
        edgelist[2*i] = i; edgelist[2*i+1] = i+1;
      }

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);

      /* vertex */
      junctions[0].type = OUTFLOW;
      junctions[1].type = JUNCT;
      junctions[2].type = JUNCT;
      junctions[3].type = OUTFLOW;

      for(i=0; i<numVertices; i++){
        junctions[i].x = i*1.0/3.0; 
      }

      /* Edge */ 
      fvedges[0].nnodes = Mx; 
      fvedges[1].nnodes = fvnet->hratio*Mx; 
      fvedges[2].nnodes = Mx; 

      for(i=0; i<numEdges;i++){
          fvedges[i].h = 1.0/3.0/(PetscReal)fvedges[i].nnodes; 
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
  Junction          junctions; 
  PetscInt          KeyEdge,KeyJunction,KeyFlux;
  PetscInt          i,e,v,eStart,eEnd,vStart,vEnd,dof = fvnet->physics.dof,vfrom,vto;
  PetscInt          nnodes_edge,nnodes_vertex,nedges_tmp; 
  PetscInt          *edgelist = NULL,*edgelists[1];
  DM                networkdm = fvnet->network;
  PetscInt          nedges,nvertices; /* local num of edges and vertices */
  FVEdge            fvedge;
  Junction          junction;
  MPI_Comm          comm = fvnet->comm;
  PetscMPIInt       size,rank;
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
  ierr = DMNetworkSetSizes(networkdm,1,&nvertices,&nedges,0,NULL);CHKERRQ(ierr);

  /* Add local edge connectivity */
  edgelists[0] = edgelist;
  ierr = DMNetworkSetEdgeList(networkdm,edgelists,NULL);CHKERRQ(ierr);
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

  ierr = DMNetworkRegisterComponent(networkdm,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"fvedgestruct",sizeof(struct _p_FVEdge),&KeyEdge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"empty",0,&KeyFlux);CHKERRQ(ierr);

  /* Add FVEdge component to all local edges. Note that as we have 
     yet to distribute the network, all data is on proc[0]. */
  for (e = eStart; e < eEnd; e++) {
    ierr = DMNetworkAddComponent(networkdm,e,KeyEdge,&fvedge[e-eStart]);CHKERRQ(ierr);
    /* Add number of variables to each edge. An edge does not own all 
       of the data belonging to its discretization. The connected vertices 
       each own the nearest bufferwidth+stencilwidth cells. */
    nnodes_edge = dof*(fvedge[e-eStart].nnodes - 2*(fvnet->bufferwidth+fvnet->stencilwidth)); 
    ierr = DMNetworkAddNumVariables(networkdm,e,dof*fvedge[e-eStart].nnodes);CHKERRQ(ierr);

/* Monitoring stuff, to be re-added once I understand how it works. 
    if (size == 1 && monifvedge) { 
      fvedge[e-eStart].length = 600.0;
      ierr = DMNetworkMonitorAdd(monitor, "Pipe Q", e, fvedge[e-eStart].nnodes, 0, 2, 0.0,fvedge[e-eStart].length, -0.8, 0.8, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMNetworkMonitorAdd(monitor, "Pipe H", e, fvedge[e-eStart].nnodes, 1, 2, 0.0,fvedge[e-eStart].length, -400.0, 800.0, PETSC_TRUE);CHKERRQ(ierr);
    }
*/    
  }
  /* Add Junction component to all local vertices, including ghost vertices! However
     all data is currently assumed to be on proc[0]. Also add the flux component */
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkAddComponent(networkdm,v,KeyJunction,&junctions[v-vStart]);CHKERRQ(ierr);
    /* Add number of variables to vertex. Each vertex owns bufferwidth+stencilwidth cells 
       from each connected edge.*/
    ierr = DMNetworkGetSupportingEdges(networkdm,v,&nedges_tmp,&edges);CHKERRQ(ierr); /* will NULL work? --it does not. */
    nnodes_vertex = dof*(fvnet->bufferwidth+fvnet->stencilwidth)*nedges_tmp;
    ierr = DMNetworkSetComponentNumVariables(networkdm,v,JUNCTION,nnodes_vertex);CHKERRQ(ierr);
    /* Add data structure for moving the vertex fluxes around. Also used to store 
       vertex reconstruction information. A working vector used for data storage 
       and message passing.*/
    ierr = DMNetworkAddComponent(networkdm,v,KeyFlux,NULL);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(networkdm,v,FLUX,dof*nedges_tmp);CHKERRQ(ierr);
  }
  DMSetUp(fvnet->network);
  /* Build the edge offset data to allow for a sensible local ordering of the 
     edges of a vertex. Needed so that the data belonging to a vertex knows
     which edge each piece should interact with.*/
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges_tmp,&edges);CHKERRQ(ierr);
    junction->numedges = nedges_tmp;
    /* Iterate through the connected edges. As we are on a single processor, DMNetworkGetSupportingEdges which returns 
       on processor edges, will be returning ALL connected edges on the graph.  */
    for(i=0; i<nedges_tmp; i++){
      e = edges[i];   
      ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void **)&fvedge);CHKERRQ(ierr);
      ierr = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];
      if(v==vto){
        fvedge->vto_offset = dof*i*(fvnet->stencilwidth+fvnet->bufferwidth); 
        fvedge->vto_recon_offset = dof*i; 
      } else if(v==vfrom){
        fvedge->vfrom_offset = dof*i*(fvnet->stencilwidth+fvnet->bufferwidth); 
        fvedge->vfrom_recon_offset = dof*i; 
      } else {
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"v %D != vfrom or vto from supporting edge %D",v,e);
      }
    }
  }
  /* we are still on a single processor*/
  PetscFunctionReturn(0);
}

/* Now we have a distributed network. It is assumed that localX and X have been created in fvnet */
PetscErrorCode FVNetworkSetupPhysics(FVNetwork fvnet){
  PetscErrorCode ierr; 
  PetscInt       e,v,i,nedges,dof = fvnet->physics.dof;
  PetscInt       eStart,eEnd,vStart,vEnd;
  PetscInt       offset;  
  PetscInt       vfrom,vto; 
  const PetscInt *cone,*edges; 
  FVEdge         fvedge; 
  Junction       junction;
  Vec            localX; 
  PetscScalar    *xarr; 

  PetscFunctionBegin;
  localX = fvnet->localX; /* Local component of the dmnetwork vector */
  ierr = VecSet(fvnet->X,0.0);CHKERRQ(ierr);
  ierr = VecSet(localX,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);

  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  /* Build the data so that vertex knows what edges point into it, and which edges point out.
     We temporarily use the flux component to set up this structure. At the end it will be locally 
     stored, but we have to do a message-passing start up to get all of the right 
     information onto the local processors. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,FLUX,v,&offset);CHKERRQ(ierr);
    /* Iterate through the (local) connected edges. Each ghost vertex of a vertex connects to a 
       a non-overlapping set of local edges. This is why we can iterate in this way without 
       potentially conflicting our scatters.*/
    for(i=0; i<nedges; i++) { 
      e = edges[i];  
      ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void **)&fvedge);CHKERRQ(ierr);
      ierr = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1]; 
      if(v==vto) {
        xarr[offset+fvedge->vto_recon_offset] = EDGEIN;
      } else if(v==vfrom) {
        xarr[offset+fvedge->vfrom_recon_offset] = EDGEOUT;
      } else {
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D != vfrom or vto from supporting edge %D",v,e);
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr); 
  ierr = DMLocalToGlobalBegin(fvnet->network,localX,ADD_VALUES,fvnet->X);CHKERRQ(ierr);  
  ierr = DMLocalToGlobalEnd(fvnet->network,localX,ADD_VALUES,fvnet->X);CHKERRQ(ierr);
  /* Now the flux components hold the edgein/edgeout information for all edges connected to the vertex (not just the local edges)
     Rebuild the local ghost vector data.*/
  ierr = DMGlobalToLocalBegin(fvnet->network,fvnet->X,INSERT_VALUES,localX);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,fvnet->X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  /* Iterate through all the vertices (including ghosts? Not sure if they actually need this info, but I guess why not)
     and build the junction component data structure dir. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,FLUX,v,&offset);CHKERRQ(ierr);
    ierr = PetscMalloc1(junction->numedges,&(junction->dir));CHKERRQ(ierr); /* Freed in the final cleanup call*/
    /* Fill in the local dir data*/
    for(i=0; i<junction->numedges; i++) { 
        junction->dir[i] = xarr[offset+i*dof];
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}

PetscErrorCode FVNetworkCleanUp(FVNetwork fvnet){
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

PetscErrorCode FVNetworkDestroy(FVNetwork fvnet) {
  PetscErrorCode ierr;
  PetscInt       i,v,vStart,vEnd;
  Junction       junction;

  PetscFunctionBegin; 
  /* Still need to destroy my dynamic componenets ... */
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr);
    ierr = PetscFree(junction->dir);CHKERRQ(ierr); /* Free dynamic memory for the junction component */
  }
  ierr = (*fvnet->physics.destroy)(fvnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<fvnet->physics.dof; i++) {ierr = PetscFree(fvnet->physics.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree4(fvnet->R,fvnet->Rinv,fvnet->cjmpLR,fvnet->cslope);CHKERRQ(ierr);
  ierr = PetscFree4(fvnet->uLR,fvnet->flux,fvnet->speeds,fvnet->uPlus);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->X);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->localX);CHKERRQ(ierr);
  ierr = VecDestroy(&fvnet->localF);CHKERRQ(ierr);  
  PetscFunctionReturn(0); 
}
PetscErrorCode FVNetCharacteristicLimit(FVNetwork fvnet,PetscScalar *uL,PetscScalar *uM,PetscScalar *uR){
    PetscErrorCode ierr; 
    PetscScalar    jmpL,jmpR,*cjmpL,*cjmpR,*uLtmp,*uRtmp,tmp;
    PetscInt       dof = fvnet->physics.dof;
    PetscInt       j,k;

    PetscFunctionBegin;
    /* Create characteristic jumps */
    ierr = (*fvnet->physics.characteristic)(fvnet->physics.user,dof,uM,fvnet->R,fvnet->Rinv,fvnet->speeds);CHKERRQ(ierr);
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
  FVNetwork      fvnet = (FVNetwork)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,dof = fvnet->physics.dof;
  PetscReal      h,cfl_idt = 0;
  PetscScalar    *f,*uL,*uR;
  PetscInt       v,e,vStart,vEnd,eStart,eEnd,vfrom,vto,vto_offset,vfrom_offset;
  PetscInt       offsetf,offset,dataoffset,offsetv,vwidth;
  const PetscInt *cone;
  Vec            localX = fvnet->localX,localF = fvnet->localF; 
  PetscScalar    *xarr;
  FVEdge         fvedge; 
  Junction       junction;
  PetscReal      maxspeed; 

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,fvnet->X,INSERT_VALUES,localX);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,fvnet->X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr); /* Why do I need to do these things through a localF? going to be real annoying with multirate */
  vwidth = (fvnet->stencilwidth+fvnet->bufferwidth);
  /* Iterate through all vertices (including ghosts) and compute the flux data for the vertex.  */
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);
  for (v=vStart; v<vEnd; v++){
    /* Reconstruct all data points and then compute vertex flux data (either coupling flux or boundary conditions) */
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,FLUX,v,&offsetf);
    ierr = DMNetworkGetComponentVariableOffset(fvnet->network,JUNCTION,v,&offset);
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction); 
    switch(junction->type){
        case JUNCT:
            for (i=0; i<junction->numedges; i++){
            /* Hard coded 2 cell one-side reconstruction. To be improved */
            dataoffset = dof*i*(fvnet->stencilwidth+fvnet->bufferwidth); 
                for(j=0; j<dof; j++){
                    xarr[offsetf+i*dof+j] = 0.5*(3*xarr[offset+dataoffset+j] - xarr[offset+dataoffset+dof+j]); /* This assumes a stencil width of at least 2 */
                }
            }
            /* Now compute the coupling flux */
            if (junction->numedges > 2){
                SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has more than 2 connected edges. Coupling flux supports only up to 2 edges currently",v);
            } else { /* Note that in creation a junction must have at least 2 connected edges */
                    if (junction->dir[0] == EDGEIN){
                        uL = xarr+offsetf;
                        uR = xarr+offsetf+dof;
                    } else {
                        uL = xarr+offsetf+dof;
                        uR = xarr+offsetf;
                    }
                fvnet->couplingflux(fvnet->physics.user,dof,uL,uR,fvnet->flux,&maxspeed);
                for(i=0; i<junction->numedges; i++){
                    for(j=0; j<dof; j++){
                        xarr[offsetf+i*dof+j] = fvnet->flux[j];
                    }
                }
            }
            break;
        case OUTFLOW: 
            if (junction->numedges != 1) {
                SETERRQ1(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"number of edge of vertex %D != 1 and is a OUTFLOW vertex. OUTFLOW vertices require exactly one edge",v);
            } else {
                /* Ghost cell Reconstruction Technique: Repeat the last cell. */
                ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr); /* Assumes stencil width of at least 2 */
                /* &fvnet->uLR[0] gives the reconstructed value on the right of the interface. The left interface reconstructed value 
                   is just the cell average to the right of the boundary. */
                ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,&xarr[offset],fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
                /* place flux in the flux component */
                for(j=0; j<dof; j++) {
                        xarr[offsetf+j] = fvnet->flux[j];
                }
            }
            break;
        default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
    } 
  }
  /* Now all the vertex flux data is available. */
  /* Iterate through the edges and update the cell data 'belonging' to that edge */
  /* WE ASSUME THAT localF HAS THE SAME STRUCTURE AS localX! FOR THIS UPDATE */
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr); 
  for(e=eStart; e<eEnd; e++){
        /* The cells are updated in the order 1) from vertex data 2) edge data 3) to vertex data */
        ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
        ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
        vfrom = cone[0];
        vto   = cone[1];
        ierr  = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
        ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,JUNCTION,vfrom,&offsetv);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,FLUX,vfrom,&offsetf);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,FVEDGE,e,&offset);CHKERRQ(ierr);
        vfrom_offset = fvedge->vfrom_offset;
        h = fvedge->h;
        /* Update the vfrom vertex flux for this edge */
        for(j=0; j<dof; j++) {
            f[offsetv+vfrom_offset+j] += xarr[fvedge->vfrom_recon_offset+j]/h;
        }
        /* Now reconstruct the value at the left cell of the 1/2 interface. I have to redo code here, should alter 
         how I compute things to avoid this.  */
        switch(junction->type) {
            case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
                for(j=0; j<dof; j++){
                    fvnet->uPlus[j] = 0.5*(3*xarr[offsetv+vfrom_offset+j] + xarr[offsetv+vfrom_offset+dof+j]); /* This assumes a stencil width of at least 2 */
                }
                break;
            case OUTFLOW: 
                /* Ghost cell Reconstruction Technique: Repeat the last cell. */
                ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vfrom_offset],&xarr[offsetv],&xarr[offsetv+vfrom_offset+dof]);CHKERRQ(ierr); /* Assumes stencil width of at least 2 */
                uR = fvnet->uLR+dof; 
                for(j=0; j<dof; j++){
                    fvnet->uPlus[j] = uR[j]; 
                }
                break;
            default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        /* Iterate through the cells owned by the vfrom vertex */
        for(i=1; i<vwidth-2; i++) {
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vfrom_offset+(i-1)*dof],&xarr[offsetv+vfrom_offset+i*dof],&xarr[offsetv+vfrom_offset+(i+1)*dof]);CHKERRQ(ierr);
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for(j=0; j<dof; j++){
              fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
            }
            cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
            for(j=0; j<dof; j++) {
                f[offsetv+vfrom_offset+(i-1)*dof+j] -= fvnet->flux[j]/h; 
                f[offsetv+vfrom_offset+i*dof+j] += fvnet->flux[j]/h; 
            }
        }
        /* Different update for the interface one index to the left the boundary of the vertex data and the cell data. 
        Require fvedge data for reconstruction */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vfrom_offset+(vwidth-2)*dof],&xarr[offsetv+vfrom_offset+(vwidth-1)*dof],&xarr[offset]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
         for(j=0; j<dof; j++){
              fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
            f[offsetv+vfrom_offset+(vwidth-2)*dof+j] -= fvnet->flux[j]/h; 
            f[offsetv+vfrom_offset+(vwidth-1)*dof+j] += fvnet->flux[j]/h; 
        }
        /* Now Compute Flux on the interface between vertex data and cell data */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vfrom_offset+(vwidth-1)*dof],&xarr[offset],&xarr[offset+dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for(j=0; j<dof; j++){
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
            f[offsetv+vfrom_offset+(vwidth-1)*dof+j] -= fvnet->flux[j]/h; 
            f[offset+j]                              += fvnet->flux[j]/h; 
        }
        /* Iterate through the interior interfaces of the fvedge */
        for(i=1; i<(fvedge->nnodes - 2); i++){
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(i-1)*dof],&xarr[offset+i*dof],&xarr[offset+(i+1)*dof]);CHKERRQ(ierr);
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for(j=0; j<dof; j++){
              fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
            }
            cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
            for(j=0; j<dof; j++) {
                f[offset+(i-1)*dof] -= fvnet->flux[j]/h;
                f[offset+i*dof]     += fvnet->flux[j]/h;
            }
        }
        ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,JUNCTION,vto,&offsetv);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponentVariableOffset(fvnet->network,FLUX,vto,&offsetf);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction);CHKERRQ(ierr); 
        vto_offset = fvedge->vto_offset;
        /* Compute the interface flux one index to the left of the vto vertex boundary */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(fvedge->nnodes-2)*dof],&xarr[offset+(fvedge->nnodes-2)*dof],&xarr[offsetv+vto_offset+(vwidth-1)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for(j=0; j<dof; j++){
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
            f[offset+(fvedge->nnodes-2)*dof] -= fvnet->flux[j]/h; 
            f[offset+(fvedge->nnodes-1)*dof] += fvnet->flux[j]/h;
        }
        /* Compute the flux on the interface between fvedge and vertex vto */
        ierr = FVNetCharacteristicLimit(fvnet,&xarr[offset+(fvedge->nnodes-1)*dof],&xarr[offsetv+vto_offset+(vwidth-1)*dof],&xarr[offsetv+vto_offset+(vwidth-2)*dof]);CHKERRQ(ierr);
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        for(j=0; j<dof; j++){
          fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
        }
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
            f[offset+(fvedge->nnodes-1)*dof]     -= fvnet->flux[j]/h; 
            f[offsetv+vto_offset+(vwidth-1)*dof] += fvnet->flux[j]/h;
        }
        /* Iterate through the vto vertex data */
        for(i = vwidth-2; i>0; i--) {
            ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vto_offset+(i+1)*dof],&xarr[offsetv+vto_offset+i*dof],&xarr[offsetv+vto_offset+(i-1)*dof]);CHKERRQ(ierr);
            ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
            for(j=0; j<dof; j++){
              fvnet->uPlus[j] = fvnet->uLR[dof+j]; 
            }
            cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
            for(j=0; j<dof; j++) {
            f[offsetv+vto_offset+(i+1)*dof] -= fvnet->flux[j]/h; 
            f[offsetv+vto_offset+i*dof]     += fvnet->flux[j]/h;
            }
        }
        /* Now reconstruct the value at the 2nd to last interface . I have to redo code here, should alter 
           how I compute things to avoid this.  */
        switch(junction->type) {
            case JUNCT:
            /* Hard coded 2 cell one-side reconstruction. To be improved */
                for(j=0; j<dof; j++){
                    fvnet->uLR[j] = 0.5*(3*xarr[offsetv+vto_offset+j] + xarr[offsetv+vto_offset+dof+j]); /* This assumes a stencil width of at least 2 */
                }
                break;
            case OUTFLOW: 
                /* Ghost cell Reconstruction Technique: Repeat the last cell. */
                ierr = FVNetCharacteristicLimit(fvnet,&xarr[offsetv+vto_offset+dof],&xarr[offsetv],&xarr[offsetv]);CHKERRQ(ierr); /* Assumes stencil width of at least 2 */
                break;
            default:
            SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"vertex %D has unsupported type %D",v,junction->type);
        } 
        ierr = (*fvnet->physics.riemann)(fvnet->physics.user,dof,fvnet->uPlus,fvnet->uLR,fvnet->flux,&maxspeed);CHKERRQ(ierr);
        cfl_idt = PetscMax(PetscAbsScalar(maxspeed/h),PetscAbsScalar(cfl_idt)); /* Update the maximum signal speed (for CFL) */
        for(j=0; j<dof; j++) {
            f[offsetv+vto_offset+dof] -= fvnet->flux[j]/h; 
            f[offsetv+vto_offset]     += fvnet->flux[j]/h;
        }
        /* Update the vfrom vertex flux for this edge */
        for(j=0; j<dof; j++) {
            f[offsetv+vto_offset+j] -= xarr[fvedge->vto_recon_offset+j]/h;
        }
        /* We have now updated the rhs for all data for this edge. */
    }
    /* Data Cleanup */
    ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
    ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,F);CHKERRQ(ierr);

    ierr = MPI_Allreduce(&cfl_idt,&fvnet->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,fvnet->comm);CHKERRQ(ierr);
    if (0) {
        /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
        PetscReal dt,tnow;
        ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
        ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
        if (dt > 0.5/fvnet->cfl_idt) {
            if (1) {
                ierr = PetscPrintf(fvnet->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/fvnet->cfl_idt));CHKERRQ(ierr);
            } else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Stability constraint exceeded, %g > %g",(double)dt,(double)(fvnet->cfl/fvnet->cfl_idt));
        }
    }
  PetscFunctionReturn(0);
}