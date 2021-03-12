#include "fvnet.h"

PetscErrorCode FVNetworkCreate(FVNetwork fvnet,PetscInt networktype,PetscInt Mx)
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
  ierr = SNESCreate(MPI_COMM_SELF,&fvnet->snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fvnet->snes);CHKERRQ(ierr);
  ierr = KSPCreate(MPI_COMM_SELF,&fvnet->ksp);CHKERRQ(ierr); 
  ierr = KSPSetFromOptions(fvnet->ksp);CHKERRQ(ierr);
  fvnet->nnodes_loc  = 0;
  ierr               = MPI_Comm_rank(fvnet->comm,&rank);CHKERRQ(ierr);
  numVertices        = 0;
  numEdges           = 0;
  edgelist           = NULL;
  fvnet->networktype = networktype; 
  /* proc[0] creates a sequential fvnet and edgelist */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Setup Initial Network %D\n",networktype);CHKERRQ(ierr);
  /* Set global number of fvedges, edges, and junctions */
  /*-------------------------------------------------*/
  switch (networktype) {
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

        edgelist[0] = 0;
        edgelist[1] = 1;
        edgelist[2] = 1;
        edgelist[3] = 2;
        edgelist[4] = 2;
        edgelist[5] = 3; 
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
    case 2:
      /* Case 2: */
      /* =================================================
      (OUTFLOW) v0 <--E0-- v1<--E1-- v2 <--E2 --v3 (OUTFLOW)
      ====================================================  
      This tests whether the coupling flux can handle the "non-standard"
      directed graph formulation of the problem. This is the same problem as 
      case 0, but changes the direction of the graph and accordingly how the discretization 
      works. The geometry of the vertices is adjusted to compensate. */
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

        edgelist[0] = 1;
        edgelist[1] = 0;
        edgelist[2] = 2;
        edgelist[3] = 1;
        edgelist[4] = 3;
        edgelist[5] = 2; 
        /* Add network components */
        /*------------------------*/
        ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
        /* vertex */
        junctions[0].type = OUTFLOW;
        junctions[1].type = JUNCT;
        junctions[2].type = JUNCT;
        junctions[3].type = OUTFLOW;

        for (i=0; i<numVertices; i++) {
          junctions[i].x = (3-i)*1.0/3.0*50.0; 
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
    case 3:
    /* Case 3: (Image is for the case we ndaughers = 2. The number of out branches is given by fvnet->ndaughers */
    /* =================================================
    (OUTFLOW) v0 --E0--> v1--E1--> v2  (OUTFLOW)
                          |
                          E2  
                          |
                          \/
                          v3 (OUTFLOW) 
    ====================================================  
    This tests the coupling condition for the simple case */
    nfvedge        = fvnet->ndaughters+1; 
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

      /* Parent Branch (pointing in) */
      edgelist[0] = 0;
      edgelist[1] = 1;
      /* Daughter Branches (pointing out from v1) */
      for (i=1; i<fvnet->ndaughters+1; ++i) {
        edgelist[2*i]   = 1; 
        edgelist[2*i+1] = i+1; 
      }

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
      /* vertex */
      junctions[0].type = OUTFLOW;
      junctions[1].type = JUNCT;
      for (i=2; i<fvnet->ndaughters+2; ++i) {
        junctions[i].type = OUTFLOW;
        junctions[i].x    = 3.0;
      }

      junctions[0].x = -3.0; 
      junctions[1].x = 0.0; 
      /* Edge */ 
      fvedges[0].nnodes = fvnet->hratio*Mx; 
      for(i=1; i<fvnet->ndaughters+1; ++i) {
        fvedges[i].nnodes = Mx;  
      }

      for (i=0; i<numEdges;i++) {
        fvedges[i].h = 3.0/(PetscReal)fvedges[i].nnodes; 
      }
    }
    break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }
  /* set edge global id */
  for (i=0; i<numEdges; i++) fvedges[i].id = i;
  /* set junction global id and set default values */
  for (i=0; i<numVertices; i++) 
  {
    junctions[i].id = i;
  }
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
  PetscInt          *edgelist = NULL;
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
  ierr = DMNetworkSetNumSubNetworks(fvnet->network,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = DMNetworkAddSubnetwork(fvnet->network,NULL,nvertices,nedges,edgelist,NULL);CHKERRQ(ierr);

  ierr = DMNetworkLayoutSetUp(fvnet->network);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"fvedgestruct",sizeof(struct _p_FVEdge),&KeyEdge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"flux",0,&KeyFlux);CHKERRQ(ierr);
  /* Add FVEdge component to all local edges. Note that as we have 
     yet to distribute the network, all data is on proc[0]. */
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkAddComponent(fvnet->network,e,KeyEdge,&fvedge[e-eStart],dof*fvedge[e-eStart].nnodes);CHKERRQ(ierr);
    /* Add a monitor for every edge in the network, label the data according the user provided physics */
<<<<<<< HEAD
<<<<<<< HEAD
    if (size == 1 && fvnet->viewfv) { 
=======
    if (size == 1 && fvnet->monifv) { 
>>>>>>> Fixed memory leak. Reorganized the file structure.
=======
    if (size == 1 && fvnet->viewfv) { 
>>>>>>> Added preliminary test to ex9. Small modifications to ex9
      length = fvedge[e-eStart].h*(fvedge[e-eStart].nnodes+1);
      for (j=0; j<dof; j++) {
         ierr = DMNetworkMonitorAdd(fvnet->monitor,fvnet->physics.fieldname[j],e,fvedge[e-eStart].nnodes,j,dof,0.0,length,fvnet->ymin,fvnet->ymax,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  }
  /* Add Junction component to all local vertices. All data is currently assumed to be on proc[0]. Also add the flux component */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkAddComponent(fvnet->network,v,KeyJunction,&junctions[v-vStart],0);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges_tmp,&edges);CHKERRQ(ierr);
    /* Add data structure primarily for moving the vertex fluxes around. Is used throughout 
       passing various data between processors. */
    ierr = DMNetworkAddComponent(fvnet->network,v,KeyFlux,NULL,dof*nedges_tmp);CHKERRQ(ierr);
  }
  ierr = DMSetUp(fvnet->network);CHKERRQ(ierr);
  /* Build the edge offset data to allow for a sensible local ordering of the 
     edges of a vertex. Needed so that the data belonging to a vertex knows
     which edge each piece should interact with. */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges_tmp,&edges);CHKERRQ(ierr);
    junction->numedges = nedges_tmp;
    /* Iterate through the connected edges. As we are on a single processor, DMNetworkGetSupportingEdges which returns 
       on processor edges, will be returning ALL connected edges on the graph. */
    for (i=0; i<nedges_tmp; i++) {
      e     = edges[i];   
      ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
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
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
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
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&nedges,&edges);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offset);CHKERRQ(ierr);
    /* Iterate through the (local) connected edges. Each ghost vertex of a vertex connects to a 
       a non-overlapping set of local edges. This is why we can iterate in this way without 
       potentially conflicting our scatters.*/
    for (i=0; i<nedges; i++) { 
      e     = edges[i];  
      ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void **)&fvedge,NULL);CHKERRQ(ierr);
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
  /* Iterate through all vertices and build the junction component data structure dir and local 
     work array flux */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offset);CHKERRQ(ierr);
    ierr = PetscMalloc1(junction->numedges,&(junction->dir));CHKERRQ(ierr); /* Freed in the network destroy call */
    ierr = PetscMalloc1(dof*junction->numedges,&(junction->flux));CHKERRQ(ierr); /* Freed in the network destroy call */
    /* Fill in the local dir data */
    for (i=0; i<junction->numedges; i++) { 
      junction->dir[i] = xarr[offset+i*dof];
    }
  }
  ierr = FVNetworkAssignCoupling(fvnet);CHKERRQ(ierr);

  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}
 /* Iterate through the vertices and assign the coupling flux functions
     This is done by a user provided function that maps the junction type (an integer) to 
     a user specified VertexFlux. A VertexFlux must be provided for all non-boundary types, that 
     is JUNCT junctions and any other user specified coupling junction types. */
PetscErrorCode FVNetworkAssignCoupling(FVNetwork fvnet)
{
  PetscErrorCode ierr;
  PetscInt       v,vStart,vEnd;
  Junction       junction; 

  PetscFunctionBegin; 

  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr); 
    
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = fvnet->physics.vfluxdestroy(fvnet,junction);CHKERRQ(ierr);
    ierr = fvnet->physics.vfluxassign(fvnet,junction);CHKERRQ(ierr);
  }
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
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    /* Free dynamic memory for the junction component */
    ierr = PetscFree(junction->dir);CHKERRQ(ierr); 
    ierr = PetscFree(junction->flux);CHKERRQ(ierr);
    ierr = VecDestroy(&junction->rcouple);CHKERRQ(ierr);
    ierr = VecDestroy(&junction->xcouple);CHKERRQ(ierr);
    ierr = MatDestroy(&junction->couplesystem);CHKERRQ(ierr);
    ierr = MatDestroy(&junction->jacobian);CHKERRQ(ierr);
  }
  ierr = (*fvnet->physics.destroy)(fvnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<fvnet->physics.dof; i++) {
    ierr = PetscFree(fvnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(fvnet->R,fvnet->Rinv,fvnet->cjmpLR,fvnet->cslope);CHKERRQ(ierr);
  ierr = PetscFree4(fvnet->uLR,fvnet->flux,fvnet->speeds,fvnet->uPlus);CHKERRQ(ierr);
  ierr = SNESDestroy(&fvnet->snes);CHKERRQ(ierr);
  ierr = KSPDestroy(&fvnet->ksp);CHKERRQ(ierr); 
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

PetscErrorCode FVNetworkSetInitial(FVNetwork fvnet,Vec X0) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,vfrom,vto,type,offset,e,eStart,eEnd,dof = fvnet->physics.dof;
  const PetscInt *cone;
  PetscScalar    *xarr,*u,*utmp;
  Junction       junction;
  FVEdge         fvedge;
  Vec            localX = fvnet->localX;
  PetscReal      h,xfrom,xto,x,xend,xstart;
  
  PetscFunctionBegin;
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,&type,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h     = fvedge->h;
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetComponent(fvnet->network,vto,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    xto   = junction->x;
    ierr  = DMNetworkGetComponent(fvnet->network,vfrom,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    xfrom = junction->x;
    if (fvnet->networktype<3) {
      /* This code assumes a geometrically 1d network. To be improved later */
      for (i=0; i<fvedge->nnodes; i++) {
        if (xto>xfrom) {
          x = xfrom+i*h;
        } else {
          x = xfrom-i*h;
        }
        u = xarr+offset+i*dof; 
        switch (fvnet->networktype) {
          case 0: 
          case 1:
          case 2:
            /*Both are networks on [0,1] and so use the same initial conditions. User provided geometrically 1d initial conditions */
            fvnet->physics.sample1d((void*)&fvnet->physics.user,fvnet->initial,0.0,x,u);
            break;
          default: 
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
        }
      }
    } else { /* Our sample function changes for each edge in the network */
      ierr = PetscMalloc1(2*dof,&utmp);CHKERRQ(ierr);
      for (i=0; i<fvedge->nnodes; i++) {
          if (xto>xfrom) {
          xstart = xfrom+i*h;
          xend   = xstart+h;
        } else {
          xstart = xfrom-i*h;
          xend   = xstart-h;
        }
        u = xarr+offset+i*dof;
        fvnet->physics.samplenetwork((void*)&fvnet->physics.user,fvnet->initial,0.0,xstart,utmp,fvedge->id);
        fvnet->physics.samplenetwork((void*)&fvnet->physics.user,fvnet->initial,0.0,xend,utmp+dof,fvedge->id);
        for(j=0;j<dof;j++) {
          u[j] = (utmp[j]+utmp[j+dof])/2.0; /* Trapezoid integration */
        }
      }
      ierr = PetscFree(utmp);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  /* Can use insert as each edge belongs to a single processor and vertex data is only for temporary computation and holds no 'real' data. */
  ierr = DMLocalToGlobalBegin(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


/* Compute the L1 Norm of the Vector X associated with the FVNetowork fvnet */
PetscErrorCode FVNetworkL1CellAvg(FVNetwork fvnet, Vec X,PetscReal *norm) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,type,offset,e,eStart,eEnd,dof = fvnet->physics.dof;
  const PetscScalar    *xarr,*u;
  FVEdge         fvedge;
  Vec            localX = fvnet->localX;
  PetscReal      h;
  
  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for (j=0;j<dof;j++) {
    norm[j] = 0.0; 
  }
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,&type,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    h     = fvedge->h;
    for (i=0; i<fvedge->nnodes; i++) {
      u = xarr+offset+i*dof;
      for(j=0; j<dof; j++) {
        norm[j] += h*PetscAbs(u[j]);
      }
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  MPI_Allreduce(&norm,&norm,dof,MPIU_REAL,MPIU_SUM,fvnet->comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}