#include "dgnet.h"

PetscErrorCode DGNetworkCreate(DGNetwork fvnet,PetscInt networktype,PetscInt Mx)
{
  PetscErrorCode ierr;
  PetscInt       nfvedge;
  PetscMPIInt    rank;
  PetscInt       i,numVertices,numEdges;
  PetscInt       *edgelist;
  Junction       junctions = NULL;
  EdgeFE         fvedges = NULL;
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
  /* proc[0] creates a sequential fvnet and edgelist */
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

      for (i=2; i<fvnet->ndaughters+2; ++i) {
        junctions[i].x    = fvnet->length;
      }

      junctions[0].x = -fvnet->length; 
      junctions[1].x = 0.0; 
      /* Edge */ 
      fvedges[0].nnodes = fvnet->hratio*Mx; 
      for(i=1; i<fvnet->ndaughters+1; ++i) {
        fvedges[i].nnodes = Mx;  
      }

      for (i=0; i<numEdges;i++) {
        fvedges[i].h = fvnet->length/(PetscReal)fvedges[i].nnodes; 
      }
    }
    break;
  case 4:
    /* Case 4: ndaughter-1-ndaughter 
    =================================================
    (OUTFLOW) v2 --E1--> v0--E0--> v1 --E3--> (OUTFLOW)
                          ^         ^
                          |         |
                          E1        E4
                          |         |
                (OUTFLOW) v3        v4 (OUTFLOW)
    ====================================================  
    This tests the coupling condition for the simple case */
    nfvedge        = 2*fvnet->ndaughters+1; 
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
      /* Left Daughter Branches (pointing into v0) */
      for (i=1; i<fvnet->ndaughters+1; ++i) {
        edgelist[2*i]   = i+1; 
        edgelist[2*i+1] = 0; 
      } /* Right Daughter Branches (pointing away from v1) */
      for(i=fvnet->ndaughters+1; i<2*fvnet->ndaughters+1;++i) {
        edgelist[2*i]   = 1;
        edgelist[2*i+1] = i+1; 
      }

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
      /* vertex */

      for (i=2; i<fvnet->ndaughters+2; ++i) {

        junctions[i].x    = -10.0;
      }
      for (i=fvnet->ndaughters+2; i<2*fvnet->ndaughters+2; ++i) {

        junctions[i].x    = 10.0;
      }
      junctions[0].x = -5.0; 
      junctions[1].x = 5.0; 
      /* Edge */ 
      fvedges[0].nnodes = fvnet->hratio*Mx; 
      for(i=1; i<numEdges; ++i) {
        fvedges[i].nnodes = Mx;  
      }
      fvedges[0].h = 10.0/(PetscReal)fvedges[0].nnodes;
      for (i=1; i<numEdges; i++) {
        fvedges[i].h = 5.0/(PetscReal)fvedges[i].nnodes; 
      }
    }
    break;
  case 5:
    /* Case 5: Roundabout 
    =================================================
      TODO FINISH DRAWING 
    ====================================================  
    This tests the coupling condition for the simple case */
    if (fvnet->ndaughters < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ndaughter must be at least 2 for network 5 (The roundabout) ");
    nfvedge        = 2*fvnet->ndaughters; 
    fvnet->nedge   = nfvedge;
    fvnet->nvertex = nfvedge;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices    = 0;
    numEdges       = 0;
    edgelist       = NULL;
    if (!rank) {
      numVertices = fvnet->nvertex;
      numEdges    = fvnet->nedge;
      ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);

      for (i=0; i<numEdges-2; i+=2) {
        edgelist[2*i]     = i; 
        edgelist[2*i+1]   = i+1; 
        edgelist[2*(i+1)] = i+1;
        edgelist[2*(i+1)+1] = i+3;
      } 
      /* final part of the roundabout */

      edgelist[2*(nfvedge-2)] =   nfvedge -2; 
      edgelist[2*(nfvedge-2)+1] = nfvedge -1;
      edgelist[2*(nfvedge-1)] =   nfvedge -1; 
      edgelist[2*(nfvedge-1)+1] = 1; 

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
      /* vertex */
      for (i=0; i<fvnet->ndaughters; ++i) {
        junctions[2*i].x      = 3.0*(2*i)-3.0; 
        junctions[2*i+1].x    = 3.0*(2*i+1)-3.0; 
      }
      for (i=0; i<fvnet->ndaughters; ++i) {
        fvedges[2*i].nnodes   = fvnet->Mx;
        fvedges[2*i+1].nnodes = fvnet->Mx*fvnet->hratio; 
      }
      for (i=0; i<numEdges; i++) {
        fvedges[i].h = 3.0/(PetscReal)fvedges[i].nnodes; 
      }
    }
    break;
  case 6: 
        /* Case 6: Periodic Boundary conditions 
    =================================================
       v1 --E1--> v0--E0--> v1 
    ====================================================  
          used for convergence tests */
    nfvedge        = 2; 
    fvnet->nedge   = nfvedge;
    fvnet->nvertex = 2;
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
      edgelist[3] = 0;
 

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&fvedges);CHKERRQ(ierr);
      /* vertex */

      junctions[0].x = -5.0; 
      junctions[1].x = 5.0; 
      /* Edge */ 
      for(i=0; i<numEdges; ++i) {
        fvedges[i].nnodes = Mx; 
        fvedges[i].h = 10.0/(PetscReal)fvedges[i].nnodes; 
      }
    }
    break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }


  fvnet->nedge    = numEdges;
  fvnet->nvertex  = numVertices;
  fvnet->edgelist = edgelist;
  fvnet->junction = junctions;
  fvnet->edgefe   = fvedges;
  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc2(dof*dof,&fvnet->R,dof*dof,&fvnet->Rinv);CHKERRQ(ierr);
  ierr = PetscMalloc4(2*dof,&fvnet->uLR,dof,&fvnet->flux,dof,&fvnet->speeds,dof,&fvnet->uPlus);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkSetComponents(DGNetwork fvnet){
  PetscErrorCode    ierr;  
  PetscInt          i,j,e,v,eStart,eEnd,vStart,vEnd,dof = fvnet->physics.dof;
  PetscInt          KeyEdge,KeyJunction,KeyFlux,vfrom,vto,nedges_tmp,nedges,nvertices; 
  PetscInt          *edgelist = NULL;
  EdgeFE            fvedge;
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
  fvedge      = fvnet->edgefe;
  /* Set up the network layout */
  ierr = DMNetworkSetNumSubNetworks(fvnet->network,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = DMNetworkAddSubnetwork(fvnet->network,NULL,nvertices,nedges,edgelist,NULL);CHKERRQ(ierr);

  ierr = DMNetworkLayoutSetUp(fvnet->network);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"fvedgestruct",sizeof(struct _p_EdgeFE),&KeyEdge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(fvnet->network,"flux",0,&KeyFlux);CHKERRQ(ierr);
  /* Add FVEdge component to all local edges. Note that as we have 
     yet to distribute the network, all data is on proc[0]. */
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkAddComponent(fvnet->network,e,KeyEdge,&fvedge[e-eStart],dof*fvedge[e-eStart].nnodes*(fvnet->basisorder+1));CHKERRQ(ierr);
    /* Add a monitor for every edge in the network, label the data according the user provided physics */
    if (fvnet->monitor) { 
      length = fvedge[e-eStart].h*(fvedge[e-eStart].nnodes+1);
      for (j=0; j<dof; j++) {
         ierr = DMNetworkMonitorAdd(fvnet->monitor,fvnet->physics.fieldname[j],e,fvedge[e-eStart].nnodes,j*(fvnet->basisorder+1),dof*(fvnet->basisorder+1),0.0,length,fvnet->ymin,fvnet->ymax,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}
/* Now we have a distributed network. It is assumed that localX and Ftmp have been created in fvnet */
PetscErrorCode FVNetworkBuildDynamic(DGNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscInt       n,j,e,v,i,nedges,dof = fvnet->physics.dof;
  PetscInt       eStart,eEnd,vStart,vEnd,vfrom,vto,offset,*deg;
  const PetscInt *cone,*edges; 
  EdgeFE         fvedge; 
  Junction       junction;
  Vec            localX = fvnet->localX; 
  PetscScalar    *xarr;
  PetscReal      *xnodes,*w,bdry = {-1,1};

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

  /* Build Reference Quadrature */
  ierr = PetscQuadratureCreate(fvnet->comm,fvnet->quad);CHKERRQ(ierr);
  n = PetscFloorReal(fvnet->quadorder/2.0) +1; 
  ierr = PetscMalloc2(n,&xnodes,n,&w);CHKERRQ(ierr);
  ierr = PetscDTGaussQuadrature(n,-1,1,xnodes,w);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(fvnet->quad,1,dof,n,xnodes,w);CHKERRQ(ierr);

  /* Build Reference Legendre Evaluations */
  ierr = PetscMalloc1(fvnet->basisorder+1,&deg);CHKERRQ(ierr);
  ierr = PetscMalloc2(n*(fvnet->basisorder+1),&fvnet->LegEval,n*(fvnet->basisorder+1),&fvnet->LegEvalD);CHKERRQ(ierr);
  for(i=0; i<=fvnet->basisorder; i++) { deg[i] = i; } 
  ierr = PetscDTLegendreEval(n,xnodes,fvnet->basisorder+1,deg,fvnet->LegEvalD,fvnet->LegEvalD,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(fvnet->basisorder+1,&fvnet->LegEvaL_bdry);CHKERRQ(ierr);
  ierr = PetscDTLegendreEval(2,&bdry,fvnet->basisorder+1,deg,fvnet->LegEvaL_bdry,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(fvnet->basisorder+1,&fvnet->Leg_L2);CHKERRQ(ierr);
  for(i=0; i<= fvnet->basisorder; i++) {fvnet->Leg_L2[i] = 1./(2.0*i +1.); } 
  ierr = PetscFree(deg);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}
 /* Iterate through the vertices and assign the coupling flux functions
     This is done by a user provided function that maps the junction type (an integer) to 
     a user specified VertexFlux. A VertexFlux must be provided for all non-boundary types, that 
     is JUNCT junctions and any other user specified coupling junction types. */
PetscErrorCode FVNetworkAssignCoupling(DGNetwork fvnet)
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
PetscErrorCode FVNetworkCleanUp(DGNetwork fvnet)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank; 

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(fvnet->comm,&rank);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->edgelist);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFree2(fvnet->junction,fvnet->edgefe);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkCreateVectors(DGNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(fvnet->network,&fvnet->X);CHKERRQ(ierr);
  ierr = VecDuplicate(fvnet->X,&fvnet->Ftmp);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localX);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FVNetworkDestroy(DGNetwork fvnet) 
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
    ierr = VecDestroy(&junction->R);CHKERRQ(ierr);
    ierr = VecDestroy(&junction->X);CHKERRQ(ierr);
    ierr = MatDestroy(&junction->mat);CHKERRQ(ierr);

  }
  ierr = (*fvnet->physics.destroy)(fvnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<fvnet->physics.dof; i++) {
    ierr = PetscFree(fvnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(fvnet->LegEval,fvnet->LegEvalD);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->Leg_L2);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->LegEvaL_bdry);CHKERRQ(ierr);
  ierr = PetscFree2(fvnet->R,fvnet->Rinv);CHKERRQ(ierr);
  ierr = PetscFree4(fvnet->uLR,fvnet->flux,fvnet->speeds,fvnet->uPlus);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(fvnet->quad);CHKERRQ(ierr);
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

PetscErrorCode FVNetworkProject(DGNetwork fvnet,Vec X0,PetscReal t) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,vfrom,vto,type,offset,e,eStart,eEnd,dof = fvnet->physics.dof,numnodes = 4,id;
  const PetscInt *cone;
  PetscScalar    *xarr,*u,*utmp;
  Junction       junction;
  EdgeFE         fvedge;
  Vec            localX = fvnet->localX;
  PetscReal      h,xfrom,xto,xend,xstart,*xnodes,*w;
  
  PetscFunctionBegin;
  
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = PetscMalloc3(numnodes,&xnodes,numnodes,&w,numnodes*dof,&utmp);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,&type,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetGlobalEdgeIndex(fvnet->network,e,&id);CHKERRQ(ierr);
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
      /* This code assumes a geometrically 1d network. */  
      for (i=0; i<fvedge->nnodes; i++) {
        if (xto>xfrom) {
          xstart = xfrom+i*h;
          xend   = xstart+h;
        } else {
          xstart = xfrom-i*h;
          xend   = xstart-h;
        }
        ierr = PetscDTGaussQuadrature(numnodes,xstart,xend,xnodes,w);CHKERRQ(ierr);
        u = xarr+offset+i*dof;
        for(j=0;j<numnodes;j++) {
          fvnet->physics.sample1d((void*)&fvnet->physics.user,fvnet->initial,t,xnodes[j],utmp+dof*j);
        }
        for(j=0;j<dof;j++) {
            u[j] = 0;
          for(k=0;k<numnodes;k++) {
            u[j] += w[k]*utmp[dof*k+j]/h; /* Gaussian Quadrature*/
          }
        }
      }
    } else { /* Our sample function changes for each edge in the network */
      for (i=0; i<fvedge->nnodes; i++) {
        if (xto>xfrom) {
          xstart = xfrom+i*h;
          xend   = xstart+h;
        } else {
          xstart = xfrom-(i+1)*h;
          xend   = xstart+h;
        }
        ierr = PetscDTGaussQuadrature(numnodes,xstart,xend,xnodes,w);CHKERRQ(ierr);
        u = xarr+offset+i*dof;
        for(j=0;j<numnodes;j++) {
          fvnet->physics.samplenetwork((void*)&fvnet->physics.user,fvnet->initial,t,xnodes[j],utmp+dof*j,id);
        }
        for(j=0;j<dof;j++) {
            u[j] = 0;
          for(k=0;k<numnodes;k++) {
            u[j] += w[k]*utmp[dof*k+j]/h; /* Gaussian Quadrature*/
          }
        }
      }   
    }
  }
  ierr = PetscFree3(xnodes,w,utmp);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  /* Can use insert as each edge belongs to a single processor and vertex data is only for temporary computation and holds no 'real' data. */
  ierr = DMLocalToGlobalBegin(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localX,INSERT_VALUES,X0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}