#include "dgnet.h"
#include <petscdraw.h>

PetscErrorCode DGNetworkCreate(DGNetwork fvnet,PetscInt networktype,PetscInt Mx)
{
  PetscErrorCode ierr;
  PetscInt       nfvedge;
  PetscMPIInt    rank;
  PetscInt       i,field,numVertices,numEdges; 
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
  /* proc[0] creates a sequential fvnet and edgelist    */
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
        fvedges[1].nnodes = Mx; 
        fvedges[2].nnodes = Mx; 

        for (i=0; i<numEdges;i++) {
          fvedges[i].length = 50.0; 
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
          fvedges[i].length = 50.0; 
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
          fvedges[i].length = 50.0; 
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
        fvedges[i].length = fvnet->length; 
      }
    }
    break;
  case 4:
    /* Case 4: ndaughter-1-ndaughter 

    TODO REDO THIS EXAMPLE FOR THE DG CASE 
    =================================================
    (OUTFLOW) v2 --E1--> v0--E0--> v1 --E3--> (OUTFLOW)
                          ^         ^
                          |         |
                          E1        E4
                          |         |
                (OUTFLOW) v3        v4 (OUTFLOW)
    ====================================================  
    This tests the coupling condition for the simple case */

    break;
  case 5:
    /* Case 5: Roundabout 
    =================================================
      TODO FINISH DRAWING 
      TODO REDO FOR DG 
    ====================================================  
    */
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
        fvedges[i].length = 5.0; 
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

  /* 
    TODO : Make all this stuff its own class 

    NOTE: Should I have tensor interface for petsc? Would be really useful for all the tabulation tensors an d
    etc I'm using. Not to mention that effectively the DG solution is a tensor (3rd order )
    element x basis x field

    something to consider.... 
  */

  /* Allocate work space for the DG solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc2(dof*dof,&fvnet->R,dof*dof,&fvnet->Rinv);CHKERRQ(ierr);
  ierr = PetscMalloc4(2*dof,&fvnet->uLR,dof,&fvnet->flux,dof,&fvnet->speeds,dof,&fvnet->uPlus);CHKERRQ(ierr);
  /* allocate work space for the limiter suff */ 

  /* this variable should be stored elsewhere */ 
  fvnet->physics.maxorder =0; 
  for(field=0; field<dof; field++){
    if (fvnet->physics.order[field] > fvnet->physics.maxorder) fvnet->physics.maxorder = fvnet->physics.order[field];
  }

  ierr = PetscMalloc5(dof,&fvnet->limitactive,(fvnet->physics.maxorder+1)*dof,&fvnet->charcoeff,dof,&fvnet->cbdryeval_L,dof,&fvnet->cbdryeval_R,dof,&fvnet->cuAvg);CHKERRQ(ierr);
  ierr = PetscMalloc2(3*dof,&fvnet->uavgs,2*dof,&fvnet->cjmpLR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkSetComponents(DGNetwork fvnet){
  PetscErrorCode    ierr;  
  PetscInt          f,i,e,v,eStart,eEnd,vStart,vEnd,dof = fvnet->physics.dof;
  PetscInt          KeyEdge,KeyJunction,KeyFlux,vfrom,vto,nedges_tmp,nedges,nvertices; 
  PetscInt          *edgelist = NULL,*numComp,*numDof,dim = 1,dmsize;
  EdgeFE            edgefe;
  Junction          junction;
  MPI_Comm          comm = fvnet->comm;
  PetscMPIInt       size,rank;
  PetscReal         low[3] = {0, 0, 0},upper[3] = {1,1,1};
  const PetscInt    *cone,*edges;
  PetscSection      section;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  nedges      = fvnet->nedge;
  nvertices   = fvnet->nvertex; /* local num of vertices, excluding ghosts */
  edgelist    = fvnet->edgelist;
  /* Construct the arrays for building the section for the plexs inside each edge 
     from the user provided physics. HERE WE ARE ASSUMING DG FUNCTION SPACES FOR EVERY 
     FIELD, ALONG WITH 1 COMPONENT PER FIELD (this can and will be altered later) */

  ierr  = PetscMalloc2(dof,&numComp,dof*(dim+1),&numDof);CHKERRQ(ierr);
  for (i = 0; i < dof*(dim+1); ++i) numDof[i] = 0;
  for (i = 0; i < dof; ++i) numComp[i] = 1; 

  /* all variables are stored at the cell level for DG (i.e edges in the 1d case here) */
  for (f = 0; f < dof; ++f) {
    numDof[f*(dim+1)+dim] = fvnet->physics.order[f]+1;
  }
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
    /*
      TODO : Remove EdgeFE from DGNet, refactor how to construct the FE network. THis is definitely a hacky way to do it.  
    */
    edgefe = &fvnet->edgefe[e-eStart]; 
    /* Construct a DMPlex on each edge to manage mesh information */
    upper[0] = edgefe->length;

    /* Anyway to turn off options for this? it will only work with dim 1 for the rest of the code */
    ierr = DMPlexCreateBoxMesh(comm,1,PETSC_FALSE,&edgefe->nnodes,low,upper,NULL,PETSC_TRUE,&edgefe->dm);CHKERRQ(ierr);

    /* Create Field section */ 
    ierr = DMSetNumFields(edgefe->dm, dof);CHKERRQ(ierr);
    ierr = DMPlexCreateSection(edgefe->dm, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, &section);CHKERRQ(ierr);
    /* 
      NOTE: I do not assign names to the field variables as I don't want every edge storing copies of the same field names. 
      These are instead stored in the user provided physics ctx. Anywhere a name is needed, look there, they will be stored in the same 
      order as the field order in this section. 
    */ 
    ierr = DMSetLocalSection(edgefe->dm,section);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(section,&dmsize);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
    ierr = DMSetUp(edgefe->dm);CHKERRQ(ierr);
    /* 
      Add the data from the dmplex to the dmnetwork. We will create the global network vector from the dmnetwork and use the dmplex to manage the 
      data on an edge after getting the offset for set the edge. The dmnetwork creates the vectors and, but the dmplex inside an edge is used to actually 
      interact with the edge componenent of the network vector 
    */
    ierr = DMNetworkAddComponent(fvnet->network,e,KeyEdge,edgefe,dmsize);CHKERRQ(ierr);
  }
  ierr = PetscFree2(numComp,numDof);CHKERRQ(ierr);
  /* Add Junction component to all local vertices. All data is currently assumed to be on proc[0]. Also add the flux component */
  for (v=vStart; v<vEnd; v++) {
    junction = &fvnet->junction[v-vStart];
    ierr = DMNetworkAddComponent(fvnet->network,v,KeyJunction,junction,0);CHKERRQ(ierr);
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
      ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
      ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
      vfrom = cone[0];
      vto   = cone[1];
      if (v==vto) {
        edgefe->offset_vto = dof*i; 
      } else if (v==vfrom) {
        edgefe->offset_vfrom = dof*i; 
      } else {
        SETERRQ2(PetscObjectComm((PetscObject)(fvnet->network)),PETSC_ERR_ARG_WRONG,"v %D != vfrom or vto from supporting edge %D",v,e);
      }
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkAddMonitortoEdges(DGNetwork dgnet, DGNetworkMonitor monitor) {
  PetscErrorCode    ierr;  
  PetscInt          e,eStart,eEnd;

  PetscFunctionBegin;
   ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  if(monitor) {
    for (e = eStart; e<eEnd; e++){
      ierr = DGNetworkMonitorAdd(monitor,e,PETSC_DECIDE,PETSC_DECIDE,dgnet->ymin,dgnet->ymax,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkAddMonitortoEdges_Glvis(DGNetwork dgnet, DGNetworkMonitor_Glvis monitor,PetscViewerGLVisType type) {
  PetscErrorCode    ierr;  
  PetscInt          e,eStart,eEnd;

  PetscFunctionBegin;
   ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  if(monitor) {
    for (e = eStart; e<eEnd; e++){
      ierr = DGNetworkMonitorAdd_Glvis(monitor,e,"localhost",type);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkAddMonitortoEdges_Glvis_3D(DGNetwork dgnet, DGNetworkMonitor_Glvis monitor,PetscViewerGLVisType type) {
  PetscErrorCode    ierr;  
  PetscInt          e,eStart,eEnd;

  PetscFunctionBegin;
   ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  if(monitor) {
    for (e = eStart; e<eEnd; e++){
      ierr = DGNetworkMonitorAdd_Glvis_3D(monitor,e,"localhost",type);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Now we have a distributed network. It is assumed that localX and Ftmp have been created in fvnet */
PetscErrorCode DGNetworkBuildDynamic(DGNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscInt       e,v,i,nedges,dof = fvnet->physics.dof;
  PetscInt       eStart,eEnd,vStart,vEnd,vfrom,vto,offset;
  const PetscInt *cone,*edges; 
  EdgeFE         fvedge; 
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
  ierr = DGNetworkAssignCoupling(fvnet);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);

  PetscFunctionReturn(0); 
}
PetscErrorCode DGNetworkBuildTabulation(DGNetwork dgnet) {
  PetscErrorCode ierr; 
  PetscInt       n,j,i,dof = dgnet->physics.dof,numunique,dim=1;
  PetscInt       *deg,*temp_taborder;
  PetscReal      *xnodes,*w,bdry[2] = {-1,1},*viewnodes;
  PetscBool      unique; 
  
  PetscFunctionBegin;
    /* Iterate through the user provided orders for each field and build the taborder and fieldtotab arrays */
  ierr = PetscMalloc1(dof,&dgnet->fieldtotab);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&temp_taborder);CHKERRQ(ierr);
  /* count number of unique field orders */
  numunique = 0; 
  for(i=0; i<dof; i++) {
    /* Search through the current unique orders for a match */
    unique = PETSC_TRUE;
    for(j=0;j<numunique; j++) {
      if(dgnet->physics.order[i] == temp_taborder[j]) {
        unique = PETSC_FALSE; 
        dgnet->fieldtotab[i] = j;
        break;
      }
    }
    if (unique) {
      dgnet->fieldtotab[i] = numunique;
      temp_taborder[numunique++] = dgnet->physics.order[i];
    }
  }
  /* now we have the number of unique orders and what they are in fieldtotab (which is being reused here) */
  ierr = PetscMalloc1(numunique,&dgnet->taborder); 
  dgnet->tabordersize = numunique;
  for(i=0; i<dgnet->tabordersize; i++) {
    dgnet->taborder[i] = temp_taborder[i]; 
  }
  ierr = PetscFree(temp_taborder);CHKERRQ(ierr);
  ierr = PetscMalloc4(dgnet->tabordersize,&dgnet->LegEval,dgnet->tabordersize,
          &dgnet->Leg_L2,dgnet->tabordersize,&dgnet->LegEvalD,dgnet->tabordersize,&dgnet->LegEvaL_bdry);CHKERRQ(ierr);
  ierr = PetscMalloc1(dgnet->tabordersize,&dgnet->comp);CHKERRQ(ierr);
  /* Internal Viewer Storage stuff (to be migrated elsewhere) */
  ierr = PetscMalloc2(dgnet->tabordersize,&dgnet->LegEval_equispaced,dgnet->tabordersize,&dgnet->numviewpts);CHKERRQ(ierr);
    /* Build Reference Quadrature (Single Quadrature for all fields (maybe generalize but not now) */
    ierr = PetscQuadratureCreate(dgnet->comm,&dgnet->quad);CHKERRQ(ierr);
    /* Find maximum ordeer */
    n = 0; 
    for(i=0; i<dgnet->tabordersize; i++) {
      if(n < PetscCeilReal(dgnet->taborder[i])+1) n =  PetscCeilReal(dgnet->taborder[i])+1;
    }
    ierr = PetscMalloc2(n,&xnodes,n,&w);CHKERRQ(ierr);
    ierr = PetscDTGaussQuadrature(n,-1,1,xnodes,w);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(dgnet->quad,dim,1,n,xnodes,w);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(dgnet->quad,2*n);CHKERRQ(ierr);
    ierr = PetscMalloc2(dof,&dgnet->pteval,dof*n,&dgnet->fluxeval);CHKERRQ(ierr);
  for (i=0; i<dgnet->tabordersize; i++) {
    /* Build Reference Legendre Evaluations */
    ierr = PetscMalloc1(dgnet->taborder[i]+1,&deg);CHKERRQ(ierr);
    ierr = PetscMalloc2(n*(dgnet->taborder[i]+1),&dgnet->LegEval[i],n*(dgnet->taborder[i]+1),&dgnet->LegEvalD[i]);CHKERRQ(ierr);
    for(j=0; j<=dgnet->taborder[i]; j++) { deg[j] = j; } 
    ierr = PetscDTLegendreEval(n,xnodes,dgnet->taborder[i]+1,deg,dgnet->LegEval[i],dgnet->LegEvalD[i],PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*(dgnet->taborder[i]+1),&dgnet->LegEvaL_bdry[i]);CHKERRQ(ierr);
    ierr = PetscDTLegendreEval(2,bdry,dgnet->taborder[i]+1,deg,dgnet->LegEvaL_bdry[i],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(dgnet->taborder[i]+1,&dgnet->Leg_L2[i]);CHKERRQ(ierr);
    for(j=0; j<=dgnet->taborder[i]; j++) {dgnet->Leg_L2[i][j] = (2.0*j +1.)/(2.); }
    /* Viewer evaluations to be migrated */
    dgnet->numviewpts[i] = dgnet->taborder[i]+1; /* DO NOT CHANGE THIS WITHOUT CREATING A TABULATION FOR GLVIS VISUALIZATION */
    ierr = PetscMalloc1(dgnet->numviewpts[i],&viewnodes);CHKERRQ(ierr);
    for(j=0; j<dgnet->numviewpts[i]; j++) viewnodes[j] = 2.*j/(dgnet->numviewpts[i]-1) - 1.;
    ierr = PetscMalloc1(dgnet->numviewpts[i]*(dgnet->taborder[i]+1),&dgnet->LegEval_equispaced[i]);CHKERRQ(ierr);
    ierr = PetscDTLegendreEval(dgnet->numviewpts[i],viewnodes,dgnet->taborder[i]+1,deg,dgnet->LegEval_equispaced[i],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscFree(viewnodes);CHKERRQ(ierr);
    ierr = PetscFree(deg);CHKERRQ(ierr);

    /* Workspace */
    ierr = PetscMalloc1(dgnet->taborder[i]+1,&dgnet->comp[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
PetscErrorCode LegendreTabulationViewer_Internal(PetscInt npoints, PetscInt ndegree, PetscViewer viewer, PetscReal *LegEval) {
  PetscErrorCode ierr;
  PetscInt       deg,qpoint; 
  PetscReal      viewerarray[npoints]; /* For some reason malloc was giving me memory corruption, but this works ... */
  
  PetscFunctionBegin;
  /* View each row individually (makes more sense to view) */
  for(deg = 0; deg<= ndegree; deg++) {
    ierr = PetscViewerASCIIPrintf(viewer,"Degree %i Evaluations \n",deg);CHKERRQ(ierr);
    for(qpoint = 0; qpoint < npoints; qpoint++) {
      *(viewerarray+qpoint) = LegEval[qpoint*(ndegree+1)+deg];
    }
    ierr = PetscRealView(npoints,viewerarray,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* 
 TODO refactor as a petsc_____view function ? 
*/
PetscErrorCode ViewDiscretizationObjects(DGNetwork dgnet,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,quadsize;
  PetscInt       ndegree; 
  PetscFunctionBegin;


  /* call standard viewers for discretization objects if available */
    ierr = PetscQuadratureView(dgnet->quad,viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(dgnet->quad,NULL,NULL,&quadsize,NULL,NULL);CHKERRQ(ierr);
  /* View the tabulation arrays
    TODO as per other comments, these arrays should be petsctabulation objects and this should be its dedicated viewing routine 
  */
    ierr = PetscViewerASCIIPrintf(viewer,"Quadsize: %i \n",quadsize);CHKERRQ(ierr);

  /* Iterate through the tabulation Orders */
  for (i=0; i<dgnet->tabordersize; i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"Legendre Tabulation Order: %i \n \n",dgnet->taborder[i]);CHKERRQ(ierr);
    /* Hack to make use of PetscRealViewer function */
    /* Maybe should be redone to have everything stored as Matrices, or custom storage? Idk man, either
       way it will work for now, though involves silly copying of data to get the arrays in the right format 
       for viewing. Basically transposing the induced matrix from this data */
    ndegree = dgnet->taborder[i];

    ierr = PetscViewerASCIIPrintf(viewer,"Legendre Evaluations at Quadrature Points \n");CHKERRQ(ierr);
    ierr = LegendreTabulationViewer_Internal(quadsize,ndegree,viewer,dgnet->LegEval[i]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"Legendre Derivative Evaluations at Quadrature Points \n");CHKERRQ(ierr);
    ierr = LegendreTabulationViewer_Internal(quadsize,ndegree,viewer,dgnet->LegEvalD[i]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"Legendre Evaluations at Boundary Quadrature \n");CHKERRQ(ierr);
    /* Fix hard coded 1D code here. We assume that the boundary evaluation quadrature has only two points */
    ierr = LegendreTabulationViewer_Internal(2,ndegree,viewer,dgnet->LegEvaL_bdry[i]);CHKERRQ(ierr); 

    ierr = PetscViewerASCIIPrintf(viewer,"Legendre Normalization\n");CHKERRQ(ierr);
    ierr = PetscRealView(ndegree+1,dgnet->Leg_L2[i],viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*
  TODO : Refactor as PetscView Function

  Function for Viewing the Mesh information inside of the dgnet (just calls dmview for each
  dmplex inside the edges)
*/
PetscErrorCode DGNetworkViewEdgeDMs(DGNetwork dgnet,PetscViewer viewer) 
{
  PetscErrorCode ierr;
  PetscInt       e,eStart,eEnd;
  EdgeFE         edgefe;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for(e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\n Mesh on Edge %i \n \n ",e);CHKERRQ(ierr);
    ierr = DMView(edgefe->dm,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* Just prints the jacobian and inverse jacobians to screen for dms inside the edgee 

ONLY WORKS FOR 1D MESHES FOR NOW !!!! */ 
PetscErrorCode DGNetworkViewEdgeGeometricInfo(DGNetwork dgnet, PetscViewer viewer){
  PetscErrorCode ierr;
  PetscInt       e,eStart,eEnd,c,cStart,cEnd;
  EdgeFE         edgefe;
  PetscReal      J,Jinv,Jdet;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(dgnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for(e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(dgnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"\n \n Geometric Info on Edge %i \n \n \n ",e);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    for (c = cStart; c<cEnd; c++) {
      ierr = DMPlexComputeCellGeometryAffineFEM(edgefe->dm,c,NULL,&J,&Jinv,&Jdet);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Cell %i: J: %e  - Jinv: %e - Jdet: %e \n  ",c,J,Jinv,Jdet);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
 /* Iterate through the vertices and assign the coupling flux functions
     This is done by a user provided function that maps the junction type (an integer) to 
     a user specified VertexFlux. A VertexFlux must be provided for all non-boundary types, that 
     is JUNCT junctions and any other user specified coupling junction types. */
PetscErrorCode DGNetworkAssignCoupling(DGNetwork fvnet)
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
PetscErrorCode DGNetworkCleanUp(DGNetwork fvnet)
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
PetscErrorCode DGNetworkCreateVectors(DGNetwork fvnet)
{
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(fvnet->network,&fvnet->X);CHKERRQ(ierr);
  ierr = VecDuplicate(fvnet->X,&fvnet->Ftmp);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localX);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fvnet->network,&fvnet->localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkDestroyTabulation(DGNetwork fvnet){
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin; 
  for (i=0; i<fvnet->tabordersize; i++) {
    ierr = PetscFree2(fvnet->LegEval[i],fvnet->LegEvalD[i]);CHKERRQ(ierr);
    ierr = PetscFree(fvnet->Leg_L2[i]);CHKERRQ(ierr);
    ierr = PetscFree(fvnet->LegEvaL_bdry[i]);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&fvnet->quad);CHKERRQ(ierr);
    ierr = PetscFree(fvnet->comp[i]);CHKERRQ(ierr);
    ierr = PetscFree(fvnet->LegEval_equispaced[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree5(fvnet->Leg_L2,fvnet->LegEval,fvnet->LegEvaL_bdry,fvnet->LegEvalD,fvnet->quad);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->taborder);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->fieldtotab);CHKERRQ(ierr);
  ierr = PetscFree(fvnet->comp);CHKERRQ(ierr);
  ierr = PetscFree2(fvnet->fluxeval,fvnet->pteval);CHKERRQ(ierr);
  ierr = PetscFree2(fvnet->LegEval_equispaced,fvnet->numviewpts);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}
PetscErrorCode DGNetworkDestroy(DGNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       i,v,e,eStart,eEnd,vStart,vEnd;
  Junction       junction;
  EdgeFE         edgefe;

  PetscFunctionBegin; 
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  for(e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = DMDestroy(&edgefe->dm);CHKERRQ(ierr);
    ierr = DMDestroy(&edgefe->dmaux);CHKERRQ(ierr);
  }
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    /* Free dynamic memory for the junction component */
    ierr = PetscFree(junction->dir);CHKERRQ(ierr); 
    ierr = PetscFree(junction->flux);CHKERRQ(ierr);
    ierr = VecDestroy(&junction->rcouple);CHKERRQ(ierr);
    ierr = VecDestroy(&junction->xcouple);CHKERRQ(ierr);
    ierr = MatDestroy(&junction->mat);CHKERRQ(ierr);
  }
  ierr = (*fvnet->physics.destroy)(fvnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<fvnet->physics.dof; i++) {
    ierr = PetscFree(fvnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(fvnet->R,fvnet->Rinv);CHKERRQ(ierr);
  ierr = PetscFree4(fvnet->uLR,fvnet->flux,fvnet->speeds,fvnet->uPlus);CHKERRQ(ierr);
  ierr = PetscFree5(fvnet->charcoeff,fvnet->limitactive,fvnet->cbdryeval_L,fvnet->cbdryeval_R,fvnet->cuAvg);CHKERRQ(ierr);
  ierr = PetscFree2(fvnet->uavgs,fvnet->cjmpLR);CHKERRQ(ierr);
  ierr = DGNetworkDestroyTabulation(fvnet);CHKERRQ(ierr);

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

/*DGNetwork Viewing Functions, to be refactored and moved elsewhere */
PetscReal evalviewpt_internal(DGNetwork dgnet, PetscInt field, PetscInt viewpt,const PetscReal *comp) {
  PetscInt deg,tab = dgnet->fieldtotab[field],ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0; 

  for(deg=0; deg<=ndegree; deg++) {
    eval += comp[deg]* dgnet->LegEval_equispaced[tab][viewpt*(ndegree+1)+deg];
  }
  return eval; 
}
PetscErrorCode DGNetworkMonitorCreate(DGNetwork dgnet,DGNetworkMonitor *monitorptr)
{
  PetscErrorCode   ierr;
  DGNetworkMonitor monitor;
  MPI_Comm         comm;
  PetscMPIInt      size;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dgnet->network,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Parallel DGNetworkMonitor is not supported yet");

  ierr = PetscMalloc1(1,&monitor);CHKERRQ(ierr);
  monitor->comm      = comm;
  monitor->dgnet     = dgnet; 
  monitor->firstnode = NULL;

  *monitorptr = monitor;
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorPop(DGNetworkMonitor monitor)
{
  PetscErrorCode       ierr;
  DGNetworkMonitorList node;

  PetscFunctionBegin;
  if (monitor->firstnode) {
    /* Update links */
    node = monitor->firstnode;
    monitor->firstnode = node->next;

    /* Free list node */
    ierr = PetscViewerDestroy(&(node->viewer));CHKERRQ(ierr);
    ierr = VecDestroy(&(node->v));CHKERRQ(ierr);
    ierr = PetscFree(node);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorDestroy(DGNetworkMonitor *monitor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while ((*monitor)->firstnode) {
    ierr = DGNetworkMonitorPop(*monitor);CHKERRQ(ierr);
  }
  ierr = PetscFree(*monitor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ymax and ymin must be removed by the caller */
PetscErrorCode DGNetworkMonitorAdd(DGNetworkMonitor monitor,PetscInt element,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscBool hold)
{
  PetscErrorCode       ierr;
  PetscDrawLG          drawlg;
  PetscDrawAxis        axis;
  PetscMPIInt          rank, size;
  DGNetworkMonitorList node;
  char                 titleBuffer[64];
  PetscInt             vStart,vEnd,eStart,eEnd,viewsize,field,cStart,cEnd;
  DM                   network=monitor->dgnet->network;
  DGNetwork            dgnet=monitor->dgnet;
  PetscInt             dof=dgnet->physics.dof;
  EdgeFE               edgefe; 

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(monitor->comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(monitor->comm, &size);CHKERRMPI(ierr);

  ierr = DMNetworkGetVertexRange(network, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(network, &eStart, &eEnd);CHKERRQ(ierr);
  /* make a viewer for each field on the componenent */
  for(field=0; field<dof; field++) {
    /* Make window title */
    if (vStart <= element && element < vEnd) {
      /* Nothing to view on the vertices for DGNetwork (for now) so skip */
      PetscFunctionReturn(0);
    } else if (eStart <= element && element < eEnd) {
      ierr = PetscSNPrintf(titleBuffer, 64, "%s @ edge %d [%d / %d]", dgnet->physics.fieldname[field], element - eStart, rank, size-1);CHKERRQ(ierr);
    } else {
      /* vertex / edge is not on local machine, so skip! */
      PetscFunctionReturn(0);
    }
    ierr = PetscMalloc1(1, &node);CHKERRQ(ierr);
    /* Setup viewer. */
    ierr = PetscViewerDrawOpen(monitor->comm, NULL, titleBuffer, PETSC_DECIDE, PETSC_DECIDE, PETSC_DRAW_QUARTER_SIZE, PETSC_DRAW_QUARTER_SIZE, &(node->viewer));CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(node->viewer, PETSC_VIEWER_DRAW_LG_XRANGE);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDrawLG(node->viewer, 0, &drawlg);CHKERRQ(ierr);
    ierr = PetscDrawLGGetAxis(drawlg, &axis);CHKERRQ(ierr);
    if (xmin != PETSC_DECIDE && xmax != PETSC_DECIDE) {
      ierr = PetscDrawAxisSetLimits(axis, xmin, xmax, ymin, ymax);CHKERRQ(ierr);
    } else {
      ierr = PetscDrawAxisSetLimits(axis, 0, 1, ymin, ymax);CHKERRQ(ierr);
    }
    ierr = PetscDrawAxisSetHoldLimits(axis, hold);CHKERRQ(ierr);

    /* Setup vector storage for drawing. */
    ierr  = DMNetworkGetComponent(network,element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    viewsize = dgnet->numviewpts[dgnet->fieldtotab[field]]*(cEnd-cStart);
    ierr = VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v));CHKERRQ(ierr);

    node->element   = element;
    node->field     = field;
    node->next         = monitor->firstnode;
    node->vsize    = viewsize;
    monitor->firstnode = node;
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorView(DGNetworkMonitor monitor,Vec x)
{
  PetscErrorCode      ierr;
  PetscInt            edgeoff,fieldoff,cStart,cEnd,c,tab,q,viewpt;
  const PetscScalar   *xx;
  PetscScalar         *vv;
  DGNetworkMonitorList node;
  DM                   network=monitor->dgnet->network;
  DGNetwork            dgnet=monitor->dgnet;
  EdgeFE               edgefe;
  PetscSection         section;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x, &xx);CHKERRQ(ierr);
  for (node = monitor->firstnode; node; node = node->next) {
    ierr = DMNetworkGetLocalVecOffset(network, node->element, FVEDGE, &edgeoff);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(dgnet->network,node->element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = VecGetArray(node->v, &vv);CHKERRQ(ierr);

    ierr  = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr  = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    tab = dgnet->fieldtotab[node->field];
    /* Evaluate at the eqiudistant point evalutions */
    viewpt = 0;
    for(c=cStart; c<cEnd; c++) {
      ierr = PetscSectionGetFieldOffset(section,c,node->field,&fieldoff);CHKERRQ(ierr);
      for(q=0; q<dgnet->numviewpts[tab]; q++) {
       vv[viewpt++]=evalviewpt_internal(dgnet,node->field,q,xx+edgeoff+fieldoff);
      }
    }
    ierr = VecRestoreArray(node->v, &vv);CHKERRQ(ierr);
    ierr = VecView(node->v, node->viewer);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x, &xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorCreate_Glvis(DGNetwork dgnet,DGNetworkMonitor_Glvis *monitorptr)
{
  PetscErrorCode   ierr;
  DGNetworkMonitor_Glvis monitor;
  MPI_Comm         comm;
  PetscMPIInt      size;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dgnet->network,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Parallel DGNetworkMonitor is not supported yet");

  ierr = PetscMalloc1(1,&monitor);CHKERRQ(ierr);
  monitor->comm      = comm;
  monitor->dgnet     = dgnet; 
  monitor->firstnode = NULL;

  *monitorptr = monitor;
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitor_g2l_internal(PetscObject V,PetscInt nfields,PetscObject Vfield[],void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis) ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe; 
  PetscInt                   c,cStart,cEnd,field,tab,dof=dgnet->physics.dof,i,fieldoff,deg,ndegree; 
  PetscErrorCode             ierr; 
  PetscSection               section;
  const PetscReal            *v; 
  PetscReal                  *vwork; 

  PetscFunctionBegin;
  ierr = DMNetworkGetComponent(network,node->element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
  ierr = VecGetArrayRead((Vec)V,&v);CHKERRQ(ierr);
  /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
  for(field=0; field<dof; field++) {
    i=0; 
    ierr = VecGetArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr); 
    for(c=cStart; c<cEnd; c++) {
      ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      for(deg=0; deg<=ndegree; deg++) {
        vwork[i++] =  evalviewpt_internal(dgnet,field,deg, v+fieldoff);
      }
    }
    ierr = VecRestoreArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead((Vec)V,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}
PetscErrorCode DGNetworkMonitor_destroyctx_internal(void *ctx) {
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorAdd_Glvis(DGNetworkMonitor_Glvis monitor,PetscInt element,const char hostname[],PetscViewerGLVisType type)
{
  PetscErrorCode       ierr;
  PetscMPIInt          rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt             viewsize,field,cStart,cEnd,tab,Dim = 1; ;
  DM                   network=monitor->dgnet->network;
  DGNetwork            dgnet=monitor->dgnet;
  PetscInt             dof=dgnet->physics.dof;
  EdgeFE               edgefe;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(monitor->comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(monitor->comm, &size);CHKERRMPI(ierr);

  ierr = PetscMalloc1(1, &node);CHKERRQ(ierr);
  ierr = PetscMalloc3(dof,&node->dim,dof,&node->v_work,dof,&node->fec_type);CHKERRQ(ierr);

  ierr = PetscViewerGLVisOpen(monitor->comm,type,hostname,PETSC_DECIDE,&node->viewer);CHKERRQ(ierr);

  ierr = DMNetworkGetComponent(network,element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
  ierr = DMClone(edgefe->dm,&node->viewdm);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
      /* make the work vector for each field */
  for(field=0; field<dof; field++) {
    /* Setup vector storage for drawing. */
    tab        = dgnet->fieldtotab[field];
    viewsize   = (cEnd-cStart)*(dgnet->taborder[tab]+1); /* number of variables for the given field */
    ierr       = VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field]));CHKERRQ(ierr);
    ierr       = PetscObjectCompose((PetscObject)node->v_work[field],"__PETSc_dm",(PetscObject)edgefe->dm);CHKERRQ(ierr); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    ierr       = PetscMalloc(64,&node->fec_type[field]);CHKERRQ(ierr);
    ierr       = PetscSNPrintf(node->fec_type[field],64,"FiniteElementCollection: L2_T4_%iD_P%i",Dim,dgnet->taborder[tab]);CHKERRQ(ierr); 
    node->dim[field] = Dim; 
  }
  ierr = DMCreateGlobalVector(edgefe->dm,&node->v);CHKERRQ(ierr);

  node->element      = element;
  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  monitor->firstnode = node;

  ierr = PetscViewerGLVisSetFields(node->viewer,dof,(const char**)node->fec_type,node->dim,DGNetworkMonitor_g2l_internal,(PetscObject*)node->v_work,(void*)node,DGNetworkMonitor_destroyctx_internal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorPop_Glvis(DGNetworkMonitor_Glvis monitor)
{
  PetscErrorCode             ierr;
  DGNetworkMonitorList_Glvis node;
  PetscInt                   field,dof = monitor->dgnet->physics.dof; 

  PetscFunctionBegin;
  if (monitor->firstnode) {
    /* Update links */
    node = monitor->firstnode;
    monitor->firstnode = node->next;
    /* Free list node */
    ierr = PetscViewerDestroy(&(node->viewer));CHKERRQ(ierr);
    ierr = VecDestroy(&(node->v));CHKERRQ(ierr);
    for(field=0; field<dof; field++) {
      ierr = VecDestroy(&node->v_work[field]);CHKERRQ(ierr);
      ierr = PetscFree(node->fec_type[field]);CHKERRQ(ierr);
    }
    ierr = PetscFree3(node->v_work,node->dim,node->fec_type);CHKERRQ(ierr);
    if(node->viewdm != NULL) ierr = DMDestroy(&node->viewdm);CHKERRQ(ierr); 
    ierr = PetscFree(node);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorDestroy_Glvis(DGNetworkMonitor_Glvis *monitor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while ((*monitor)->firstnode) {
    ierr = DGNetworkMonitorPop_Glvis(*monitor);CHKERRQ(ierr);
  }
  ierr = PetscFree(*monitor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorView_Glvis(DGNetworkMonitor_Glvis monitor,Vec x)
{
  PetscErrorCode      ierr;
  PetscInt            edgeoff,i,vecsize;
  const PetscScalar   *xx;
  PetscScalar         *vv;
  DGNetworkMonitorList_Glvis node;
  DM                   network = monitor->dgnet->network;
  DGNetwork            dgnet   = monitor->dgnet;
  EdgeFE               edgefe;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x, &xx);CHKERRQ(ierr);
  for (node = monitor->firstnode; node; node = node->next) {
    ierr = DMNetworkGetLocalVecOffset(network, node->element, FVEDGE, &edgeoff);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(dgnet->network,node->element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = VecGetArray(node->v, &vv);CHKERRQ(ierr);
    ierr = VecGetSize(node->v,&vecsize);CHKERRQ(ierr);
    for(i=0; i<vecsize; i++) {
      vv[i] = xx[edgeoff+i];
    }
    ierr = VecRestoreArray(node->v, &vv);CHKERRQ(ierr);
    ierr = VecView(node->v, node->viewer);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x, &xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_l(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy   = 2*x[1]-1,zz = 2*x[2]-1; 

  xp[1] = yy*PetscSqrtReal(1-PetscPowReal(zz,2)/2.)/10.; 
  xp[2] =  zz*PetscSqrtReal(1-PetscPowReal(yy,2)/2.)/10.; 
  xp[0] = 2.*x[0]+0.1; 
}
/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy   = 2*x[1]-1,zz = 2*x[2]-1; 

  xp[1] = yy*PetscSqrtReal(1-PetscPowReal(zz,2)/2.); 
  xp[2] =  zz*PetscSqrtReal(1-PetscPowReal(yy,2)/2.); 
  xp[0] = x[0]; 
}
/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_r(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy   = 2*x[1]-1,zz = 2*x[2]-1; 

  xp[1] = yy*PetscSqrtReal(1-PetscPowReal(zz,2)/2.)/10.0; 
  xp[2] =  zz*PetscSqrtReal(1-PetscPowReal(yy,2)/2.)/10.0; 
  xp[0] = x[0]*2. - 2.1; /*hack for presentation */
}
/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy   = 2*x[1]-1,zz = 2*x[2]-1; 

  xp[0] = yy*PetscSqrtReal(1-PetscPowReal(zz,2)/2.)/10.; 
  xp[2] =  zz*PetscSqrtReal(1-PetscPowReal(yy,2)/2.)/10.; 
  xp[1] = 2.*x[0]+0.1; 
}
/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_b(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy   = 2*x[1]-1,zz = 2*x[2]-1; 

  xp[0] = yy*PetscSqrtReal(1-PetscPowReal(zz,2)/2.)/10.; 
  xp[2] =  zz*PetscSqrtReal(1-PetscPowReal(yy,2)/2.)/10.; 
  xp[1] = -2.*x[0]-0.1; 
}
static PetscErrorCode DGNetworkCreateViewDM(DM dm)
{
  DM             cdm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, dE, cStart,size;
  PetscBool      simplex;
  PetscErrorCode ierr;
  PetscReal      *coord; 
  Vec            Coord; 

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex,3,PETSC_DECIDE, &fe);CHKERRQ(ierr);
  ierr = DMProjectCoordinates(dm, fe);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm,&Coord);CHKERRQ(ierr);
  ierr = VecGetSize(Coord,&size);
  ierr = VecGetArray(Coord,&coord);CHKERRQ(ierr);
  ierr = VecRestoreArray(Coord,&coord);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMPlexRemapGeometry(dm, 0.0, f0_circle);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm,&Coord);CHKERRQ(ierr);
  ierr = VecGetSize(Coord,&size);
  ierr = VecGetArray(Coord,&coord);CHKERRQ(ierr);
  ierr = VecRestoreArray(Coord,&coord);
  PetscFunctionReturn(0);
}
static PetscErrorCode DGNetworkCreateViewDM2(DM dm)
{
  DM             cdm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, dE, cStart;
  PetscBool      simplex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex,3,PETSC_DECIDE, &fe);CHKERRQ(ierr);
  ierr = DMProjectCoordinates(dm, fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitor_3D_g2l_internal(PetscObject V,PetscInt nfields,PetscObject Vfield[],void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis) ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe; 
  PetscInt                   copy,c,cStart,cEnd,field,tab,dof=dgnet->physics.dof,i,fieldoff,deg,ndegree; 
  PetscErrorCode             ierr; 
  PetscSection               section;
  const PetscReal            *v; 
  PetscReal                  *vwork; 

  PetscFunctionBegin;
  ierr = DMNetworkGetComponent(network,node->element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
  ierr = VecGetArrayRead((Vec)V,&v);CHKERRQ(ierr);
  /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
  for(field=0; field<dof; field++) {
    i=0; 
    ierr = VecGetArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr); 
    for(c=cStart; c<cEnd; c++) {
      ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
      tab = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      for(deg=0; deg<=ndegree; deg++) {
        vwork[i] =  evalviewpt_internal(dgnet,field,deg, v+fieldoff);
        for(copy=1; copy<(ndegree+1)*(ndegree+1); copy++) {
          vwork[i+copy*(ndegree+1)] = vwork[i];
        }
        i++; 
      }
      i+=(ndegree+1)*((ndegree+1)*(ndegree+1)-1);
    }
    ierr = VecRestoreArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead((Vec)V,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorAdd_Glvis_3D(DGNetworkMonitor_Glvis monitor,PetscInt element,const char hostname[],PetscViewerGLVisType type)
{
  PetscErrorCode       ierr;
  PetscMPIInt          rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt             viewsize,field,cStart,cEnd,tab,Dim = 3;
  DM                   network=monitor->dgnet->network;
  DGNetwork            dgnet=monitor->dgnet;
  PetscInt             dof=dgnet->physics.dof;
  EdgeFE               edgefe;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(monitor->comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(monitor->comm, &size);CHKERRMPI(ierr);

  ierr = PetscMalloc1(1, &node);CHKERRQ(ierr);
  ierr = PetscMalloc3(dof,&node->dim,dof,&node->v_work,dof,&node->fec_type);CHKERRQ(ierr);

  ierr = PetscViewerGLVisOpen(monitor->comm,type,hostname,PETSC_DECIDE,&node->viewer);CHKERRQ(ierr);

  ierr = DMNetworkGetComponent(network,element,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  PetscInt faces[3]={cEnd-cStart,1,1}; 
  ierr = DMPlexCreateBoxMesh(monitor->comm, 3, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &node->viewdm);CHKERRQ(ierr);
  ierr = DGNetworkCreateViewDM(node->viewdm);CHKERRQ(ierr);
      /* make the work vector for each field */
  for(field=0; field<dof; field++) {
    /* Setup vector storage for drawing. */
    tab        = dgnet->fieldtotab[field];
    viewsize   = (cEnd-cStart)*PetscPowInt((dgnet->taborder[tab]+1),Dim); /* number of variables for the given field */
    ierr       = VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field]));CHKERRQ(ierr);
    ierr       = PetscObjectCompose((PetscObject)node->v_work[field],"__PETSc_dm",(PetscObject)node->viewdm);CHKERRQ(ierr); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    ierr       = PetscMalloc(64,&node->fec_type[field]);CHKERRQ(ierr);
    ierr       = PetscSNPrintf(node->fec_type[field],64,"FiniteElementCollection: L2_T4_%iD_P%i",Dim,dgnet->taborder[tab]);CHKERRQ(ierr); 
    node->dim[field] = Dim; 
  }
  ierr = DMCreateGlobalVector(edgefe->dm,&node->v);CHKERRQ(ierr);

  node->element      = element;
  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  monitor->firstnode = node;

  ierr = PetscViewerGLVisSetFields(node->viewer,dof,(const char**)node->fec_type,node->dim,DGNetworkMonitor_3D_g2l_internal,(PetscObject*)node->v_work,(void*)node,DGNetworkMonitor_destroyctx_internal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Experimental work on "adding" dmplex objects together */


/* Convience create from DAG function that only creates the topology, leaving the geometry dm and section uncreated */

PetscErrorCode DMPlexCreateFromDAG_Topological(DM dm, PetscInt depth, const PetscInt numPoints[], const PetscInt coneSize[], const PetscInt cones[], const PetscInt coneOrientations[])
{
  PetscInt       firstVertex = -1, pStart = 0, pEnd = 0, p, dim, dimEmbed, d, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  if (dimEmbed < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Embedding dimension %D cannot be less than intrinsic dimension %d",dimEmbed,dim);
  for (d = 0; d <= depth; ++d) pEnd += numPoints[d];
  ierr = DMPlexSetChart(dm, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, p, coneSize[p-pStart]);CHKERRQ(ierr);
    if (firstVertex < 0 && !coneSize[p - pStart]) {
      firstVertex = p - pStart;
    }
  }
  if (firstVertex < 0 && numPoints[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected %D vertices but could not find any", numPoints[0]);
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
  for (p = pStart, off = 0; p < pEnd; off += coneSize[p-pStart], ++p) {
    ierr = DMPlexSetCone(dm, p, &cones[off]);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, p, &coneOrientations[off]);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* The overall goal for code like this would be to provide high level toplogy manipulation support for dmplex, 
to add dmplex objects together by "indentifying" a set of depth 0 cells for example or more general operations. 
Hopefully allowing for topological "stitching" operations. What this would look like in general I'm not sure. I'll
just add what I need as I go and maybe generalize once I understand what the generalization should look like. 
I guess the overall vision is that dmplex support a representation of cell complexs, and any operation that makes sense 
on cell complexs should have a corresponding high level command in petsc. So algebraic topology in petsc :). Need to 
learn algebraic toplogy first though */ 


/* Here we have a command to add a set of dm objects disconnectedly. So we simply have a set of N dm objects 
"added" to produce a global number of all N meshes, but without changing any topological information. The purpose 
of this is to add the dmplex objects in each edge of the dgnetwork to form a global dmplex object, so I can 
make use of standard dmplex output format techniques, in particular I can visualize a dgnetwork object 
using glvis as a single network object. Currently limited to visuzalizing each edge dmplex object seperately (along )
with field information on each dmplex object */

/* 
  TODO - This code is sequential for now only. Then will be extended to parallel with the assumption that 
  each dm "added" lives entirely on single processor. Finally the full version will be added later (though 
  is not needed for my purposes so definitely less motivation)
*/ 

/* 
  This is actually pretty tricky to "correctly". I think these operations should actually be low-level 
  kernel-esque operations. Manual manipulation of the internals of dmplex. For example building up 
  the mapping from the summands to sum dm, (giving offset) is fairly tricky, and with how I'm doing it will 
  only work for depth 0 cells and codepth 0 cells (vertices), as I use DMPlexCreateFromCellListParallelPetsc
  to create the sum dmplex object, and have no direct control over the numbering for the other 
  cw-plex entities and so cannot (easily) generate a mapping. The logic for using DMPlexCreateFromCellListParallelPetsc
  is that it can build a dmplex from parallel input, directly with PETSc api, which I currently 
  don't know how to do manually myself (have to manipulate the petscsf object in the dmplx, which I don't 
  know how to do.

  tldr: This function is a hack that needs to be rewritten with help from petsc dmplex people. Especially if 
  I want to generalize "indentifying" cw-plex entities from the summand cw-plexs. 
*/

/*
  Serial ONLY !!!
*/

/* Note that here we may reorder the standard dmplex storage order of Elements, Vertices, other stratutm 
and instead just order depth down. Shouldn't matter... we shall see if it breaks anything */

PetscErrorCode DMPlexAdd_Disconnected(DM *dmlist,PetscInt numdm, DM *dmsum, PetscSection *stratumoffsets) {
  PetscErrorCode ierr; 
  PetscInt       p,i,j,k,depth,depth_temp,dim,dim_prev,dim_top,dim_top_temp,pStart,pEnd,chartsize,stratum,totalconesize;
  PetscInt       *numpoints_g, *coneSize_g, *cones_g, *coneOrientations_g,coneSize,off,prevtotal;
  const PetscInt *cone,*coneOrientation;
  const PetscScalar *vertexcoords; 
  DMType         dmtype;
  MPI_Comm       comm = PetscObjectComm((PetscObject)dmlist[0]);
  PetscSection   offsets; 
  char           fieldname[64]; /* Should be long enough unless we get crazy deep complexs */
  DM             dm_sum;
  PetscBool      flag; 

  PetscFunctionBegin; 

  /* input checks */
  if (numdm <= 0) PetscFunctionReturn(0);
  ierr = DMGetCoordinateDim(dmlist[0],&dim_prev);CHKERRQ(ierr);
  for (i=0; i<numdm; i++) {
    ierr = DMGetType(dmlist[i],&dmtype);CHKERRQ(ierr); 
    ierr = PetscStrncmp(dmtype,DMPLEX,64,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(PetscObjectComm((PetscObject)dmlist[i]),PETSC_ERR_ARG_WRONG,"Wrong DM Object can only be DMPlex"); 
    ierr = DMGetCoordinateDim(dmlist[i],&dim);CHKERRQ(ierr);
    if (dim_prev != dim) SETERRQ(PetscObjectComm((PetscObject)dmlist[i]),PETSC_ERR_ARG_WRONG,"All Input DM objects must have the same Geometric Dimension");
  }

  /* Acquire maximum depth size across all dms and maximum topologial dimension chartsize */ 
  depth = 0; 
  dim_top = 0; 
  chartsize = 0; 
  for(i=0; i<numdm; i++){
    ierr = DMPlexGetDepth(dmlist[i],&depth_temp);CHKERRQ(ierr);
    if (depth < depth_temp) depth = depth_temp; 
    ierr = DMGetDimension(dmlist[i],&dim_top_temp);CHKERRQ(ierr);
    if (dim_top < dim_top_temp) dim_top = dim_top_temp;
    ierr = DMPlexGetChart(dmlist[i],&pStart,&pEnd);CHKERRQ(ierr);
    chartsize += (pEnd-pStart);
  }
  
  ierr = PetscMalloc1(chartsize,&coneSize_g);CHKERRQ(ierr);
  ierr = PetscCalloc1(depth+1,&numpoints_g);CHKERRQ(ierr);
  /* set up the stratum offset section */
  ierr = PetscSectionCreate(comm,&offsets);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(offsets, depth+1);CHKERRQ(ierr); /* one field per stratum */
  ierr = PetscSectionSetChart(offsets,0,numdm);CHKERRQ(ierr);
  for (j=0; j<=depth; j++) {
    ierr = PetscSectionSetFieldComponents(offsets, j, 1);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fieldname,64,"Stratum Depth %D",j);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(offsets,j,fieldname);CHKERRQ(ierr);
  }
  /* Iterate through the meshes and compute the number of points at each stratum */

  for (i=0; i<numdm; i++) {
    ierr = DMPlexGetDepth(dmlist[i],&depth_temp);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(offsets,i,depth_temp+1);CHKERRQ(ierr);
    for(stratum=0;stratum <= depth_temp; stratum++) {
      ierr = PetscSectionSetFieldDof(offsets,i,stratum,1);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(dmlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
      /* manually specify the section offset information, as the domain chart is not the same 
         as the range chart, and is not an onto mapbrping */
      ierr = PetscSectionSetFieldOffset(offsets,i,stratum,numpoints_g[stratum]-pStart); 
      numpoints_g[stratum] += (pEnd-pStart);
    }
  }
  /* Now we have the offset information for the input dm stratum into the new dm stratum */
  
  /* Create the cone size information */
  totalconesize = 0;
  for (i=0; i<numdm; i++) {
    ierr = DMPlexGetDepth(dmlist[i],&depth_temp);CHKERRQ(ierr);
    for(stratum=0;stratum <= depth_temp; stratum++) {
      ierr = DMPlexGetDepthStratum(dmlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
      prevtotal=0;
      for(j=0; j<stratum; j++) prevtotal += numpoints_g[j]; 
      ierr = PetscSectionGetFieldOffset(offsets,i,stratum,&off);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldOffset(offsets,i,stratum,off+prevtotal);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(offsets,i,stratum,&off);CHKERRQ(ierr);
      for(p=pStart; p<pEnd; p++) {
        ierr = DMPlexGetConeSize(dmlist[i],p,&coneSize);CHKERRQ(ierr);
        coneSize_g[p+off] = coneSize;
        totalconesize += coneSize;
      }
    }
  }
  /* create the cone and cone orientations */
  
  ierr = PetscMalloc2(totalconesize,&cones_g,totalconesize,&coneOrientations_g);CHKERRQ(ierr);
  k=0;
  for(stratum=0;stratum <= depth; stratum++) {
    for(i=0; i<numdm; i++){
      ierr = DMPlexGetDepth(dmlist[i],&depth_temp);CHKERRQ(ierr);
      if (stratum <= depth_temp) {
        ierr = DMPlexGetDepthStratum(dmlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
        if (stratum > 0) { /* stratum = 0 doesn't matter as the cones for stratum = 0 are empty */
          ierr = PetscSectionGetFieldOffset(offsets,i,stratum-1,&off);CHKERRQ(ierr);
        } 
        for(p=pStart; p<pEnd; p++) {
          ierr = DMPlexGetCone(dmlist[i],p,&cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dmlist[i],p,&coneOrientation);CHKERRQ(ierr);
          ierr = DMPlexGetConeSize(dmlist[i],p,&coneSize);CHKERRQ(ierr);
          for(j=0; j<coneSize; j++) {
            coneOrientations_g[k] = coneOrientation[j];
            cones_g[k++] = cone[j]+off; /* account for the offset in the cone stratum (stratum -1) */
          }
        }
      }
    }
  }
  /* Hack to make geometry work. I associate a a zero vector for the geometry field, in order the have all the 
  sections and etc built automatically. To be redone when I am more skilled */

  /* In theory we have everything ready to create the new global dm */
  ierr = DMPlexCreate(comm,&dm_sum);CHKERRQ(ierr);
  ierr = DMSetDimension(dm_sum,dim_top);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm_sum,dim);CHKERRQ(ierr);

  ierr = PetscCalloc1(numpoints_g[0]*dim,&vertexcoords);CHKERRQ(ierr);

  ierr = DMPlexCreateFromDAG(dm_sum,depth,numpoints_g,coneSize_g,cones_g,coneOrientations_g,vertexcoords);CHKERRQ(ierr);
  ierr = PetscFree(numpoints_g);CHKERRQ(ierr);
  ierr = PetscFree(coneSize_g);CHKERRQ(ierr);
  ierr = PetscFree2(cones_g,coneOrientations_g);CHKERRQ(ierr);
  
  /* Now we map the coordinates ... somehow */

  *dmsum = dm_sum;
  *stratumoffsets = offsets;
  PetscFunctionReturn(0);  
}

PetscErrorCode DGNetworkCreateNetworkDMPlex(DGNetwork dgnet,const PetscInt edgelist[],PetscInt edgelistsize,DM *dmsum,PetscSection *stratumoffset) {
  PetscErrorCode ierr; 
  PetscInt       i=0,e,eStart,eEnd;
  DM             *dmlist, network = dgnet->network;
  EdgeFE         edgefe;

  PetscFunctionBegin; 

  if (edgelist == NULL) { /* Assume the entire network is used */
    ierr = DMNetworkGetEdgeRange(network,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(eEnd-eStart,&dmlist);CHKERRQ(ierr);
    for (e=eStart; e<eEnd; e++) {
      ierr = DMNetworkGetComponent(network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
      dmlist[i++] = edgefe->dm; 
    }
    ierr = DMPlexAdd_Disconnected(dmlist,i,dmsum,stratumoffset);CHKERRQ(ierr);
    ierr = PetscFree(dmlist);CHKERRQ(ierr);
  } else {
      /* TODO */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DGNetworkCreateNetworkDMPlex_3D(DGNetwork dgnet,const PetscInt edgelist[],PetscInt edgelistsize,DM *dmsum,PetscSection *stratumoffset,DM **dm_list,PetscInt *numdm) {
  PetscErrorCode ierr; 
  PetscInt       i=0,j,e,eStart,eEnd,cStart,cEnd,dim,dE,pStart,pEnd,dof,p,off,off_g,off_stratum,secStart,secEnd,depth,stratum;
  DM             *dmlist, network = dgnet->network,cdm;
  EdgeFE         edgefe;
  PetscSection   coordsec,coordsec_g; 
  PetscBool      simplex; 
  PetscFE        fe;
  DMPolytopeType ct;
  Vec            Coord_g,Coord; 
  PetscReal      *coord_g,*coord; 

  PetscFunctionBegin; 

  if (edgelist == NULL) { /* Assume the entire network is used */
    ierr = DMNetworkGetEdgeRange(network,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(eEnd-eStart,&dmlist);CHKERRQ(ierr);
    for (e=eStart; e<eEnd; e++) {
      ierr = DMNetworkGetComponent(network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
      PetscInt faces[3]={cEnd-cStart,1,1}; 
      ierr = DMPlexCreateBoxMesh(PETSC_COMM_SELF, 3, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dmlist[i]);CHKERRQ(ierr);
      ierr = DGNetworkCreateViewDM2(dmlist[i]);CHKERRQ(ierr);
      if (e ==eStart){
        ierr = DMPlexRemapGeometry(dmlist[i++],0,f0_circle_r);CHKERRQ(ierr);
      } else if(e==eStart+1) {
          ierr = DMPlexRemapGeometry(dmlist[i++],0,f0_circle_t);CHKERRQ(ierr);
      } else if(e==eStart+2) {
          ierr = DMPlexRemapGeometry(dmlist[i++],0,f0_circle_l);CHKERRQ(ierr);
      } else {
          ierr = DMPlexRemapGeometry(dmlist[i++],0,f0_circle_b);CHKERRQ(ierr);
      }
    }
    *numdm = i; 
    ierr = DMPlexAdd_Disconnected(dmlist,*numdm,dmsum,stratumoffset);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(*dmsum, &cdm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dmsum, &dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(*dmsum, &dE);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(*dmsum, cStart, &ct);CHKERRQ(ierr);
    simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex,3,PETSC_DECIDE, &fe);CHKERRQ(ierr);
    ierr = DMProjectCoordinates(*dmsum, fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(*dmsum,&coordsec_g);CHKERRQ(ierr); 
    ierr = DMGetCoordinates(*dmsum,&Coord_g);CHKERRQ(ierr);
    ierr = VecGetArray(Coord_g,&coord_g);CHKERRQ(ierr);
    /* Now map the coordinate data */
    for(i=0; i<*numdm; i++) {
      ierr = DMGetCoordinates(dmlist[i],&Coord);CHKERRQ(ierr);
      ierr = VecGetArray(Coord,&coord);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(dmlist[i],&coordsec);CHKERRQ(ierr);
      
      ierr = PetscSectionGetChart(coordsec,&secStart,&secEnd);CHKERRQ(ierr);
      /* Iterate through the stratum */
      ierr = DMPlexGetDepth(dmlist[i],&depth);CHKERRQ(ierr);
      for (stratum = 0; stratum <= depth; stratum++){
        ierr = DMPlexGetDepthStratum(dmlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(*stratumoffset,i,stratum,&off_stratum);CHKERRQ(ierr);
        /* there is a better way of doing this ... for later */
        for (p=pStart;p<pEnd&&p<secEnd;p++) {
          if( p >= secStart) {
            ierr = PetscSectionGetFieldOffset(coordsec,p,0,&off);CHKERRQ(ierr); /* domain offset */
            ierr = PetscSectionGetFieldDof(coordsec,p,0,&dof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(coordsec_g,p+off_stratum,0,&off_g);CHKERRQ(ierr); /*range offset */
            for (j=0; j<dof;j++){
              coord_g[off_g+j] = coord[off+j];
            }
          }
        }
      }
      ierr = VecRestoreArray(Coord,&coord);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(Coord_g,&coord_g);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dmsum,Coord_g);CHKERRQ(ierr);
    
    /* in theory the coordinates are now mapped correctly ... we shall see */
    *dm_list = dmlist; 
  } else {
      /* TODO */
  }
  PetscFunctionReturn(0);
}


/* More viewer stuff */ 

/* Here we assume we are viewing the entire DGNetwork vector */
PetscErrorCode DGNetworkMonitor_3D_NET_g2l_internal(PetscObject V,PetscInt nfields,PetscObject Vfield[],void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis) ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe; 
  PetscInt                   copy,c,cStart,cEnd,field,tab,dof=dgnet->physics.dof,i,fieldoff,deg,ndegree,e,eStart,eEnd,cCount,off_e; 
  PetscErrorCode             ierr; 
  PetscSection               section;
  const PetscReal            *v; 
  PetscReal                  *vwork; 
  PetscInt                   Dim = 3; 
  PetscFunctionBegin;
  ierr = VecGetArrayRead((Vec)V,&v);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(network,&eStart,&eEnd);CHKERRQ(ierr);
  cCount = 0;
  for(e=eStart;e<eEnd;e++) {
    ierr = DMNetworkGetComponent(network,e,FVEDGE,NULL,(void**)&edgefe,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(network,e,FVEDGE,&off_e);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(edgefe->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMGetSection(edgefe->dm,&section);CHKERRQ(ierr);
    /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
    for(field=0; field<dof; field++) {
      tab = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      i = cCount * PetscPowInt(ndegree+1,Dim);
      ierr = VecGetArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr); 
      for(c=cStart; c<cEnd; c++) {
        ierr = PetscSectionGetFieldOffset(section,c,field,&fieldoff);CHKERRQ(ierr);
        for(deg=0; deg<=ndegree; deg++) {
          vwork[i] =  evalviewpt_internal(dgnet,field,deg, v+fieldoff+off_e);
          for(copy=1; copy<(ndegree+1)*(ndegree+1); copy++) {
            vwork[i+copy*(ndegree+1)] = vwork[i];
          }
          i++; 
        }
        i+=(ndegree+1)*((ndegree+1)*(ndegree+1)-1);
      }
      ierr = VecRestoreArray((Vec)Vfield[field],&vwork);CHKERRQ(ierr);
    }
    cCount += cEnd-cStart;
  }

  ierr = VecRestoreArrayRead((Vec)V,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode DGNetworkMonitorAdd_Glvis_3D_NET(DGNetworkMonitor_Glvis monitor,const char hostname[],PetscViewerGLVisType type)
{
  PetscErrorCode       ierr;
  PetscMPIInt          rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt             viewsize,field,cStart,cEnd,tab,Dim = 3;
  DGNetwork            dgnet=monitor->dgnet;
  PetscInt             dof=dgnet->physics.dof;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(monitor->comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(monitor->comm, &size);CHKERRMPI(ierr);

  ierr = PetscMalloc1(1, &node);CHKERRQ(ierr);
  ierr = PetscMalloc3(dof,&node->dim,dof,&node->v_work,dof,&node->fec_type);CHKERRQ(ierr);

  ierr = PetscViewerGLVisOpen(monitor->comm,type,hostname,PETSC_DECIDE,&node->viewer);CHKERRQ(ierr);
  ierr = DGNetworkCreateNetworkDMPlex_3D(dgnet,NULL,0,&node->viewdm,&node->stratumoffset,&node->dmlist,&node->numdm);CHKERRQ(ierr);
  /* Create the network mesh */ 
  ierr = DMPlexGetHeightStratum(node->viewdm,0,&cStart,&cEnd);CHKERRQ(ierr);
  /* make the work vector for each field */
  for(field=0; field<dof; field++) {
    /* Setup vector storage for drawing. */
    tab        = dgnet->fieldtotab[field];
    viewsize   = (cEnd-cStart)*PetscPowInt((dgnet->taborder[tab]+1),Dim); /* number of variables for the given field */
    ierr       = VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field]));CHKERRQ(ierr);
    ierr       = PetscObjectCompose((PetscObject)node->v_work[field],"__PETSc_dm",(PetscObject)node->viewdm);CHKERRQ(ierr); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    ierr       = PetscMalloc(64,&node->fec_type[field]);CHKERRQ(ierr);
    ierr       = PetscSNPrintf(node->fec_type[field],64,"FiniteElementCollection: L2_T4_%iD_P%i",Dim,dgnet->taborder[tab]);CHKERRQ(ierr); 
    node->dim[field] = Dim; 
  }

  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  monitor->firstnode = node;

  ierr = PetscViewerGLVisSetFields(node->viewer,dof,(const char**)node->fec_type,node->dim,DGNetworkMonitor_3D_NET_g2l_internal,(PetscObject*)node->v_work,(void*)node,DGNetworkMonitor_destroyctx_internal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DGNetworkMonitorView_Glvis_NET(DGNetworkMonitor_Glvis monitor,Vec x)
{
  PetscErrorCode      ierr;
  DGNetworkMonitorList_Glvis node;

  PetscFunctionBegin;
  for (node = monitor->firstnode; node; node = node->next) {
    ierr = VecView(x, node->viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}