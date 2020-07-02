#include "fvnet.h"


PetscErrorCode FVNetworkCreate(MPI_Comm comm,PetscInt initial,FVNetwork *fvnet_ptr,PetscInt Mx)
{
  PetscErrorCode ierr;
  PetscInt       nfvedge;
  PetscMPIInt    rank;
  FVNetwork      fvnet=NULL;
  PetscInt       i,numVertices,numEdges,*vtype;
  PetscInt       *edgelist;
  Junction       junctions=NULL;
  FVEdge         fvedges=NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscCalloc1(1,&fvnet);CHKERRQ(ierr); /* hmm...*/
  fvnet->comm = comm;
  *fvnet_ptr  = fvnet;
  fvnet->nnodes_loc = 0;

  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;

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
      junctions[1].type = JUNCTION;
      junctions[2].type = JUNCTION;
      junctions[3].type = OUTFLOW;

    for(i=0; i<numVertices; i++){
        junctions[i].x = i*1.0/3.0; 
    }

      /* Edge */ 
      fvedges[0].nnodes = Mx; 
      fvedges[1].nnodes = 2*Mx; 
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

  *fvnet_ptr      = fvnet;
  fvnet->nedge    = numEdges;
  fvnet->nvertex  = numVertices;
  fvnet->edgelist = edgelist;
  fvnet->junction = junctions;

  PetscFunctionReturn(0);
}

PetscErrorCode FVNetworkSetComponents(FVNetwork fvnet){
  PetscErrorCode    ierr;
  Junction          junctions; 
  PetscInt          KeyEdge,KeyJunction,KeyFlux;
  PetscInt          i,e,v,eStart,eEnd,vStart,vEnd,key,dof = fvnet->physics.dof;
  PetscInt          nnodes_edge,nnodes_vertex,nedges_tmp; 
  PetscInt          *edgelist = NULL,*edgelists[1];
  DM                networkdm = fvnet->network;
  PetscInt          nedges,nvertices; /* local num of edges and vertices */
  FVEdge            fvedge;
  MPI_Comm          comm = fvnet->comm;
  PetscMPIInt       size,rank;

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

  /*Note that the order in which these are added determine the compnum, which is used 
    throughout in DMNetwork for data access. Do not swap the order! */
  ierr = DMNetworkRegisterComponent(networkdm,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"fvedgestruct",sizeof(struct _p_FVEdge),&KeyEdge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"fluxstruct",0,&KeyFlux);CHKERRQ(ierr);

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
    ierr = DMNetworkGetSupportingEdges(networkdm,v,&nedges_tmp,NULL);CHKERRQ(ierr); /* will NULL work? */
    nnodes_vertex = dof*(fvnet->bufferwidth+fvnet->stencilwidth)*nedges_tmp;
    ierr = DMNetworkSetComponentNumVariables(networkdm,v,JUNCTION,nnodes_vertex);CHKERRQ(ierr);
    /* Add data structure for moving the vertex fluxes around. Also used to store 
       vertex reconstruction information. A working vector used for data storage 
       and message passing.*/
    ierr = DMNetworkAddComponent(networkdm,v,KeyFlux,NULL);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(networkdm,v,FLUX,nedges_tmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

FVNetworkSetupPhysics(FVNetwork fvnet){
PetscErrorCode ierr; 
PetscInt       e,i,j,nv,ne,vtx,edges;
PetscInt       eStart,eEnd;
PetscInt       vfrom,vto; 
const PetscInt *cone; 
FVEdge         fvedge; 
Junction       junction; 

ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);

  for (e = eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);

    ierr = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    
    ierr = DMNetworkGetComponent(fvnet->network,vfrom,,NULL,(void**)&junction);CHKERRQ(ierr);  
  }
}


PetscErrorCode FVRHSFunction_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hxf,hxs,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points                                     */
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points                              */
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points       */
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);                                   /* F is the right hand side function corresponds to center points */

  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);                  /* contains ghost points                                           */

  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hxs = hxs;
    info.hxf = hxf;
    (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
      slope[i*dof+j] = tmp;
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i < sf) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
  PetscFunctionReturn(0);
}