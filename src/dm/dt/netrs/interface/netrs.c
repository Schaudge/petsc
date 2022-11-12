#include <petscriemannsolver.h>       
#include <petscviewer.h>
#include <petscdraw.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petsc/private/netrsimpl.h>
#include <petscnetrs.h>
#include <petscdm.h>
#include <petscsf.h>

#include <petsc/private/riemannsolverimpl.h> /* to be removed after adding fluxfunction class */

/*@
   NetRSSetUp - Sets up the internal data structures for the later use of a NetRS. 

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from RiemanSolverCreate()  


   Level: advanced

.seealso: NetRSCreate(), NetRSSetFlux()
@*/
PetscErrorCode  NetRSSetUp(NetRS rs)
{
  DM             network; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  if (rs->setupcalled) PetscFunctionReturn(0); 

  /* find the list of vertex degrees in the local network. Used to generate 
  the list of preallocated objects needed by the solvers */
  PetscCall(NetRSGetNetwork(rs,&network)); 
  /* can assume that the network exists from here on */
  if (rs->ops->setup) {
    PetscCall((*rs->ops->setup)(rs));
  }
  rs->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   NetRSReset - Resets a NetRS context and removes any allocated internal petsc objects

   Collective on NetRS

   Input Parameter:
.  rs - the RiemanmSolver context obtained from NetRSCreate()

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode  NetRSReset(NetRS rs)
{
  PetscInt i,numnetrp; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscTryTypeMethod(rs,reset);
  PetscCall(NetRSResetVectorSpace(rs)); 
  
  PetscCall(DMLabelGetNumValues(rs->subgraphs,&numnetrp)); 
  for (i=0; i<numnetrp; i++) {
    PetscCall(NetRPDestroy(&rs->netrp[i])); 
    PetscCall(PetscHSetIDestroy(&rs->vertexdegrees[i])); 
  }

  PetscCall(PetscHSetIClear(rs->vertexdegrees_total)); 
  PetscCall(DMDestroy(&rs->network)); 
  PetscCall(DMLabelReset(rs->subgraphs)); 
  PetscCall(DMLabelReset(rs->VertexDeg_shared)); 

  rs->vertexdeg_shared_cached = PETSC_FALSE; 
  rs->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
/*@
   NetRSResetVectorSpace - Resets a NetRS vector space context and removes any related internal petsc objects. 
   Undos the NetRSSetupVectorSpace function. 

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from NetRSCreate()

   Level: intermediate

.seealso: NetRSCreate(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode NetRSResetVectorSpace(NetRS rs)
{
  PetscInt i,numnetrp; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscTryTypeMethod(rs,resetvecspace);
  PetscCall(DMLabelGetNumValues(rs->subgraphs,&numnetrp)); 
  for(i=0; i<numnetrp; i++) PetscCall(PetscHSetIClear(rs->vertexdegrees[i])); 
  PetscCall(VecDestroy(&rs->totalFlux)); 
  PetscCall(VecDestroy(&rs->totalU));
  rs->setupvectorspace = PETSC_FALSE;
  PetscFunctionReturn(0); 
}

/*@
   NetRSDestroy - Destroys the NetRS context that was created
   with NetRSCreate().

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from NetRSCreate()

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp()
@*/
PetscErrorCode  NetRSDestroy(NetRS *rs)
{
  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*rs,NETRS_CLASSID,1);
  if (--((PetscObject)(*rs))->refct > 0) {*rs = NULL; PetscFunctionReturn(0);}
  if ((*rs)->ops->destroy) PetscCall((*(*rs)->ops->destroy)((*rs)));
  PetscCall(NetRSReset(*rs));
  PetscCall(DMLabelDestroy(&(*rs)->VertexDeg_shared));
  PetscCall(DMLabelDestroy(&(*rs)->subgraphs)); 
  PetscCall(PetscHeaderDestroy(rs));
  PetscFunctionReturn(0);
}

/*
  NetRSDuplicate - Create a new netrs of the same type as the original with the same settings. Still requires a call to setup after this call 
  as the intended use is to set the parameters for a "master" netrs duplicate it to other NetRS and change the types of the new netrs to the desired types. 
  This is the quick way of getting multiple netrs of different types for the same physics. 
*/

PetscErrorCode NetRSDuplicate(NetRS netrs,NetRS *newnetrs)
{
  MPI_Comm       comm;
  NetRS          netrs_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs,NETRS_CLASSID,1);
  PetscValidPointer(newnetrs,2);
  PetscValidType(netrs,1);

  PetscCall(PetscObjectGetComm((PetscObject)netrs,&comm));
  PetscCall(NetRSCreate(comm,&netrs_new)); 
  /* copy over the parameters and physics from netrs to newnetrs */ 

  /* topology */
  PetscCall(DMClone(netrs->network,&netrs_new->network));
  /* physics*/
  netrs_new->user      = netrs->user; 
  PetscCall(NetRSSetFlux(netrs_new,netrs->rs));
  *newnetrs = netrs_new;  
  PetscFunctionReturn(0);
}

/*@
   NetRSSetApplicationContext - Sets an optional user-defined context for
   the NetRS.

   Logically Collective on NetRS

   Input Parameters:
+  rs - the NetRS context obtained from NetRSCreate()
-  usrP - optional user context

   Level: intermediate

.seealso: NetRSGetApplicationContext()
@*/
PetscErrorCode  NetRSSetApplicationContext(NetRS rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  rs->user = usrP;
  PetscFunctionReturn(0);
}

/*@
   NetRSSetNetwork - Set the DMNetwork defining the topology of the network 
   Riemann problem(s). NetRS does not take ownership of the DM and it must be destroyed by
   the caller
   
   Collective on NetRS

   Input Parameters:
+  rs      - the NetRS context obtained from NetRSCreate()
-  network - The DMNetwork network.

   Level: beginner

.seealso: 
@*/

PetscErrorCode NetRSSetNetwork(NetRS rs, DM network)
{
  DM networkclone; 
  PetscFunctionBegin; 
  PetscValidHeaderSpecificType(network,DM_CLASSID,2,DMNETWORK);
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);

  PetscCheck(!rs->setupcalled,PetscObjectComm((PetscObject)rs),PETSC_ERR_ARG_WRONGSTATE,"NetRSSetUp() has already been called, the Network cannot be changed. Call NetRSReset() if you need change the network. ");
  if (rs->network) PetscCall(DMDestroy(&rs->network));
  PetscCall(DMClone(network,&networkclone)); 
  rs->network = networkclone; 
  PetscFunctionReturn(0);
}


/*@
   NetRSGetNetwork - Get the DMNetwork defining the topology of the network Riemann problem(s).
   Currently an internal only function to prevent users shooting themselves in the foot. 

   Collective on NetRS

   Input Parameters:
+  rs      - the NetRS context obtained from NetRSCreate()
-  network - The DMNetwork network.

   Level: developer

.seealso: `NetRSSetNetwork()`
@*/

PetscErrorCode NetRSGetNetwork(NetRS rs, DM *network)
{
  PetscFunctionBegin; 
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscCheck(rs->network,PetscObjectComm((PetscObject)rs),PETSC_ERR_ARG_WRONGSTATE,"NetRS has no network. One must be set by NetRSSetNetwork()");
  *network = rs->network; 
  PetscFunctionReturn(0);
}

/*@
    NetRSGetApplicationContext - Gets the user-defined context for the
    NetRS

    Not Collective

    Input Parameter:
.   rs - the NetRS context obtained from NetRSCreate()

    Output Parameter:
.   usrP - user context

    Level: intermediate

.seealso: NetRSSetApplicationContext()
@*/
PetscErrorCode  NetRSGetApplicationContext(NetRS rs,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  *(void**)usrP = rs->user;
  PetscFunctionReturn(0);
}

/*@
  NetRSSetFromOptions - sets parameters in a NetRS from the options database

  Collective on NetRS

  Input Parameter:
. rs - the NetRS object to set options for

  Options Database:

  Level: intermediate

.seealso 
@*/
PetscErrorCode NetRSSetFromOptions(NetRS rs)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* Type Option */
  if (!((PetscObject) rs)->type_name) {
    defaultType = NETRSBASIC;
  } else {
    defaultType = ((PetscObject) rs)->type_name;
  }
  if (!NetRSRegisterAllCalled) PetscCall(NetRSRegisterAll());

  PetscObjectOptionsBegin((PetscObject) rs);
  PetscCall(PetscOptionsFList("-netrs_type", "NetRS", "NetRSSetType", NetRSList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(NetRSSetType(rs, name));
  } else if (!((PetscObject) rs)->type_name) {
    PetscCall(NetRSSetType(rs, defaultType));
  }
  /* parameter selection */
  /*
  PetscCall(PetscOptionsReal("-netrs_finetol","Tolerance to swap to fine netrs solver","",rs->finetol,&rs->finetol,NULL));
  PetscCall(PetscOptionsBool("-netrs_use_estimator","Use error estimator if available","",rs->useestimator,&rs->useestimator,NULL));
  PetscCall(PetscOptionsBool("-netrs_use_adaptivity","Use adaptivity if available","",rs->useadaptivity,&rs->useadaptivity,NULL));
  PetscCall(PetscOptionsFList("-netrs_fine", "Fine NetRS to use with adaptivity", "NetRSSetType", NetRSList, rs->finetype, name, 256, &flg));
  if (flg) {rs->finetype = name;}
  */


  /* handle implementation specific options */
  if (rs->ops->setfromoptions) {
    PetscCall((*rs->ops->setfromoptions)(PetscOptionsObject,rs));
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject) rs,PetscOptionsObject));
  PetscOptionsEnd();
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(0);
}

/*@C
    NetRSView - Prints the NetRS data structure.

    Collective on NetRS. 

    For now I use this to print error and adaptivity information to file. 

    Input Parameters:
+   rs - the NetRS context obtained from NetRSCreate()
-   viewer - visualization context

    Options Database Key:
   TODO: 
    Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  NetRSView(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

PetscErrorCode NetRSSetFlux(NetRS nrs, RiemannSolver flux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nrs,NETRS_CLASSID,1);
  PetscValidHeaderSpecific(flux,RIEMANNSOLVER_CLASSID,1);
  if(nrs->rs) PetscCall(RiemannSolverDestroy(&nrs->rs)); 
  PetscCall(PetscObjectReference((PetscObject)flux)); 
  nrs->rs = flux; 
  PetscFunctionReturn(0);
}

PetscErrorCode NetRSGetFlux(NetRS rs, RiemannSolver *flux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1);
  PetscValidHeaderSpecific(*flux,RIEMANNSOLVER_CLASSID,1); 
  if(rs->rs) *flux = rs->rs; 
  PetscFunctionReturn(0);
}

/* internal function for adding the NetRP to the NetRS. This assumes that the NetRP has not already 
been added. 
*/
static PetscErrorCode NetRSAddNetRP(NetRS rs, NetRP rp)
{
  PetscInt   numlabelvalues; 
  NetRP      *netrp_new;
  PetscHSetI *vertexdegs_new;  

  PetscFunctionBegin; 
  PetscCall(DMLabelGetNumValues(rs->subgraphs,&numlabelvalues)); /* current number of netrp stored */
  PetscCall(DMLabelAddStratum(rs->subgraphs,numlabelvalues));
  PetscCall(PetscHMapNetRPISet(rs->netrphmap,rp,numlabelvalues));

  /* create new memory and move */
  PetscCall(PetscMalloc1(numlabelvalues+1,&netrp_new)); 
  PetscCall(PetscMalloc1(numlabelvalues+1,&vertexdegs_new)); 
  PetscCall(PetscArraycpy(netrp_new,rs->netrp,numlabelvalues));
  PetscCall(PetscArraycpy(vertexdegs_new,rs->vertexdegrees,numlabelvalues));

  PetscCall(PetscFree(rs->netrp)); 
  PetscCall(PetscFree(rs->vertexdegrees)); 
  rs->vertexdegrees = vertexdegs_new; 
  rs->netrp = netrp_new; 

  PetscCall(PetscHSetICreate(&rs->vertexdegrees[numlabelvalues])); 
  rs->netrp[numlabelvalues] = rp; 
  PetscCall(PetscObjectReference((PetscObject)rp)); 
  PetscFunctionReturn(0); 
}


/* uses local vertex numbering */
PetscErrorCode NetRSAddNetRPatVertex(NetRS rs,PetscInt v, NetRP rp)
{
  DM network; 
  PetscBool flg;
  PetscInt  vStart,vEnd, defaultval,pval,index; 
  
  PetscFunctionBegin; 
  PetscCall(NetRSSetUp(rs)); 
  PetscCall(NetRSGetNetwork(rs,&network));


  /* Check if this NetRP has already been added to the network */
  PetscCall(PetscHMapNetRPIHas(rs->netrphmap,rp,&flg)); 
  if (!flg) PetscCall(NetRSAddNetRP(rs,rp));

  /* check if vertex v belongs to the DMNetwork */
  PetscCall(DMNetworkGetVertexRange(rs->network,&vStart,&vEnd));
  PetscCheck(vStart<=v && v<vEnd,PetscObjectComm((PetscObject)rs),PETSC_ERR_USER_INPUT,"Input Vertex %"PetscInt_FMT" is not a vertex on the DMNetwork attached to NetRS, which has range %"PetscInt_FMT " to %" PetscInt_FMT,v,vStart,vEnd);
  /* Check if vertex v has any values associated with it. NetRS assumes that there is only one NetRP
  for each vertex.
  
  Perhaps should create a disjoint label implementation that takes care of this automatically? Would definitely be more performant, 
  and disjoint partitions are a useful thing in general. 
  */
  PetscCall(DMLabelGetDefaultValue(rs->subgraphs,&defaultval));
  PetscCall(DMLabelGetValue(rs->subgraphs,v,&pval));
  PetscCheck(pval==defaultval,PetscObjectComm((PetscObject)rs),PETSC_ERR_USER_INPUT,"NetRS assumes a single NetRP for each vertex. Inputted vertex %"PetscInt_FMT" already has an assigned NetRP. TODO Print the name of the NetRP already assigned here",v); 

  /* Add vertex v to the label value associated with the NetRP */
  PetscCall(PetscHMapNetRPIGet(rs->netrphmap,rp,&index)); 
  PetscCall(DMLabelSetValue(rs->subgraphs,v,index));
  PetscFunctionReturn(0); 
}


/* TODO: Migrate this functionality to DMNetwork itself */
PetscErrorCode DMNetworkCacheVertexDegrees(NetRS rs, DM network) 
{  
  PetscInt     v, i,vStart,vEnd,nroots,nleaves,nedges; 
  PetscSF      sf;
  DM           plex; 
  const PetscInt *ilocal; 
  PetscInt     *rootdata,*leafdata;
  PetscMPIInt              size,rank;
  MPI_Comm                 comm;


  PetscFunctionBegin; 
  PetscValidHeaderSpecificType(network,DM_CLASSID,2,DMNETWORK);
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1); 

  if (rs->vertexdeg_shared_cached) PetscFunctionReturn(0); /* already been cached */
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (size == 1) 
  {
    rs->vertexdeg_shared_cached = PETSC_TRUE; 
    PetscFunctionReturn(0);
  }
  
  /* Pull out the pointsf used by the underlying DMPlex for the distributed network */
  PetscCall(DMNetworkGetPlex(network,&plex));
  PetscCall(DMGetPointSF(plex,&sf));

  /* SUPER IMPORTANT NOTE: THE SF FROM PLEX ASSUMES THE SAME ARRAY SIZE FOR ROOT DATA AND LEAF DATA, WITH 
  ILOCAL HOLDING THE ACTUAL LEAF ENTRIES IN THAT ARRAY. */
  PetscCall(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL)); 
  PetscCheck(ilocal!= NULL || nleaves==0,comm,PETSC_ERR_SUP,"Currently assumes a plex format for the DM PointSF, where leafdata has the same size as rootdata, and ilocal holds offsets. Should not have leaf data in continguous storage");

  PetscCall(PetscCalloc2(nroots,&rootdata,nroots,&leafdata)); 
  

  /* Fill the leaf data with the local vertex degrees */
  PetscCall(DMNetworkGetVertexRange(network,&vStart,&vEnd));
  for(i=0; i<nleaves; i++)
  {
    if(ilocal[i]>=vEnd || ilocal[i]<vStart) break; 
    PetscCall(DMNetworkGetSupportingEdges(network,ilocal[i],&leafdata[ilocal[i]],NULL)); 
  }


  
  /* reduce degree data from leaves to root. This gives the correct vertex degree on the
  distributed graph */
  PetscCall(PetscSFReduceBegin(sf,MPIU_INT,leafdata,rootdata,MPIU_SUM)); 
  PetscCall(PetscSFReduceEnd(sf,MPIU_INT,leafdata,rootdata,MPIU_SUM)); 

  /* any nonzero entry in rootdata then has leaves, these are added to the lable as 
     shared vertices. The local vertex degree are then added to the rootdata to create the 
     correct vertex degree. */

  for(v=vStart; v<vEnd; v++) {
    if(!rootdata[v]) break; 
    PetscCall(DMNetworkGetSupportingEdges(network,v,&nedges,NULL)); 
    rootdata[v] += nedges; 
    PetscCall(DMLabelSetValue(rs->VertexDeg_shared,v,rootdata[v])); 
  }
  PetscCall(PetscSynchronizedFlush(comm,NULL));

  /* Rootdata contains the correct vertex degs, and these have been added to the labrl*/
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata,MPI_REPLACE)); 
  PetscCall(PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata,MPI_REPLACE));
  /*iterate through the leaf vertices and add their values to the label */
  for(i=0; i<nleaves; i++){
    if(ilocal[i]>=vEnd || ilocal[i]<vStart) break; 
    PetscCall(DMLabelSetValue(rs->VertexDeg_shared,ilocal[i],leafdata[ilocal[i]])); 
  }
  /* no new entries will be added, compute index for faster membership lookup */
  PetscCall(DMLabelComputeIndex(rs->VertexDeg_shared)); 
  PetscCall(PetscFree2(rootdata,leafdata)); 
  rs->vertexdeg_shared_cached = PETSC_TRUE; 
  PetscFunctionReturn(0); 
}

/* TODO: Duplicates a lot of code from DMNetworkCacheVertexDegrees. Should b
refactored along with that. */

/* 
Note, could use that data computed here to compute the stuff needed in 
DMNetworkCacheVertexDegrees for "Free". Only a single Bcast is needed instead of 
needing a reduce  + bcast. 
*/

 PetscErrorCode DMNetworkCreateLocalEdgeNumbering(NetRS rs, DM network)
{
  DM  plex; 
  PetscSF sf; 
  PetscSection rootsection;
  PetscInt    i,pStart,pEnd, v,p,vStart,vEnd,multirootsize,dof,off,v_off,vlocaldeg,nleaves,nroots;
  PetscInt     *leafdata,*multirootdata; 
  const PetscInt *rootdegree,*ilocal; 
  PetscMPIInt              size;
  MPI_Comm                 comm;


  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network,DM_CLASSID,2,DMNETWORK);
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1); 
  if(rs->vertex_shared_offset) PetscFunctionReturn(0); /* already created the map */
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) 
  {
    rs->vertexdeg_shared_cached = PETSC_TRUE; 
    PetscFunctionReturn(0);
  }

  PetscCall(PetscHMapICreate(&rs->vertex_shared_offset)); 
  /* Pull out the pointsf used by the underlying DMPlex for the distributed network */
  PetscCall(DMNetworkGetPlex(network,&plex));
  PetscCall(DMGetPointSF(plex,&sf));
  PetscCall(DMPlexGetChart(plex,&pStart,&pEnd)); 
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&rootsection)); /* used for organizing the multiroot data coming for sf gather */
  PetscCall(PetscSectionSetChart(rootsection,pStart,pEnd)); 
  PetscCall(PetscSFComputeDegreeBegin(sf,&rootdegree)); 
  PetscCall(PetscSFComputeDegreeEnd(sf,&rootdegree)); 
  for(p=pStart; p<pEnd; ++p) PetscCall(PetscSectionSetDof(rootsection,p,rootdegree[p-pStart]));
  PetscCall(PetscSectionSetUp(rootsection)); 
  /*allocate the multiroot data and leaf data */
  PetscCall(PetscSectionGetStorageSize(rootsection,&multirootsize)); 
  PetscCall(PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL));
  PetscCheck(ilocal!= NULL || nleaves==0,comm,PETSC_ERR_SUP,"Currently assumes a plex format for the DM PointSF, where leafdata has the same size as rootdata, and ilocal holds offsets. Should not have leaf data in continguous storage");
  PetscCall(PetscMalloc2(multirootsize,&multirootdata,nroots,&leafdata));

 /* Fill the leaf data with the local vertex degrees */
  PetscCall(DMNetworkGetVertexRange(network,&vStart,&vEnd));
  for(i=0; i<nleaves; i++){
    if(ilocal[i]>=vEnd || ilocal[i]<vStart) break; 
    PetscCall(DMNetworkGetSupportingEdges(network,ilocal[i],&leafdata[ilocal[i]],NULL)); 
  }
  /* Gather the local vertex degrees of each leaf to the root */
  PetscCall(PetscSFGatherBegin(sf,MPIU_INT,leafdata,multirootdata));
  PetscCall(PetscSFGatherEnd(sf,MPIU_INT,leafdata,multirootdata)); 

  /* Generate the shared vertex edge local ordering in place */
  for(v=vStart; v<vEnd; v++) {
    PetscCall(PetscSectionGetDof(rootsection,v,&dof)); 
    if(!dof) break; 
    PetscCall(PetscSectionGetOffset(rootsection,v,&off)); 
    v_off = 0; 
    for(i=off; i<dof+off; i++){
      vlocaldeg = multirootdata[i]; 
      multirootdata[i] = v_off; 
      v_off+=vlocaldeg; 
    }
    PetscCall(PetscHMapISet(rs->vertex_shared_offset,v,v_off)); 
  }
  /*now multirootdata contains all the offsets for the leaves of roots, 
  and the HMapI already contains the offsets for the roots. Scatter 
  The offsets back to leaves and then added leaf offsets to the HMapI */

  PetscCall(PetscSFScatterBegin(sf,MPIU_INT,multirootdata,leafdata)); 
  PetscCall(PetscSFScatterEnd(sf,MPIU_INT,multirootdata,leafdata)); 

  /*iterate and add to HMap*/
  for(i=0; i<nleaves; i++)
  {
    if(ilocal[i]>=vEnd || ilocal[i]<vStart) break; 
    PetscCall(PetscHMapISet(rs->vertex_shared_offset,ilocal[i],leafdata[ilocal[i]])); 
  }
  PetscCall(PetscFree2(multirootdata,leafdata)); 
  PetscCall(PetscSectionDestroy(&rootsection)); 
  PetscFunctionReturn(0); 
}




/*@
    DMNetworkComputeUniqueVertexDegrees_Local- Returns the unique set of vertex 
    degrees of the local DMNetwork graph. However this includes the edges attached to shared 
    vertices in the computation, hence the collective nature. Internal use only for now, particularly NetRS. 

    For example the network 

        P0         |       P1
    v0 --E1----v1  |     v1 --- E2 --- v2
                   | 

    Would return a set of {1,2} for both processors, as in the global graph deg(v1) = 2, even though 
    in the local graph it has deg 1. 

    Collective

    Input Parameter:
.   network - The setup DMNetwork 
.   marked  - A DMLabel which marks which vertices to consider for computing the vertex degree. Any entry included in the DMLabel
              will have its vertex degree added to the set. A NULL entry corresponds to computing on the full graph. 

    Output Parameter:
.    vertexdegrees - array of the vertex degrees sets, one for each subgraph induced by the labeled vertices. NULL if no label is passed in. 
.    totalvertexdegree - set of vertex degrees for the subgraph induced by the union of all marked vertex. This is the union of of all the vertex degree sets. 
    Level: developer

.seealso: 
@*/
PetscErrorCode DMNetworkComputeUniqueVertexDegrees_Local(NetRS rs,DM network,DMLabel marked, PetscHSetI *vertexdegrees, PetscHSetI totalvertexdegrees)
{
  PetscInt numsubgraphs,i,j,v,vStart,vEnd,vdeg,numpoints;  
  IS       values_is,point_is; 
  const PetscInt *values,*points; 
  PetscBool flg; 

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network,DM_CLASSID,1,DMNETWORK);
  PetscValidHeaderSpecific(rs,NETRS_CLASSID,1); 

  /* Clear Hash */
  PetscCall(PetscHSetIClear(totalvertexdegrees));
  /* Generate the vertex degree labeling */
  PetscCall(DMNetworkCacheVertexDegrees(rs,network)); 

  PetscCall(DMNetworkGetVertexRange(network,&vStart,&vEnd)); 
  /* generate the hash set for the entire local graph first */
  for(v=vStart; v<vEnd; v++){
    /* check if v is in the vertex degree labeling. That means it is shared among processors 
       and needs to use the value from the labeling */
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared,v,&flg)); 
    if(flg){ /*get vertex degree from label*/
       PetscCall(DMLabelGetValue(rs->VertexDeg_shared,v,&vdeg)); 
    } else { /* vertex is entirely local, so DMNetworkGetSupportingEdge returns correct vertex deg */
      PetscCall(DMNetworkGetSupportingEdges(network,v,&vdeg,NULL)); 
    }
    PetscCall(PetscHSetIAdd(totalvertexdegrees,vdeg)); 
  }
  
  /* repeat but iterate only through marked vertices */
  if (marked) { 
    PetscCheck(vertexdegrees,PetscObjectComm((PetscObject)network),PETSC_ERR_USER_INPUT,"If providing a label of marked vertices, an array of PetscHSetI, one for each value in the label must be provided");
    PetscCall(DMLabelGetNumValues(marked,&numsubgraphs));
    PetscCall(DMLabelGetValueIS(marked,&values_is)); 
    PetscCall(ISGetIndices(values_is,&values));

    for(i=0; i<numsubgraphs; i++) {
      PetscCall(PetscHSetIClear(vertexdegrees[i]));
      PetscCall(DMLabelGetStratumIS(marked,values[i],&point_is)); 
      if(point_is == NULL) break; 
      PetscCall(ISGetSize(point_is,&numpoints)); 
      PetscCall(ISGetIndices(point_is,&points)); 
      for(j=0; j<numpoints;j++){
        v = points[j];
        PetscCall(DMLabelHasPoint(rs->VertexDeg_shared,v,&flg)); 
        if(flg){ /*get vertex degree from label*/
          PetscCall(DMLabelGetValue(rs->VertexDeg_shared,v,&vdeg)); 
        } else { /* vertex is entirely local, so DMNetworkGetSupportingEdge returns correct vertex deg */
          PetscCall(DMNetworkGetSupportingEdges(network,v,&vdeg,NULL)); 
        }
        PetscCall(PetscHSetIAdd(vertexdegrees[i],vdeg)); 
      }
      PetscCall(ISRestoreIndices(point_is,&points)); 
      PetscCall(ISDestroy(&point_is)); 
    }
    PetscCall(ISRestoreIndices(values_is,&values));
    PetscCall(ISDestroy(&values_is)); 
  }
  PetscFunctionReturn(0);
}


