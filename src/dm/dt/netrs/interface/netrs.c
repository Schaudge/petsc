#include "petscdmlabel.h"
#include "petscnetrp.h"
#include "petscsystypes.h"
#include <petscnetrs.h> /*I "petscnetrs.h" I*/
#include <petsc/private/netrsimpl.h>
#include <petsc/private/riemannsolverimpl.h> /* to be removed after adding fluxfunction class */

PetscLogEvent NetRS_SetUp_VecSpace;
PetscLogEvent NetRS_Solve_Total;
PetscLogEvent NetRS_Solve_Communication;

PetscLogEvent NetRS_Solve_SubVecBuild;
PetscLogEvent NetRS_Solve_TopologyBuild;
PetscLogEvent NetRS_Solve_IS;

/*@
   NetRSSetUp - Sets up the internal data structures for the later use of a NetRS. 

   Note that for now this only provides a state where network can be set. After setup a network is 
   garunteed to have been set, and it cannot be reset. Functions that rquire the network to be set 
   should call NetRSSetUp(). 

   Collective on NetRS

   Input Parameter:
.  rs - the NetRS context obtained from NetRSCreate()  


   Level: advanced

.seealso: NetRSCreate(), NetRSSetFlux()
@*/
PetscErrorCode NetRSSetUp(NetRS rs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (rs->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(rs->network, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "No Network Set to the NetRS.");
  if (rs->ops->setup) { PetscCall((*rs->ops->setup)(rs)); }
  rs->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   NetRSReset - Resets a NetRS context and removes any allocated internal petsc objects

   Collective on NetRS

   Input Parameter:
.  rs - the RiemanmSolver context obtained from NetRSCreate()

   Level: beginner

.seealso: NetRSCreate(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode NetRSReset(NetRS rs)
{
  PetscInt i, numnetrp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscTryTypeMethod(rs, reset);
  PetscCall(NetRSResetVectorSpace(rs));
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  for (i = 0; i < numnetrp; i++) { PetscCall(NetRPDestroy(&rs->netrp[i])); }
  PetscCall(PetscFree(rs->netrp));
  PetscCall(DMDestroy(&rs->network));
  PetscCall(DMLabelReset(rs->subgraphs));
  PetscCall(DMLabelReset(rs->VertexDeg_shared));
  PetscCall(DMLabelReset(rs->InVertexDeg));
  PetscCall(DMLabelReset(rs->OutVertexDeg));
  PetscCall(PetscHMapNetRPIReset(rs->netrphmap));
  PetscCall(PetscHMapIDestroy(&rs->vertex_shared_offset));
  rs->vertexdeg_shared_cached = PETSC_FALSE;
  rs->inoutvertexdeg_cached   = PETSC_FALSE;
  rs->setupcalled             = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscInt i, numnetrp, size;
  DM       new_network;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscTryTypeMethod(rs, resetvecspace);
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  for (i = 0; i < numnetrp; i++) {
    PetscCall(PetscHSetIDestroy(&rs->vertexdegrees[i]));
    PetscCall(PetscHSetIJDestroy(&rs->inoutdegs[i]));
    PetscCall(ISDestroy(&rs->subgraphIS[i]));
  }
  PetscCall(PetscFree(rs->vertexdegrees));
  PetscCall(PetscFree(rs->inoutdegs));
  PetscCall(PetscFree(rs->subgraphIS));
  PetscCall(PetscHSetIDestroy(&rs->vertexdegrees_total));
  PetscCall(PetscHSetIJDestroy(&rs->inoutdeg_total));
  PetscCall(PetscHMapIDestroy(&rs->vertex_shared_vec_offset));
  PetscCall(VecDestroy(&rs->U));
  PetscCall(PetscFree(rs->is_wrk_index));

  PetscCall(PetscHMapIGetSize(rs->dofs_to_Vec, &size));
  for (i = 0; i < size; i++) {
    PetscCall(VecDestroy(&rs->Uv[i]));
    PetscCall(VecDestroy(&rs->Fluxv[i]));
  }
  PetscCall(PetscFree2(rs->Uv, rs->Fluxv));
  PetscCall(PetscHMapIClear(rs->dofs_to_Vec));
  /* need to undo the finalized components on the network */
  if (rs->network) {
    PetscCall(DMClone(rs->network, &new_network));
    PetscCall(DMDestroy(&rs->network));
    rs->network = new_network;
  }
  PetscCall(PetscFree(rs->edgein_shared));
  PetscCall(PetscFree(rs->edgein_wrk));
  PetscCall(PetscHMapIDestroy(&rs->edgein_shared_offset));

  rs->setupvectorspace = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode NetRSDestroy(NetRS *rs)
{
  PetscFunctionBegin;
  if (!*rs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*rs, NETRS_CLASSID, 1);
  if (--((PetscObject)(*rs))->refct > 0) {
    *rs = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*rs)->ops->destroy) PetscCall((*(*rs)->ops->destroy)((*rs)));
  PetscCall(NetRSReset(*rs));
  PetscCall(DMLabelDestroy(&(*rs)->VertexDeg_shared));
  PetscCall(DMLabelDestroy(&(*rs)->subgraphs));
  PetscCall(DMLabelDestroy(&(*rs)->InVertexDeg));
  PetscCall(DMLabelDestroy(&(*rs)->OutVertexDeg));
  PetscCall(PetscHMapNetRPIDestroy(&(*rs)->netrphmap));
  PetscCall(PetscHMapIDestroy(&(*rs)->dofs_to_Vec));
  PetscCall(ISDestroy(&(*rs)->is_wrk));
  if ((*rs)->rs) PetscCall(RiemannSolverDestroy(&(*rs)->rs));
  PetscCall(PetscHeaderDestroy(rs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  NetRSDuplicate - Create a new netrs of the same type as the original with the same settings. Still requires a call to setup after this call 
  as the intended use is to set the parameters for a "master" netrs duplicate it to other NetRS and change the types of the new netrs to the desired types. 
  This is the quick way of getting multiple netrs of different types for the same physics. 
*/

PetscErrorCode NetRSDuplicate(NetRS netrs, NetRS *newnetrs)
{
  MPI_Comm comm;
  NetRS    netrs_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(netrs, NETRS_CLASSID, 1);
  PetscValidPointer(newnetrs, 2);
  PetscValidType(netrs, 1);

  PetscCall(PetscObjectGetComm((PetscObject)netrs, &comm));
  PetscCall(NetRSCreate(comm, &netrs_new));
  /* copy over the parameters and physics from netrs to newnetrs */

  /* topology */
  PetscCall(DMClone(netrs->network, &netrs_new->network));
  /* physics*/
  netrs_new->user = netrs->user;
  PetscCall(NetRSSetFlux(netrs_new, netrs->rs));
  *newnetrs = netrs_new;
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode NetRSSetApplicationContext(NetRS rs, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  rs->user = usrP;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  PetscCheck(!rs->setupcalled, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "NetRSSetUp() has already been called, the Network cannot be changed. Call NetRSReset() if you need change the network. ");
  if (rs->network) PetscCall(DMDestroy(&rs->network));
  PetscCall(DMClone(network, &networkclone));
  rs->network = networkclone;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscCheck(rs->network, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "NetRS has no network. One must be set by NetRSSetNetwork()");
  *network = rs->network;
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode NetRSGetApplicationContext(NetRS rs, void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  *(void **)usrP = rs->user;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  const char *defaultType;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* Type Option */
  if (!((PetscObject)rs)->type_name) {
    defaultType = NETRSBASIC;
  } else {
    defaultType = ((PetscObject)rs)->type_name;
  }
  if (!NetRSRegisterAllCalled) PetscCall(NetRSRegisterAll());

  PetscObjectOptionsBegin((PetscObject)rs);
  PetscCall(PetscOptionsFList("-netrs_type", "NetRS", "NetRSSetType", NetRSList, defaultType, name, 256, &flg));
  if (flg) {
    PetscCall(NetRSSetType(rs, name));
  } else if (!((PetscObject)rs)->type_name) {
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
  if (rs->ops->setfromoptions) { PetscCall((*rs->ops->setfromoptions)(PetscOptionsObject, rs)); }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)rs, PetscOptionsObject));
  PetscOptionsEnd();
  /*
    TODO:  View from options here ? 
  */
  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode NetRSView(NetRS rs, PetscViewer viewer)
{
  PetscFunctionBegin;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSSetFlux(NetRS nrs, RiemannSolver flux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nrs, NETRS_CLASSID, 1);
  PetscValidHeaderSpecific(flux, RIEMANNSOLVER_CLASSID, 1);
  PetscCheck(!nrs->setupvectorspace, PetscObjectComm((PetscObject)nrs), PETSC_ERR_ARG_WRONGSTATE, "The NetRS Vector Space has already been setup. The Flux cannot be changed now. If needed Call NetRSResetVectorSpace() first");
  if (nrs->rs) PetscCall(RiemannSolverDestroy(&nrs->rs));
  PetscCall(PetscObjectReference((PetscObject)flux));
  nrs->rs = flux;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSGetFlux(NetRS rs, RiemannSolver *flux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscValidHeaderSpecific(*flux, RIEMANNSOLVER_CLASSID, 1);
  if (rs->rs) *flux = rs->rs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* internal function for adding the NetRP to the NetRS. This assumes that the NetRP has not already 
been added. 
*/
static PetscErrorCode NetRSAddNetRP(NetRS rs, NetRP rp)
{
  PetscInt numlabelvalues;
  NetRP   *netrp_new;

  PetscFunctionBegin;
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numlabelvalues)); /* current number of netrp stored */
  PetscCall(DMLabelAddStratum(rs->subgraphs, numlabelvalues));
  PetscCall(PetscHMapNetRPISet(rs->netrphmap, rp, numlabelvalues));

  /* create new memory and move */
  PetscCall(PetscMalloc1(numlabelvalues + 1, &netrp_new));
  PetscCall(PetscArraycpy(netrp_new, rs->netrp, numlabelvalues));

  PetscCall(PetscFree(rs->netrp));
  rs->netrp = netrp_new;

  rs->netrp[numlabelvalues] = rp;
  PetscCall(PetscObjectReference((PetscObject)rp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* uses local vertex numbering */
PetscErrorCode NetRSAddNetRPatVertex(NetRS rs, PetscInt v, NetRP rp)
{
  DM        network;
  PetscBool flg;
  PetscInt  vStart, vEnd, defaultval, pval, index;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscValidHeaderSpecific(rp, NETRP_CLASSID, 2);
  PetscCheck(!rs->setupvectorspace, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "The NetRS Vector Space has already been setup. No new NetRP can be added to vertices. If needed Call NetRSResetVectorSpace() first");

  PetscCall(NetRSSetUp(rs));
  PetscCall(NetRSGetNetwork(rs, &network));

  /* Check if this NetRP has already been added to the network */
  PetscCall(PetscHMapNetRPIHas(rs->netrphmap, rp, &flg));
  if (!flg) PetscCall(NetRSAddNetRP(rs, rp));

  /* check if vertex v belongs to the DMNetwork */
  PetscCall(DMNetworkGetVertexRange(rs->network, &vStart, &vEnd));
  PetscCheck(vStart <= v && v < vEnd, PetscObjectComm((PetscObject)rs), PETSC_ERR_USER_INPUT, "Input Vertex %" PetscInt_FMT " is not a vertex on the DMNetwork attached to NetRS, which has range %" PetscInt_FMT " to %" PetscInt_FMT, v, vStart, vEnd);
  /* Check if vertex v has any values associated with it. NetRS assumes that there is only one NetRP
  for each vertex.
  
  Perhaps should create a disjoint label implementation that takes care of this automatically? Would definitely be more performant, 
  and disjoint partitions are a useful thing in general. 
  */
  PetscCall(DMLabelGetDefaultValue(rs->subgraphs, &defaultval));
  PetscCall(DMLabelGetValue(rs->subgraphs, v, &pval));
  PetscCheck(pval == defaultval, PetscObjectComm((PetscObject)rs), PETSC_ERR_USER_INPUT, "NetRS assumes a single NetRP for each vertex. Inputted vertex %" PetscInt_FMT " already has an assigned NetRP. TODO Print the name of the NetRP already assigned here", v);

  /* Add vertex v to the label value associated with the NetRP */
  PetscCall(PetscHMapNetRPIGet(rs->netrphmap, rp, &index));
  PetscCall(DMLabelSetValue(rs->subgraphs, v, index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*should be part of DMNetwork itself */
PetscErrorCode NetRSGetVertexDegree(NetRS rs, PetscInt v, PetscInt *vdeg)
{
  PetscInt  vStart, vEnd;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscCall(DMNetworkCacheVertexDegrees(rs, rs->network));
  PetscCall(DMNetworkGetVertexRange(rs->network, &vStart, &vEnd));
  PetscCheck(v >= vStart && v < vEnd, PetscObjectComm((PetscObject)rs), PETSC_ERR_USER_INPUT, "Input Vertex: %" PetscInt_FMT " is outside the vertex range of the DMNetwork, vStart: %" PetscInt_FMT "  vEnd: %" PetscInt_FMT, v, vStart, vEnd);
  PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
  if (flg) {
    PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, vdeg));
  } else {
    PetscCall(DMNetworkGetSupportingEdges(rs->network, v, vdeg, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSGetDirectedVertexDegrees(NetRS rs, PetscInt v, PetscInt *indeg, PetscInt *outdeg)
{
  PetscInt vStart, vEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscCall(DMNetworkCacheInOutVertexDegrees(rs, rs->network));
  PetscCall(DMNetworkGetVertexRange(rs->network, &vStart, &vEnd));
  PetscCheck(v >= vStart && v < vEnd, PetscObjectComm((PetscObject)rs), PETSC_ERR_USER_INPUT, "Input Vertex: %" PetscInt_FMT " is outside the vertex range of the DMNetwork, vStart: %" PetscInt_FMT "  vEnd: %" PetscInt_FMT, v, vStart, vEnd);
  PetscCall(DMLabelGetValue(rs->InVertexDeg, v, indeg));
  PetscCall(DMLabelGetValue(rs->OutVertexDeg, v, outdeg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* internal logic for passing down netrs physics down to the underlyin netrp solvers */
static PetscErrorCode NetRSSetNetRPPhysics(NetRS rs)
{
  PetscInt      numnetrp, i;
  PetscBool     netrp_setup;
  RiemannSolver flux;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  for (i = 0; i < numnetrp; i++) {
    PetscCall(NetRPisSetup(rs->netrp[i], &netrp_setup));
    if (netrp_setup) continue;
    PetscCall(NetRPGetFlux(rs->netrp[i], &flux));
    if (flux) {
      PetscCall(NetRPSetUp(rs->netrp[i]));
    } else {
      PetscCall(NetRSGetFlux(rs, &flux));
      PetscCall(NetRPSetFlux(rs->netrp[i], flux));
      PetscCall(NetRPSetUp(rs->netrp[i]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  Sets Up the global vector spaces for the flux and initial states for the Riemann problems. 
  After thiis no new NetRP objects can be added to the network, and the physics cannot be changed. 
 */

PetscErrorCode NetRSSetUpVectorSpace(NetRS rs)
{
  PetscInt        v, numnetrp, i, j, compindex, size, numfields, vdeg, maxsize, off, index, maxdeg, maxnumfields, vStart, vEnd;
  PetscInt       *vals, *keys, *indeg, *outdeg, *vdegs, *dofs;
  const PetscInt *v_subgraph;
  char            compname[64];
  PetscBool       flg;
  PetscHSetI      vec_wrk_size;
  PetscHashIJKey *ijkey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (rs->setupvectorspace) PetscFunctionReturn(PETSC_SUCCESS);
  rs->setupvectorspace = PETSC_TRUE;
  PetscLogEventBegin(NetRS_SetUp_VecSpace, 0, 0, 0, 0);
  PetscCall(NetRSSetNetRPPhysics(rs));

  /* For each NetRP on NetRS, add a component to the network and add dofs for the local Riemann problem 
  for the marked vertices  */
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  PetscCheck(numnetrp > 0, PetscObjectComm((PetscObject)rs), PETSC_ERR_ARG_WRONGSTATE, "NetRS has no NetRP. Cannot build a vector space.");
  PetscCall(PetscMalloc1(numnetrp, &rs->subgraphIS));
  maxnumfields = 0;
  for (i = 0; i < numnetrp; i++) {
    PetscCall(PetscSNPrintf(compname, 64, "NetRP_%i", i));
    PetscCall(DMNetworkRegisterComponent(rs->network, compname, 0, &compindex));
    PetscCheck(i == compindex, PetscObjectComm((PetscObject)rs), PETSC_ERR_SUP, "This should not happen.");
    PetscCall(DMLabelGetStratumIS(rs->subgraphs, i, &rs->subgraphIS[i]));
    PetscCall(ISGetIndices(rs->subgraphIS[i], &v_subgraph));
    PetscCall(ISGetSize(rs->subgraphIS[i], &size));
    PetscCall(NetRPGetNumFields(rs->netrp[i], &numfields));
    if (numfields > maxnumfields) maxnumfields = numfields;
    /* add dofs to each marked vertex */
    for (j = 0; j < size; j++) {
      v = v_subgraph[j];
      PetscCall(NetRSGetVertexDegree(rs, v, &vdeg));
      PetscCall(DMNetworkAddComponent(rs->network, v, i, NULL, numfields * vdeg)); /* each vertex riemann problem has this size */
    }
    PetscCall(ISRestoreIndices(rs->subgraphIS[i], &v_subgraph));
  }
  PetscCall(DMNetworkFinalizeComponents(rs->network));
  /* create hmap and array for storing the edgein information for shared vertices */
  PetscCall(DMNetworkCreateEdgeInInfo(rs));

  /* Create the vectors used in the solver */
  PetscCall(DMCreateGlobalVector(rs->network, &rs->U));
  /* Now we preallocate the NetRP solvers */
  PetscCall(PetscMalloc1(numnetrp, &rs->vertexdegrees));
  PetscCall(PetscMalloc1(numnetrp, &rs->inoutdegs));
  for (i = 0; i < numnetrp; i++) {
    PetscCall(PetscHSetICreate(&rs->vertexdegrees[i]));
    PetscCall(PetscHSetIJCreate(&rs->inoutdegs[i]));
  }
  PetscCall(PetscHSetICreate(&rs->vertexdegrees_total));
  PetscCall(PetscHSetIJCreate(&rs->inoutdeg_total));
  PetscCall(DMNetworkComputeUniqueVertexDegreesLocal(rs, rs->network, rs->subgraphs, rs->vertexdegrees, rs->vertexdegrees_total));
  PetscCall(DMNetworkComputeUniqueVertexInOutDegreesLocal(rs, rs->network, rs->subgraphs, rs->inoutdegs, rs->inoutdeg_total));
  maxsize = 0;
  for (i = 0; i < numnetrp; i++) {
    PetscCall(PetscHSetIJGetSize(rs->inoutdegs[i], &size));
    if (size > maxsize) maxsize = size;
  }
  PetscCall(PetscMalloc3(maxsize, &ijkey, maxsize, &indeg, maxsize, &outdeg));
  maxdeg = 0;
  for (i = 0; i < numnetrp; i++) {
    off = 0;
    PetscCall(PetscHSetIJGetElems(rs->inoutdegs[i], &off, ijkey));
    for (j = 0; j < off; j++) {
      indeg[j]  = ijkey[j].i;
      outdeg[j] = ijkey[j].j;
    }
    PetscCall(NetRPCacheSolvers(rs->netrp[i], off, indeg, outdeg));
    for (j = 0; j < off; j++) {
      if (indeg[j] + outdeg[j] > maxdeg) maxdeg = indeg[j] + outdeg[j];
    }
  }
  PetscCall(PetscFree3(ijkey, indeg, outdeg));
  /* create IS wrk array for indices */
  PetscCall(PetscMalloc1(maxdeg * maxnumfields, &rs->is_wrk_index));
  /* create boolean wrk array for edge in */
  PetscCall(PetscMalloc1(maxdeg, &rs->edgein_wrk));

  /*Generate the work vecs Uv and Fluxv */
  PetscCall(PetscHSetICreate(&vec_wrk_size));
  for (i = 0; i < numnetrp; i++) {
    PetscCall(PetscHSetIGetSize(rs->vertexdegrees[i], &size));
    PetscCall(PetscMalloc1(size, &vdegs));
    off = 0;
    PetscCall(PetscHSetIGetElems(rs->vertexdegrees[i], &off, vdegs));
    PetscCall(NetRPGetNumFields(rs->netrp[i], &numfields));
    for (j = 0; j < size; j++) { PetscCall(PetscHSetIAdd(vec_wrk_size, numfields * vdegs[j])); }
    PetscFree(vdegs);
  }
  PetscCall(PetscHSetIGetSize(vec_wrk_size, &size));
  PetscCall(PetscMalloc2(size, &rs->Uv, size, &rs->Fluxv));
  PetscCall(PetscMalloc1(size, &dofs));
  off = 0;
  PetscCall(PetscHMapIClear(rs->dofs_to_Vec)); /* just in case */
  PetscCall(PetscHSetIGetElems(vec_wrk_size, &off, dofs));
  for (i = 0; i < size; i++) {
    PetscCall(PetscHMapISet(rs->dofs_to_Vec, dofs[i], i));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dofs[i], &rs->Uv[i]));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, dofs[i], &rs->Fluxv[i]));
  }
  PetscCall(PetscHSetIDestroy(&vec_wrk_size));
  PetscFree(dofs);

  /* generate the local offsets for shared vertices and their vector spaces*/
  PetscCall(DMNetworkCreateLocalEdgeNumbering(rs, rs->network));

  /* create vector space offsets */
  PetscCall(PetscHMapICreate(&rs->vertex_shared_vec_offset));
  PetscCall(PetscHMapIGetSize(rs->vertex_shared_offset, &size));
  PetscCall(PetscMalloc2(size, &keys, size, &vals));
  off = 0;
  PetscCall(PetscHMapIGetPairs(rs->vertex_shared_offset, &off, keys, vals));

  /* iterate through the shared vertex offsets and, if belonging to a netrp adjust offset 
  by that netrp's numfields. Note this may be different for different vertices in this general implementation */
  PetscCall(DMNetworkGetVertexRange(rs->network, &vStart, &vEnd));
  PetscCall(DMLabelCreateIndex(rs->subgraphs, vStart, vEnd));
  for (i = 0; i < size; i++) {
    PetscCall(DMLabelHasPoint(rs->subgraphs, keys[i], &flg));
    if (!flg) continue;
    PetscCall(DMLabelGetValue(rs->subgraphs, keys[i], &index));
    PetscCall(NetRPGetNumFields(rs->netrp[index], &numfields));
    PetscCall(PetscHMapISet(rs->vertex_shared_vec_offset, keys[i], vals[i] * numfields));
  }
  PetscCall(PetscFree2(keys, vals));
  PetscLogEventEnd(NetRS_SetUp_VecSpace, 0, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMNetworkCacheInOutVertexDegrees(NetRS rs, DM network)
{
  PetscInt        v, i, vStart, vEnd, nroots, nleaves, nedges, invdeg, outvdeg;
  PetscSF         sf;
  DM              plex;
  const PetscInt *ilocal, *edges, *cone;
  PetscInt       *rootdata, *leafdata;
  PetscMPIInt     size, rank;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  if (rs->inoutvertexdeg_cached) PetscFunctionReturn(PETSC_SUCCESS); /* already been cached */
  rs->inoutvertexdeg_cached = PETSC_TRUE;
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Compute the inout degrees for the local portions of the Network */
  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  PetscCall(DMLabelReset(rs->InVertexDeg));
  PetscCall(DMLabelReset(rs->OutVertexDeg));

  for (v = vStart; v < vEnd; v++) {
    invdeg  = 0;
    outvdeg = 0;
    PetscCall(DMNetworkGetSupportingEdges(network, v, &nedges, &edges));
    for (i = 0; i < nedges; i++) {
      PetscCall(DMNetworkGetConnectedVertices(network, edges[i], &cone));
      if (cone[1] == v) {
        invdeg++;
      } else {
        outvdeg++;
      }
    }
    PetscCall(DMLabelSetValue(rs->InVertexDeg, v, invdeg));
    PetscCall(DMLabelSetValue(rs->OutVertexDeg, v, outvdeg));
  }
  if (size == 1) {
    PetscCall(DMLabelCreateIndex(rs->InVertexDeg, vStart, vEnd));  /* No new points added */
    PetscCall(DMLabelCreateIndex(rs->OutVertexDeg, vStart, vEnd)); /* No new points added */
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Adjust the Local In/Out Vertex Degrees by the shared vertices  */

  /* Pull out the pointsf used by the underlying DMPlex for the distributed network */
  PetscCall(DMNetworkGetPlex(network, &plex));
  PetscCall(DMGetPointSF(plex, &sf));

  /* SUPER IMPORTANT NOTE: THE SF FROM PLEX ASSUMES THE SAME ARRAY SIZE FOR ROOT DATA AND LEAF DATA, WITH 
  ILOCAL HOLDING THE ACTUAL LEAF ENTRIES IN THAT ARRAY. */
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  PetscCheck(ilocal != NULL || nleaves == 0, comm, PETSC_ERR_SUP, "Currently assumes a plex format for the DM PointSF, where leafdata has the same size as rootdata, and ilocal holds offsets. Should not have leaf data in continguous storage");
  PetscCall(PetscCalloc2(nroots, &rootdata, nroots, &leafdata));

  /* Fill the leaf data with the local In Vertex degrees */
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    PetscCall(DMLabelGetValue(rs->InVertexDeg, ilocal[i], &invdeg));
    leafdata[ilocal[i]] = invdeg;
  }
  /* reduce degree data from leaves to root. This gives the correct in vertex degree on the
  distributed graph */
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));
  for (v = vStart; v < vEnd; v++) {
    if (!rootdata[v]) continue;
    PetscCall(DMLabelGetValue(rs->InVertexDeg, v, &invdeg)); /* local value */
    rootdata[v] += invdeg;
    //  PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,v,rootdata[v]);
    PetscCall(DMLabelSetValue(rs->InVertexDeg, v, rootdata[v]));
  }
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  /*iterate through the leaf vertices and add their values to the label */
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    //PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,ilocal[i],leafdata[ilocal[i]]);
    PetscCall(DMLabelSetValue(rs->InVertexDeg, ilocal[i], leafdata[ilocal[i]]));
  }
  /* no new entries will be added, compute index for faster membership lookup */
  PetscCall(DMLabelCreateIndex(rs->InVertexDeg, vStart, vEnd));

  /* Now do OutVertDeg */
  PetscCall(PetscArrayzero(rootdata, nroots));
  PetscCall(PetscArrayzero(leafdata, nroots));
  /* Fill the leaf data with the local Out Vertex degrees */
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    PetscCall(DMLabelGetValue(rs->OutVertexDeg, ilocal[i], &outvdeg));
    leafdata[ilocal[i]] = outvdeg;
  }
  /* reduce degree data from leaves to root. This gives the correct in vertex degree on the
  distributed graph */
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));
  for (v = vStart; v < vEnd; v++) {
    if (!rootdata[v]) continue;
    PetscCall(DMLabelGetValue(rs->OutVertexDeg, v, &outvdeg)); /* local value */
    rootdata[v] += outvdeg;
    //  PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,v,rootdata[v]);
    PetscCall(DMLabelSetValue(rs->OutVertexDeg, v, rootdata[v]));
  }
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  /*iterate through the leaf vertices and add their values to the label */
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    //PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,ilocal[i],leafdata[ilocal[i]]);
    PetscCall(DMLabelSetValue(rs->OutVertexDeg, ilocal[i], leafdata[ilocal[i]]));
  }
  /* no new entries will be added, compute index for faster membership lookup */
  PetscCall(DMLabelCreateIndex(rs->OutVertexDeg, vStart, vEnd));
  PetscCall(PetscFree2(rootdata, leafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: Migrate this functionality to DMNetwork itself */

/* Caches vdeg, invdeg, and outvdeg for the vertices of the network. 
  - vdeg is cached only on shared vertices, as its locally stored on DMNetwork already 
  - invdeg and outvdeg are cached on all vertices DMNetwork are not stored anywhere 
*/
PetscErrorCode DMNetworkCacheVertexDegrees(NetRS rs, DM network)
{
  PetscInt        v, i, vStart, vEnd, nroots, nleaves, nedges;
  PetscSF         sf;
  DM              plex;
  const PetscInt *ilocal;
  PetscInt       *rootdata, *leafdata;
  PetscMPIInt     size, rank;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  if (rs->vertexdeg_shared_cached) PetscFunctionReturn(PETSC_SUCCESS); /* already been cached */
  rs->vertexdeg_shared_cached = PETSC_TRUE;
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) {
    PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
    PetscCall(DMLabelCreateIndex(rs->VertexDeg_shared, vStart, vEnd)); /* needed for has point calls */
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Pull out the pointsf used by the underlying DMPlex for the distributed network */
  PetscCall(DMNetworkGetPlex(network, &plex));
  PetscCall(DMGetPointSF(plex, &sf));

  /* SUPER IMPORTANT NOTE: THE SF FROM PLEX ASSUMES THE SAME ARRAY SIZE FOR ROOT DATA AND LEAF DATA, WITH 
  ILOCAL HOLDING THE ACTUAL LEAF ENTRIES IN THAT ARRAY. */
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  PetscCheck(ilocal != NULL || nleaves == 0, comm, PETSC_ERR_SUP, "Currently assumes a plex format for the DM PointSF, where leafdata has the same size as rootdata, and ilocal holds offsets. Should not have leaf data in continguous storage");
  PetscCall(PetscCalloc2(nroots, &rootdata, nroots, &leafdata));
  /* Fill the leaf data with the local vertex degrees */
  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    PetscCall(DMNetworkGetSupportingEdges(network, ilocal[i], &leafdata[ilocal[i]], NULL));
  }
  /* reduce degree data from leaves to root. This gives the correct vertex degree on the
  distributed graph */
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, leafdata, rootdata, MPIU_SUM));

  /* any nonzero entry in rootdata then has leaves, these are added to the label as 
     shared vertices. The local vertex degree are then added to the rootdata to create the 
     correct vertex degree. */
  PetscCall(DMLabelReset(rs->VertexDeg_shared));
  for (v = vStart; v < vEnd; v++) {
    if (!rootdata[v]) continue;
    PetscCall(DMNetworkGetSupportingEdges(network, v, &nedges, NULL));
    rootdata[v] += nedges;
    //  PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,v,rootdata[v]);
    PetscCall(DMLabelSetValue(rs->VertexDeg_shared, v, rootdata[v]));
  }
  /* Rootdata contains the correct vertex degs, and these have been added to the label*/
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  /*iterate through the leaf vertices and add their values to the label */
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    //PetscPrintf(PETSC_COMM_SELF,"[%i] v: %"PetscInt_FMT " vdeg %"PetscInt_FMT"\n",rank,ilocal[i],leafdata[ilocal[i]]);
    PetscCall(DMLabelSetValue(rs->VertexDeg_shared, ilocal[i], leafdata[ilocal[i]]));
  }
  /* no new entries will be added, compute index for faster membership lookup */
  PetscCall(DMLabelCreateIndex(rs->VertexDeg_shared, vStart, vEnd));
  PetscCall(PetscFree2(rootdata, leafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The existence of this code is an abomination */

PetscErrorCode DMNetworkCreateEdgeInInfo(NetRS rs)
{
  PetscMPIInt     commsize, rank;
  MPI_Comm        comm;
  Vec             tmpVec, tmpVecloc;
  DM              tmpclone;
  PetscInt        i, j, v, vStart, vEnd, vdeg, numedges, off, offv, size, comp_index;
  const PetscInt *edges, *cone;
  PetscBool       flg, *edgein_v;
  PetscScalar    *edgein;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (commsize == 1) {
    PetscCall(DMLabelGetNumValues(rs->subgraphs, &comp_index));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMClone(rs->network, &tmpclone));
  PetscCall(DMNetworkRegisterComponent(tmpclone, "edge in", 0, &comp_index));
  /* needed ... */
  PetscCall(DMNetworkCacheVertexDegrees(rs, rs->network));
  PetscCall(DMNetworkCreateLocalEdgeNumbering(rs, rs->network));

  PetscCall(DMNetworkGetVertexRange(rs->network, &vStart, &vEnd));

  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
    if (!flg) continue;
    PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
    PetscCall(DMNetworkAddComponent(tmpclone, v, 0, NULL, vdeg));
  }
  PetscCall(DMNetworkFinalizeComponents(tmpclone));

  /* can now use vecs from tmpclone to communicate the edge information for each vertex */

  PetscCall(DMCreateLocalVector(tmpclone, &tmpVecloc));
  PetscCall(DMCreateGlobalVector(tmpclone, &tmpVec));

  PetscCall(VecZeroEntries(tmpVec));
  PetscCall(VecZeroEntries(tmpVecloc));
  PetscCall(VecGetArray(tmpVecloc, &edgein));
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
    if (!flg) continue;
    PetscCall(DMNetworkGetSupportingEdges(tmpclone, v, &numedges, &edges));
    PetscCall(DMNetworkGetLocalVecOffset(tmpclone, v, ALL_COMPONENTS, &off));
    PetscCall(PetscHMapIGet(rs->vertex_shared_offset, v, &offv));
    for (i = 0; i < numedges; i++) {
      PetscCall(DMNetworkGetConnectedVertices(tmpclone, edges[i], &cone));
      edgein[i + offv + off] = (cone[1] == v) ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  PetscCall(VecRestoreArray(tmpVecloc, &edgein));
  PetscCall(DMLocalToGlobal(tmpclone, tmpVecloc, ADD_VALUES, tmpVec));
  PetscCall(DMGlobalToLocal(tmpclone, tmpVec, INSERT_VALUES, tmpVecloc));

  /* now tmpVecloc contains the edgein information for all edges connected to the shared
  vertices on this processor. Add this as component of the rs->network */
  PetscCall(VecGetArray(tmpVecloc, &edgein));
  PetscCall(VecGetSize(tmpVecloc, &size));
  PetscCall(PetscMalloc1(size, &rs->edgein_shared)); /* single array for holding edge in data */
  edgein_v = rs->edgein_shared;
  PetscCall(PetscHMapICreate(&rs->edgein_shared_offset));
  j = 0;
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
    if (!flg) continue;
    PetscCall(DMNetworkGetLocalVecOffset(tmpclone, v, ALL_COMPONENTS, &off));
    PetscCall(DMNetworkGetComponent(tmpclone, v, ALL_COMPONENTS, NULL, NULL, &vdeg));
    for (i = 0; i < vdeg; i++) { edgein_v[i] = (PetscBool)edgein[off + i]; }
    edgein_v += vdeg;
    PetscCall(PetscHMapISet(rs->edgein_shared_offset, v, j));
    j += vdeg;
  }
  PetscCall(VecRestoreArray(tmpVecloc, &edgein));

  PetscCall(VecDestroy(&tmpVec));
  PetscCall(VecDestroy(&tmpVecloc));
  PetscCall(DMDestroy(&tmpclone));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: Duplicates a lot of code from DMNetworkCacheVertexDegrees. Should be
refactored along with that. */

/* 
Note, could use that data computed here to compute the stuff needed in 
DMNetworkCacheVertexDegrees for "Free". Only a single Bcast is needed instead of 
needing a reduce  + bcast. 
*/

PetscErrorCode DMNetworkCreateLocalEdgeNumbering(NetRS rs, DM network)
{
  DM              plex;
  PetscSF         sf;
  PetscSection    rootsection;
  PetscInt        i, pStart, pEnd, v, p, vStart, vEnd, multirootsize, dof, off, v_off, vlocaldeg, nleaves, nroots;
  PetscInt       *leafdata, *multirootdata;
  const PetscInt *rootdegree, *ilocal;
  PetscMPIInt     size;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (rs->vertex_offset_cached) PetscFunctionReturn(PETSC_SUCCESS); /* already created the map */
  rs->vertex_offset_cached = PETSC_TRUE;

  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(PetscHMapICreate(&rs->vertex_shared_offset));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscHMapICreate(&rs->vertex_shared_offset));
  /* Pull out the pointsf used by the underlying DMPlex for the distributed network */
  PetscCall(DMNetworkGetPlex(network, &plex));
  PetscCall(DMGetPointSF(plex, &sf));
  PetscCall(DMPlexGetChart(plex, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &rootsection)); /* used for organizing the multiroot data coming for sf gather */
  PetscCall(PetscSectionSetChart(rootsection, pStart, pEnd));
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  for (p = pStart; p < pEnd; ++p) PetscCall(PetscSectionSetDof(rootsection, p, rootdegree[p - pStart]));
  PetscCall(PetscSectionSetUp(rootsection));
  /*allocate the multiroot data and leaf data */
  PetscCall(PetscSectionGetStorageSize(rootsection, &multirootsize));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  PetscCheck(ilocal != NULL || nleaves == 0, comm, PETSC_ERR_SUP, "Currently assumes a plex format for the DM PointSF, where leafdata has the same size as rootdata, and ilocal holds offsets. Should not have leaf data in continguous storage");
  PetscCall(PetscMalloc2(multirootsize, &multirootdata, nroots, &leafdata));

  /* Fill the leaf data with the local vertex degrees */
  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    PetscCall(DMNetworkGetSupportingEdges(network, ilocal[i], &leafdata[ilocal[i]], NULL));
  }
  /* Gather the local vertex degrees of each leaf to the root */
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, leafdata, multirootdata));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, leafdata, multirootdata));

  /* Generate the shared vertex edge local ordering in place */
  for (v = vStart; v < vEnd; v++) {
    PetscCall(PetscSectionGetDof(rootsection, v, &dof));
    if (!dof) continue;
    PetscCall(PetscSectionGetOffset(rootsection, v, &off));
    v_off = 0;
    for (i = off; i < dof + off; i++) {
      vlocaldeg        = multirootdata[i];
      multirootdata[i] = v_off;
      v_off += vlocaldeg;
    }
    PetscCall(PetscHMapISet(rs->vertex_shared_offset, v, v_off));
  }
  /*now multirootdata contains all the offsets for the leaves of roots, 
  and the HMapI already contains the offsets for the roots. Scatter 
  The offsets back to leaves and then added leaf offsets to the HMapI */

  PetscCall(PetscSFScatterBegin(sf, MPIU_INT, multirootdata, leafdata));
  PetscCall(PetscSFScatterEnd(sf, MPIU_INT, multirootdata, leafdata));

  /*iterate and add to HMap*/
  for (i = 0; i < nleaves; i++) {
    if (ilocal[i] >= vEnd || ilocal[i] < vStart) continue;
    PetscCall(PetscHMapISet(rs->vertex_shared_offset, ilocal[i], leafdata[ilocal[i]]));
  }
  PetscCall(PetscFree2(multirootdata, leafdata));
  PetscCall(PetscSectionDestroy(&rootsection));
  rs->vertexdeg_shared_cached = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscHMapIView(PetscHMapI hmap, MPI_Comm comm)
{
  PetscInt    size, i, off;
  PetscInt   *keys, *vals;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscHMapIGetSize(hmap, &size));
  PetscCall(PetscMalloc2(size, &keys, size, &vals));
  off = 0;
  PetscCall(PetscHMapIGetPairs(hmap, &off, keys, vals));
  PetscCall(PetscSynchronizedPrintf(comm, "Rank [%i]\n\n", rank));

  for (i = 0; i < size; i++) { PetscCall(PetscSynchronizedPrintf(comm, "key[%" PetscInt_FMT "] val = %" PetscInt_FMT " \n", keys[i], vals[i])); }
  PetscCall(PetscSynchronizedFlush(comm, NULL));
  PetscCall(PetscFree2(keys, vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMNetworkComputeUniqueVertexDegreesLocal - Returns the unique set of vertex 
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
PetscErrorCode DMNetworkComputeUniqueVertexDegreesLocal(NetRS rs, DM network, DMLabel marked, PetscHSetI *vertexdegrees, PetscHSetI totalvertexdegrees)
{
  PetscInt        numsubgraphs, i, j, v, vStart, vEnd, vdeg, numpoints;
  IS              values_is, point_is;
  const PetscInt *values, *points;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  /* Clear Hash */
  PetscCall(PetscHSetIClear(totalvertexdegrees));
  /* Generate the vertex degree labeling */
  PetscCall(DMNetworkCacheVertexDegrees(rs, network));

  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  /* generate the hash set for the entire local graph first */
  for (v = vStart; v < vEnd; v++) {
    /* check if v is in the vertex degree labeling. That means it is shared among processors 
       and needs to use the value from the labeling */
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
    if (flg) { /*get vertex degree from label*/
      PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
    } else { /* vertex is entirely local, so DMNetworkGetSupportingEdge returns correct vertex deg */
      PetscCall(DMNetworkGetSupportingEdges(network, v, &vdeg, NULL));
    }
    PetscCall(PetscHSetIAdd(totalvertexdegrees, vdeg));
  }

  /* repeat but iterate only through marked vertices */
  if (marked) {
    PetscCheck(vertexdegrees, PetscObjectComm((PetscObject)network), PETSC_ERR_USER_INPUT, "If providing a label of marked vertices, an array of PetscHSetI, one for each value in the label must be provided");
    PetscCall(DMLabelGetNumValues(marked, &numsubgraphs));
    PetscCall(DMLabelGetValueIS(marked, &values_is));
    PetscCall(ISGetIndices(values_is, &values));

    for (i = 0; i < numsubgraphs; i++) {
      PetscCall(PetscHSetIClear(vertexdegrees[i]));
      PetscCall(DMLabelGetStratumIS(marked, values[i], &point_is));
      if (point_is == NULL) continue;
      ;
      PetscCall(ISGetSize(point_is, &numpoints));
      PetscCall(ISGetIndices(point_is, &points));
      for (j = 0; j < numpoints; j++) {
        v = points[j];
        if (v < vStart || v >= vEnd) continue;
        PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
        if (flg) { /*get vertex degree from label*/
          PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
        } else { /* vertex is entirely local, so DMNetworkGetSupportingEdge returns correct vertex deg */
          PetscCall(DMNetworkGetSupportingEdges(network, v, &vdeg, NULL));
        }
        PetscCall(PetscHSetIAdd(vertexdegrees[i], vdeg));
      }
      PetscCall(ISRestoreIndices(point_is, &points));
      PetscCall(ISDestroy(&point_is));
    }
    PetscCall(ISRestoreIndices(values_is, &values));
    PetscCall(ISDestroy(&values_is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMNetworkComputeUniqueVertexInOutDegreesLocal - TODO
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
PetscErrorCode DMNetworkComputeUniqueVertexInOutDegreesLocal(NetRS rs, DM network, DMLabel marked, PetscHSetIJ *inoutdegs, PetscHSetIJ inoutdeg_total)
{
  PetscInt        numsubgraphs, i, j, v, vStart, vEnd, numpoints;
  IS              values_is, point_is;
  const PetscInt *values, *points;
  PetscHashIJKey  ijkey;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  /* Clear Hash */
  PetscCall(PetscHSetIJClear(inoutdeg_total));
  /* Generate the in/out vertex degree labeling */
  PetscCall(DMNetworkCacheInOutVertexDegrees(rs, network));

  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  /* generate the hash set for the entire local graph first */
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMLabelGetValue(rs->InVertexDeg, v, &ijkey.i));
    PetscCall(DMLabelGetValue(rs->OutVertexDeg, v, &ijkey.j));
    PetscCall(PetscHSetIJAdd(inoutdeg_total, ijkey));
  }

  /* repeat but iterate only through marked vertices */
  if (marked) {
    PetscCheck(inoutdegs, PetscObjectComm((PetscObject)network), PETSC_ERR_USER_INPUT, "If providing a label of marked vertices, an array of PetscHSetIJ, one for each value in the label must be provided");
    PetscCall(DMLabelGetNumValues(marked, &numsubgraphs));
    PetscCall(DMLabelGetValueIS(marked, &values_is));
    PetscCall(ISGetIndices(values_is, &values));

    for (i = 0; i < numsubgraphs; i++) {
      PetscCall(PetscHSetIJClear(inoutdegs[i]));
      PetscCall(DMLabelGetStratumIS(marked, values[i], &point_is));
      if (point_is == NULL) continue;
      PetscCall(ISGetSize(point_is, &numpoints));
      PetscCall(ISGetIndices(point_is, &points));
      for (j = 0; j < numpoints; j++) {
        v = points[j];
        if (v < vStart || v >= vEnd) continue;
        PetscCall(DMLabelGetValue(rs->InVertexDeg, v, &ijkey.i));
        PetscCall(DMLabelGetValue(rs->OutVertexDeg, v, &ijkey.j));
        PetscCall(PetscHSetIJAdd(inoutdegs[i], ijkey));
      }
      PetscCall(ISRestoreIndices(point_is, &points));
      PetscCall(ISDestroy(&point_is));
    }
    PetscCall(ISRestoreIndices(values_is, &values));
    PetscCall(ISDestroy(&values_is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  should be replaced by a new version of petscsection that allows each point to have "subsection" 
  to further split the vector space structure 
*/

PetscErrorCode NetRSGetVecSizeAtVertex(NetRS rs, PetscInt v, PetscInt *localsize, PetscInt *totalsize)
{
  PetscInt index, defaultval, numfields, numedges;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (totalsize) PetscCall(DMNetworkGetComponent(rs->network, v, ALL_COMPONENTS, NULL, NULL, totalsize));
  if (localsize) {
    PetscCall(DMLabelGetValue(rs->subgraphs, v, &index));
    PetscCall(DMLabelGetDefaultValue(rs->subgraphs, &defaultval));
    if (index == defaultval) *localsize = 0;
    else {
      PetscCall(NetRPGetNumFields(rs->netrp[index], &numfields));
      PetscCall(DMNetworkGetSupportingEdges(rs->network, v, &numedges, NULL));
      *localsize = numedges * numfields;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSGetVertexVecOffset(NetRS rs, PetscInt v, PetscInt *offlocal, PetscInt *globaloff)
{
  PetscInt  off, off_shared;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  if (globaloff) { PetscCall(DMNetworkGetLocalVecOffset(rs->network, v, ALL_COMPONENTS, globaloff)); }
  if (offlocal) {
    PetscCall(DMNetworkGetLocalVecOffset(rs->network, v, ALL_COMPONENTS, &off));
    PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
    if (flg) {
      PetscCall(PetscHMapIGet(rs->vertex_shared_vec_offset, v, &off_shared));
      *offlocal = off + off_shared;
    } else {
      *offlocal = off;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSCreateLocalVec(NetRS rs, Vec *localvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  PetscCall(NetRSSetUpVectorSpace(rs));
  PetscCall(DMCreateLocalVector(rs->network, localvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Bad interface so far, need to rework things .... */

/* currently does every! solve on the vertex, so only communication of Uloc not Fluxloc*/
PetscErrorCode NetRSSolveFlux(NetRS rs, Vec Uloc, Vec Fluxloc)
{
  PetscInt           i, j, index, numnetrp, nvert, ndofs, v, off, vdeg, vdegin, vdegout, edgeinoff, wrk_vec_index;
  const PetscInt    *v_subgraph, *edges, *cone;
  PetscBool          flg, *edgein;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscScalar       *flux;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* add error checking */

  PetscCall(NetRSSetUpVectorSpace(rs));
  PetscLogEventBegin(NetRS_Solve_Total, 0, 0, 0, 0);
  /* assumes U, F came from the NetRS DM, should enforce this by keeping them internal */
  PetscCall(VecZeroEntries(rs->U));
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscLogEventBegin(NetRS_Solve_Communication, 0, 0, 0, 0);
  PetscCall(DMLocalToGlobalBegin(rs->network, Uloc, ADD_VALUES, rs->U)); /* should optimize this with Barry's work */
  PetscCall(DMLocalToGlobalEnd(rs->network, Uloc, ADD_VALUES, rs->U));
  PetscCall(DMGlobalToLocalBegin(rs->network, rs->U, INSERT_VALUES, Uloc));
  PetscCall(DMGlobalToLocalEnd(rs->network, rs->U, INSERT_VALUES, Uloc));
  PetscLogEventEnd(NetRS_Solve_Communication, 0, 0, 0, 0);

  /* iterate through every single netrp stored and then every single vertex in those sets */
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  PetscCall(VecGetArray(Fluxloc, &flux));
  PetscCall(VecGetArrayRead(Uloc, &u));
  for (index = 0; index < numnetrp; index++) {
    PetscCall(ISGetIndices(rs->subgraphIS[index], &v_subgraph));
    PetscCall(ISGetLocalSize(rs->subgraphIS[index], &nvert));
    for (i = 0; i < nvert; i++) {
      v = v_subgraph[i];
      PetscLogEventBegin(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      PetscCall(DMNetworkGetComponent(rs->network, v, ALL_COMPONENTS, NULL, NULL, &ndofs));
      PetscCall(DMNetworkGetLocalVecOffset(rs->network, v, ALL_COMPONENTS, &off));
      PetscCall(PetscHMapIGet(rs->dofs_to_Vec, ndofs, &wrk_vec_index));
      PetscCall(VecPlaceArray(rs->Uv[wrk_vec_index], u + off));
      PetscCall(VecPlaceArray(rs->Fluxv[wrk_vec_index], flux + off));
      PetscLogEventEnd(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      /* get the edgein information to pass to the NetRP solver */
      PetscLogEventBegin(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
      if (flg) {
        PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
        PetscCall(PetscHMapIGet(rs->edgein_shared_offset, v, &edgeinoff));
        edgein = rs->edgein_shared + edgeinoff;
      } else {
        edgein = rs->edgein_wrk;
        PetscCall(DMNetworkGetSupportingEdges(rs->network, v, &vdeg, &edges));
        for (j = 0; j < vdeg; j++) {
          PetscCall(DMNetworkGetConnectedVertices(rs->network, edges[j], &cone));
          edgein[j] = (cone[1] == v) ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      // compute the vdegin and vdegout (should be stored in DMNetwork?)
      vdegin  = 0;
      vdegout = 0;
      for (j = 0; j < vdeg; j++) {
        if (edgein[j] == PETSC_TRUE) vdegin++;
        else vdegout++;
      }
      PetscLogEventEnd(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      PetscCall(NetRPSolveFlux(rs->netrp[index], vdegin, vdegout, edgein, rs->Uv[wrk_vec_index], rs->Fluxv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Uv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Fluxv[wrk_vec_index]));
    }
    PetscCall(ISRestoreIndices(rs->subgraphIS[index], &v_subgraph));
  }
  PetscCall(VecRestoreArray(Fluxloc, &flux));
  PetscCall(VecRestoreArrayRead(Uloc, &u));
  PetscLogEventEnd(NetRS_Solve_Total, 0, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* currently does every! solve on the vertex, so only communication of Uloc not Fluxloc*/
PetscErrorCode NetRSSolveFluxBegin(NetRS rs, Vec Uloc, Vec Fluxloc)
{
  PetscInt           i, j, index, numnetrp, nvert, ndofs, v, off, vdeg, vdegin, vdegout, edgeinoff, wrk_vec_index;
  const PetscInt    *v_subgraph, *edges, *cone;
  PetscBool          flg, *edgein;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscScalar       *flux;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* add error checking */

  PetscCall(NetRSSetUpVectorSpace(rs));
  PetscLogEventBegin(NetRS_Solve_Total, 0, 0, 0, 0);
  /* assumes U, F came from the NetRS DM, should enforce this by keeping them internal */
  PetscCall(VecZeroEntries(rs->U));
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscLogEventBegin(NetRS_Solve_Communication, 0, 0, 0, 0);
  PetscCall(DMLocalToGlobalBegin(rs->network, Uloc, ADD_VALUES, rs->U)); /* should optimize this with Barry's work */
  PetscLogEventEnd(NetRS_Solve_Communication, 0, 0, 0, 0);

  /* iterate through every single netrp stored and then every single vertex in those sets */
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  PetscCall(VecGetArray(Fluxloc, &flux));
  PetscCall(VecGetArrayRead(Uloc, &u));
  for (index = 0; index < numnetrp; index++) {
    PetscCall(ISGetIndices(rs->subgraphIS[index], &v_subgraph));
    PetscCall(ISGetLocalSize(rs->subgraphIS[index], &nvert));
    for (i = 0; i < nvert; i++) {
      v = v_subgraph[i];
      PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
      if (flg) continue; /* ONLY Solve processor local riemann problems */
      PetscLogEventBegin(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      PetscCall(DMNetworkGetComponent(rs->network, v, ALL_COMPONENTS, NULL, NULL, &ndofs));
      PetscCall(DMNetworkGetLocalVecOffset(rs->network, v, ALL_COMPONENTS, &off));
      PetscCall(PetscHMapIGet(rs->dofs_to_Vec, ndofs, &wrk_vec_index));
      PetscCall(VecPlaceArray(rs->Uv[wrk_vec_index], u + off));
      PetscCall(VecPlaceArray(rs->Fluxv[wrk_vec_index], flux + off));
      PetscLogEventEnd(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      /* get the edgein information to pass to the NetRP solver */
      PetscLogEventBegin(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      if (flg) {
        PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
        PetscCall(PetscHMapIGet(rs->edgein_shared_offset, v, &edgeinoff));
        edgein = rs->edgein_shared + edgeinoff;
      } else {
        edgein = rs->edgein_wrk;
        PetscCall(DMNetworkGetSupportingEdges(rs->network, v, &vdeg, &edges));
        for (j = 0; j < vdeg; j++) {
          PetscCall(DMNetworkGetConnectedVertices(rs->network, edges[j], &cone));
          edgein[j] = (cone[1] == v) ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      // compute the vdegin and vdegout (should be stored in DMNetwork?)
      vdegin  = 0;
      vdegout = 0;
      for (j = 0; j < vdeg; j++) {
        if (edgein[j] == PETSC_TRUE) vdegin++;
        else vdegout++;
      }
      PetscLogEventEnd(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      PetscCall(NetRPSolveFlux(rs->netrp[index], vdegin, vdegout, edgein, rs->Uv[wrk_vec_index], rs->Fluxv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Uv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Fluxv[wrk_vec_index]));
    }
    PetscCall(ISRestoreIndices(rs->subgraphIS[index], &v_subgraph));
  }
  PetscCall(VecRestoreArray(Fluxloc, &flux));
  PetscCall(VecRestoreArrayRead(Uloc, &u));

  PetscLogEventBegin(NetRS_Solve_Communication, 0, 0, 0, 0);
  PetscCall(DMLocalToGlobalEnd(rs->network, Uloc, ADD_VALUES, rs->U));
  PetscCall(DMGlobalToLocalBegin(rs->network, rs->U, INSERT_VALUES, Uloc));
  PetscLogEventEnd(NetRS_Solve_Communication, 0, 0, 0, 0);
  PetscLogEventEnd(NetRS_Solve_Total, 0, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NetRSSolveFluxEnd(NetRS rs, Vec Uloc, Vec Fluxloc)
{
  PetscInt           i, j, index, numnetrp, nvert, ndofs, v, off, vdeg, vdegin, vdegout, edgeinoff, wrk_vec_index;
  const PetscInt    *v_subgraph, *edges, *cone;
  PetscBool          flg, *edgein;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscScalar       *flux;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);
  /* add error checking */
  PetscLogEventBegin(NetRS_Solve_Total, 0, 0, 0, 0);
  /* assumes U, F came from the NetRS DM, should enforce this by keeping them internal */
  PetscCall(PetscObjectGetComm((PetscObject)rs, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscLogEventBegin(NetRS_Solve_Communication, 0, 0, 0, 0);
  PetscCall(DMGlobalToLocalEnd(rs->network, rs->U, INSERT_VALUES, Uloc)); /* finish communication */
  PetscLogEventEnd(NetRS_Solve_Communication, 0, 0, 0, 0);

  /* solve only the shared vertex riemann problems that havent yet been solved */

  /* iterate through every single netrp stored and then every single vertex in those sets */
  PetscCall(DMLabelGetNumValues(rs->subgraphs, &numnetrp));
  PetscCall(VecGetArray(Fluxloc, &flux));
  PetscCall(VecGetArrayRead(Uloc, &u));
  for (index = 0; index < numnetrp; index++) {
    PetscCall(ISGetIndices(rs->subgraphIS[index], &v_subgraph));
    PetscCall(ISGetLocalSize(rs->subgraphIS[index], &nvert));
    for (i = 0; i < nvert; i++) {
      v = v_subgraph[i];
      PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, &flg));
      if (!flg) continue; /* only solve shared vertex riemann problems */
      PetscLogEventBegin(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      PetscCall(DMNetworkGetComponent(rs->network, v, ALL_COMPONENTS, NULL, NULL, &ndofs));
      PetscCall(DMNetworkGetLocalVecOffset(rs->network, v, ALL_COMPONENTS, &off));
      PetscCall(PetscHMapIGet(rs->dofs_to_Vec, ndofs, &wrk_vec_index));
      PetscCall(VecPlaceArray(rs->Uv[wrk_vec_index], u + off));
      PetscCall(VecPlaceArray(rs->Fluxv[wrk_vec_index], flux + off));
      PetscLogEventEnd(NetRS_Solve_SubVecBuild, 0, 0, 0, 0);
      /* get the edgein information to pass to the NetRP solver */
      PetscLogEventBegin(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      if (flg) {
        PetscCall(DMLabelGetValue(rs->VertexDeg_shared, v, &vdeg));
        PetscCall(PetscHMapIGet(rs->edgein_shared_offset, v, &edgeinoff));
        edgein = rs->edgein_shared + edgeinoff;
      } else {
        edgein = rs->edgein_wrk;
        PetscCall(DMNetworkGetSupportingEdges(rs->network, v, &vdeg, &edges));
        for (j = 0; j < vdeg; j++) {
          PetscCall(DMNetworkGetConnectedVertices(rs->network, edges[j], &cone));
          edgein[j] = (cone[1] == v) ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      // compute the vdegin and vdegout (should be stored in DMNetwork?)
      vdegin  = 0;
      vdegout = 0;
      for (j = 0; j < vdeg; j++) {
        if (edgein[j] == PETSC_TRUE) vdegin++;
        else vdegout++;
      }
      PetscLogEventEnd(NetRS_Solve_TopologyBuild, 0, 0, 0, 0);
      PetscCall(NetRPSolveFlux(rs->netrp[index], vdegin, vdegout, edgein, rs->Uv[wrk_vec_index], rs->Fluxv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Uv[wrk_vec_index]));
      PetscCall(VecResetArray(rs->Fluxv[wrk_vec_index]));
    }
    PetscCall(ISRestoreIndices(rs->subgraphIS[index], &v_subgraph));
  }
  PetscCall(VecRestoreArray(Fluxloc, &flux));
  PetscCall(VecRestoreArrayRead(Uloc, &u));
  PetscLogEventEnd(NetRS_Solve_Total, 0, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMNetworkIsParallelVertex(NetRS rs, DM network, PetscInt v, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(network, DM_CLASSID, 2, DMNETWORK);
  PetscValidHeaderSpecific(rs, NETRS_CLASSID, 1);

  PetscCall(DMNetworkCacheVertexDegrees(rs, network));
  PetscCall(DMLabelHasPoint(rs->VertexDeg_shared, v, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}