#include "petscsys.h"
#include "petscvec.h"
#include "petscviewer.h"
static const char help[] = "DGNetwork Conservation Law Test Function. \n\
Just Runs an Simulation with the specified Setup. \n\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"
#include <petscviewerhdf5.h>

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  PetscCall(DGNetworkMonitorView(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDGNetworkMonitor_GLVis(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDGNetworkMonitor_GLVis_NET(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis_NET(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 
  this a type of traffic distribution that generically satisfies the criteria 
  to generate a unique linear programming solution

  columns are of the form  a_i = (1/2)^(i+1) if i \neq outdeg-2. 

  then a_{outdeg-2} = (1/2)^(outdeg-1) + (1/2)^(outdeg)

  which has \sum a_i = 1, and satisifes the technical constraint. 

*/
static PetscErrorCode TrafficDistribution(NetRP rp, PetscInt indeg, PetscInt outdeg, Mat distribution)
{
  PetscScalar *mat;
  PetscInt     i, j;
  PetscReal    val;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArray(distribution, &mat));
  if (outdeg == 2 && indeg == 1) {
    mat[0] = 0.5;
    mat[1] = 0.5;
  } else if (outdeg == 1) {
    for (i = 0; i < indeg; i++) { mat[i] = 1.0; }
  } else {
    /* equal distribution */
    for (j = 0; j < indeg; j++) {
      for (i = 0; i < outdeg; i++) {
        val = PetscPowRealInt(0.5, i + 1);
        if (i == outdeg - 2) val += PetscPowRealInt(0.5, outdeg);
        mat[j * outdeg + i] = val;
      }
    }
  }
  PetscCall(MatDenseRestoreArray(distribution, &mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TrafficPriority(NetRP rp, PetscInt indeg, PetscInt outdeg, Vec priority)
{
  PetscScalar *p;
  PetscInt     i;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(priority, &p));
  if (outdeg == 1 && indeg == 2) {
    p[0] = 4;
    p[1] = 1;
  } else {
    for (i = 0; i < indeg; i++) { p[i] = i + 1; }
    PetscCall(VecRestoreArray(priority, &p));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DGNetworkAssignNetRS_TrafficCustom(DGNetwork dgnet)
{
  PetscInt     v, vStart, vEnd, vdeg, indeg, outdeg, index;
  NetRP        netrpbdry, netrpcouple, netrpcouple_priority, netrpconstant1, netrpconstant2;
  PetscScalar *uconstant;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrpbdry));
  PetscCall(NetRPSetType(netrpbdry, NETRPOUTFLOW));
  PetscCall(NetRPSetFlux(netrpbdry, dgnet->physics.rs));

  PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrpcouple));
  PetscCall(NetRPSetType(netrpcouple, NETRPTRAFFICLWR));
  PetscCall(NetRPSetFlux(netrpcouple, dgnet->physics.rs));
  PetscCall(NetRPTrafficSetDistribution(netrpcouple, TrafficDistribution));

  PetscCall(NetRPDuplicate(netrpcouple, &netrpcouple_priority));
  PetscCall(NetRPSetType(netrpcouple_priority, NETRPTRAFFICLWR_PRIORITY));
  PetscCall(NetRPTrafficSetDistribution(netrpcouple_priority, TrafficDistribution));
  PetscCall(NetRPTrafficSetPriorityVec(netrpcouple_priority, TrafficPriority));

  PetscCall(NetRPDuplicate(netrpcouple, &netrpconstant1));
  PetscCall(NetRPSetType(netrpconstant1, NETRPCONSTANT));
  PetscCall(PetscMalloc1(1, &uconstant));
  uconstant[0] = 0.25;
  PetscCall(NetRPConstantSetData(netrpconstant1, uconstant));
  PetscCall(NetRPDuplicate(netrpconstant1, &netrpconstant2));
  PetscCall(NetRPSetType(netrpconstant2, NETRPCONSTANT));
  uconstant[0] = 0.4;
  PetscCall(NetRPConstantSetData(netrpconstant2, uconstant));
  PetscCall(PetscFree(uconstant));

  PetscCall(NetRSCreate(PETSC_COMM_WORLD, &dgnet->netrs));
  PetscCall(NetRSSetFromOptions(dgnet->netrs));
  PetscCall(NetRSSetFlux(dgnet->netrs, dgnet->physics.rs));
  PetscCall(NetRSSetNetwork(dgnet->netrs, dgnet->network));
  PetscCall(NetRSSetUp(dgnet->netrs));

  for (v = vStart; v < vEnd; v++) {
    PetscCall(NetRSGetVertexDegree(dgnet->netrs, v, &vdeg));
    /*
      type dispatching depending on number of edges
    */
    if (vdeg == 1) {
      PetscCall(DMNetworkGetGlobalVertexIndex(dgnet->network, v, &index));
      if (index == 4) {
        PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpconstant1));
      } else if (index == 6) {
        PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpconstant2));
      } else {
        PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpbdry));
      }
    } else {
      PetscCall(NetRSGetDirectedVertexDegrees(dgnet->netrs, v, &indeg, &outdeg));
      if (indeg <= outdeg) {
        PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpcouple));
      } else {
        PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpcouple_priority));
      }
    }
  }
  PetscCall(NetRSCreateLocalVec(dgnet->netrs, &dgnet->Flux));
  PetscCall(NetRSCreateLocalVec(dgnet->netrs, &dgnet->RiemannData));
  PetscCall(NetRPDestroy(&netrpbdry));
  PetscCall(NetRPDestroy(&netrpcouple));
  PetscCall(NetRPDestroy(&netrpcouple_priority));
          PetscCall(NetRPDestroy(&netrpconstant1));
        PetscCall(NetRPDestroy(& netrpconstant2));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  char                   physname[256] = "traffic", outputfile[256];
  PetscFunctionList      physics       = 0;
  MPI_Comm               comm;
  TS                     ts;
  DGNetwork              dgnet;
  PetscInt               maxorder = 2, systemsize, rhsversion = 2;
  PetscReal              maxtime;
  PetscMPIInt            size, rank;
  PetscBool              flg, limit = PETSC_TRUE, view3d = PETSC_FALSE, viewglvis = PETSC_FALSE, glvismode = PETSC_FALSE, viewfullnet = PETSC_FALSE, savefinal = PETSC_FALSE;
  DGNetworkMonitor       monitor = NULL;
  DGNetworkMonitor_Glvis monitor_gl;
  PetscViewer            vecbinary;
  char                   ofname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
  PetscViewer            viewer;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics, "shallow", PhysicsCreate_Shallow));
  PetscCall(PetscFunctionListAdd(&physics, "traffic", PhysicsCreate_Traffic));

  PetscCall(PetscCalloc1(1, &dgnet)); /* Replace with proper dgnet creation function */
  /* Set default values */
  dgnet->comm           = comm;
  dgnet->cfl            = 0.9;
  dgnet->networktype    = 6;
  dgnet->hratio         = 1;
  maxtime               = 2.0;
  dgnet->Mx             = 10;
  dgnet->initial        = 1;
  dgnet->ndaughters     = 2;
  dgnet->length         = 10.0;
  dgnet->view           = PETSC_FALSE;
  dgnet->jumptol        = 0.5;
  dgnet->diagnosticlow  = 0.5;
  dgnet->diagnosticup   = 1e-4;
  dgnet->linearcoupling = PETSC_FALSE;
  dgnet->M              = 50;
  dgnet->edgethickness  = -1;
  dgnet->dx             = 1000;

  /* Command Line Options */
  PetscOptionsBegin(comm, NULL, "DGNetwork solver options", "");
  PetscCall(PetscOptionsFList("-physics", "Name of physics model to use", "", physics, physname, physname, sizeof(physname), NULL));
  PetscCall(PetscOptionsInt("-initial", "Initial Condition (depends on the physics)", "", dgnet->initial, &dgnet->initial, NULL));
  PetscCall(PetscOptionsInt("-network", "Network topology to load, along with boundary condition information", "", dgnet->networktype, &dgnet->networktype, NULL));
  PetscCall(PetscOptionsReal("-cfl", "CFL number to time step at", "", dgnet->cfl, &dgnet->cfl, NULL));
  PetscCall(PetscOptionsReal("-length", "Length of Edges in the Network", "", dgnet->length, &dgnet->length, NULL));
  PetscCall(PetscOptionsInt("-Mx", "Smallest number of cells for an edge", "", dgnet->Mx, &dgnet->Mx, NULL));
  PetscCall(PetscOptionsInt("-ndaughters", "Number of daughter branches for network type 3", "", dgnet->ndaughters, &dgnet->ndaughters, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of the DG Basis", "", maxorder, &maxorder, NULL));
  PetscCall(PetscOptionsInt("-rhsversion", "Version of the RHS to use", "", rhsversion, &rhsversion, NULL));
  PetscCall(PetscOptionsBool("-savefinal", "View GLVis of Edge", "", savefinal, &savefinal, NULL));
  PetscCall(PetscOptionsBool("-view", "View the DG solution", "", dgnet->view, &dgnet->view, NULL));
  PetscCall(PetscOptionsBool("-uselimiter", "Use a limiter for the DG solution", "", limit, &limit, NULL));
  PetscCall(PetscOptionsReal("-jumptol", "Set jump tolerance for lame one-sided limiter", "", dgnet->jumptol, &dgnet->jumptol, NULL));
  PetscCall(PetscOptionsBool("-lincouple", "Use lax curve diagnostic for coupling", "", dgnet->linearcoupling, &dgnet->linearcoupling, NULL));
  PetscCall(PetscOptionsBool("-view_dump", "Dump the Glvis view or socket", "", glvismode, &glvismode, NULL));
  PetscCall(PetscOptionsBool("-view_3d", "View a 3d version of edge", "", view3d, &view3d, NULL));
  PetscCall(PetscOptionsBool("-view_glvis", "View GLVis of Edge", "", viewglvis, &viewglvis, NULL));
  PetscCall(PetscOptionsBool("-view_full_net", "View GLVis of Entire Network", "", viewfullnet, &viewfullnet, NULL));
  PetscCall(PetscOptionsReal("-dx", "Size of Cells in some cases", "", dgnet->dx, &dgnet->dx, NULL));
  PetscCall(PetscOptionsReal("-edge_thickness", "Thickness of edges in visualization", "", dgnet->edgethickness, &dgnet->edgethickness, NULL));
  PetscCall(PetscOptionsString("-ofilename", "The output mesh file", "ex55.c", ofname, ofname, sizeof(ofname), &flg));
  PetscOptionsEnd();
  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics, physname, &r));
    if (!r) SETERRQ(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(dgnet));
  }
  PetscCall(PetscMalloc1(dgnet->physics.dof, &dgnet->physics.order)); /* should be constructed by physics */
  PetscCall(MakeOrder(dgnet->physics.dof, dgnet->physics.order, maxorder));
  if (flg) { /* load a DMNetwork direclty and create tthe DGNet data. REALLY REALLY NEEDS A REWORK */
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &dgnet->network));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, ofname, FILE_MODE_READ, &viewer));
    PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
    PetscCall(DMLoad(dgnet->network, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMView(dgnet->network, PETSC_VIEWER_STDOUT_WORLD));

    /* recreate the dgnet info */
    PetscInt           nedges, nvertices, cdim, vStart, v, vEnd, off, e, eStart, eEnd, f, dof = dgnet->physics.dof, numdof, field;
    DGNETJunction      junctions = NULL;
    EdgeFE             DGEdges   = NULL;
    Vec                lcoord;
    const PetscInt    *cone;
    const PetscScalar *coord;
    PetscScalar        x1, x2, y1, y2;
    DM                 cdm;

    PetscCall(DMNetworkGetNumEdges(dgnet->network, &nedges, NULL));
    PetscCall(DMNetworkGetNumVertices(dgnet->network, &nvertices, NULL));
    PetscCall(PetscCalloc2(nvertices, &junctions, nedges, &DGEdges));

    PetscCall(DMGetCoordinateDim(dgnet->network, &cdim));
    PetscCall(DMGetCoordinatesLocal(dgnet->network, &lcoord));
    PetscCall(DMGetCoordinateDM(dgnet->network, &cdm));
    PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
    PetscCall(VecGetArrayRead(lcoord, &coord));
    for (v = vStart; v < vEnd; v++) {
      PetscCall(DMNetworkGetLocalVecOffset(cdm, v, ALL_COMPONENTS, &off));
      junctions[v - vStart].x = coord[off];
      junctions[v - vStart].y = coord[off + 1];
      //  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x: %e y: %e offset %"PetscInt_FMT "\n",  coord[off], coord[off+1],off));
    }
    PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
    for (e = eStart; e < eEnd; e++) {
      PetscCall(DMNetworkGetConnectedVertices(cdm, e, &cone));
      PetscCall(DMNetworkGetLocalVecOffset(cdm, cone[0], ALL_COMPONENTS, &off));
      x1 = coord[off];
      y1 = coord[off + 1];
      PetscCall(DMNetworkGetLocalVecOffset(cdm, cone[1], ALL_COMPONENTS, &off));
      x2 = coord[off];
      y2 = coord[off + 1];
      /* compute length */
      DGEdges[e - eStart].nnodes = dgnet->Mx;
      DGEdges[e - eStart].length = PetscSqrtScalar(PetscSqr(x2 - x1) + PetscSqr(y2 - y1));
    }
    PetscCall(VecRestoreArrayRead(lcoord, &coord));
    //PetscCall(VecView(lcoord,PETSC_VIEWER_STDOUT_WORLD));

    /* Allocate work space for the DG solver (so it doesn't have to be reallocated on each function evaluation) */
    PetscCall(PetscMalloc2(dof * dof, &dgnet->R, dof * dof, &dgnet->Rinv));
    PetscCall(PetscMalloc5(2 * dof, &dgnet->cuLR, 2 * dof, &dgnet->uLR, dof, &dgnet->flux, dof, &dgnet->speeds, dof, &dgnet->uPlus));
    /* allocate work space for the limiter suff */

    /* this variable should be stored elsewhere */
    dgnet->physics.maxorder = 0;
    for (field = 0; field < dof; field++) {
      if (dgnet->physics.order[field] > dgnet->physics.maxorder) { dgnet->physics.maxorder = dgnet->physics.order[field]; }
    }

    PetscCall(PetscMalloc5(dof, &dgnet->limitactive, (dgnet->physics.maxorder + 1) * dof, &dgnet->charcoeff, dof, &dgnet->cbdryeval_L, dof, &dgnet->cbdryeval_R, dof, &dgnet->cuAvg));
    PetscCall(PetscMalloc2(3 * dof, &dgnet->uavgs, 2 * dof, &dgnet->cjmpLR));
    PetscInt      KeyEdge, KeyJunction, dmsize;
    EdgeFE        edgefe;
    DGNETJunction junction;

    /* now add the components */
    numdof = 0;
    for (f = 0; f < dof; f++) { numdof += dgnet->physics.order[f] + 1; }
    PetscCall(DMNetworkRegisterComponent(dgnet->network, "junctionstruct", sizeof(struct _p_DGNETJunction), &KeyJunction));
    PetscCall(DMNetworkRegisterComponent(dgnet->network, "fvedgestruct", sizeof(struct _p_EdgeFE), &KeyEdge));
    for (e = eStart; e < eEnd; e++) {
      edgefe = &DGEdges[e - eStart];
      /*
      Add the data from the dmplex to the dmnetwork. We will create the global network vector from the dmnetwork and use the dmplex to manage the
      data on an edge after getting the offset for set the edge. The dmnetwork creates the vectors and, but the dmplex inside an edge is used to actually
      interact with the edge componenent of the network vector
    */
      dmsize = numdof * edgefe->nnodes;
      PetscCall(DMNetworkAddComponent(dgnet->network, e, KeyEdge, edgefe, dmsize));
    }
    /* Add Junction component to all local vertices. */
    for (v = vStart; v < vEnd; v++) {
      junction = &junctions[v - vStart];
      // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x: %e y: %e \n", junction->x,junction->y));
      PetscCall(DMNetworkAddComponent(dgnet->network, v, KeyJunction, junction, 0));
    }
    PetscCall(DMSetUp(dgnet->network));
    PetscCall(DMNetworkFinalizeComponents(dgnet->network));
    PetscCall(PetscFree2(junctions, DGEdges));
  } else {
    /* Generate Network Data */
    PetscCall(DGNetworkCreate(dgnet, dgnet->networktype, dgnet->Mx));
    /* Set Network Data into the DMNetwork (on proc[0]) */
    PetscCall(DGNetworkSetComponents(dgnet));
    /* Delete unneeded data in dgnet */
    PetscCall(DGNetworkCleanUp(dgnet));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "I made it \n"));

  PetscCall(DGNetworkBuildTabulation(dgnet));
  PetscCall(DMNetworkDistribute(&dgnet->network, 0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "I made it \n"));

  /* Create Vectors */
  PetscCall(DGNetworkCreateVectors(dgnet));
  /* Set up component dynamic data structures */
  PetscCall(DGNetworkBuildDynamic(dgnet));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "I made it \n"));

  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorCreate_Glvis(dgnet, &monitor_gl));
      if (viewfullnet) {
        PetscCall(DGNetworkMonitorAdd_Glvis_2D_NET(monitor_gl, "localhost", glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
      } else {
        if (view3d) {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis_3D(dgnet, monitor_gl, glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        } else {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis(dgnet, monitor_gl, glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        }
      }
    } else {
      PetscCall(DGNetworkMonitorCreate(dgnet, &monitor));
      PetscCall(DGNetworkAddMonitortoEdges(dgnet, monitor));
    }
  }
  /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to
   set all the physics parts at once) */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "I made it \n"));

  PetscCall(RiemannSolverCreate(dgnet->comm, &dgnet->physics.rs));
  PetscCall(RiemannSolverSetApplicationContext(dgnet->physics.rs, dgnet->physics.user));
  PetscCall(RiemannSolverSetFromOptions(dgnet->physics.rs));
  PetscCall(RiemannSolverSetFluxEig(dgnet->physics.rs, dgnet->physics.fluxeig));
  PetscCall(RiemannSolverSetRoeAvgFunct(dgnet->physics.rs, dgnet->physics.roeavg));
  PetscCall(RiemannSolverSetRoeMatrixFunct(dgnet->physics.rs, dgnet->physics.roemat));
  PetscCall(RiemannSolverSetEigBasis(dgnet->physics.rs, dgnet->physics.eigbasis));
  PetscCall(RiemannSolverSetFlux(dgnet->physics.rs, 1, dgnet->physics.dof, dgnet->physics.flux2));
  PetscCall(RiemannSolverSetLaxCurve(dgnet->physics.rs, dgnet->physics.laxcurve));
  PetscCall(RiemannSolverSetJacobian(dgnet->physics.rs, dgnet->physics.fluxder));
  PetscCall(RiemannSolverSetUp(dgnet->physics.rs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "I made it \n"));

  /* Set up NetRS */
  PetscCall(DGNetworkAssignNetRS_TrafficCustom(dgnet));
  PetscCall(DGNetworkProject(dgnet, dgnet->X, 0.0));
  if (viewglvis && dgnet->view) { PetscCall(DGNetworkMonitorView_Glvis_NET(monitor_gl, dgnet->X)); }
  PetscCall(VecGetSize(dgnet->X, &systemsize));
  PetscCall(PetscPrintf(comm, "\nWe have %" PetscInt_FMT " Dofs\n\n", systemsize));
  /* Create a time-stepping object */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetApplicationContext(ts, dgnet));

  switch (rhsversion) {
  case 0:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS, dgnet));
    break;
  case 1:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_V2, dgnet));
    break;
  default:
  case 2:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_V3, dgnet));
    break;
  }

  PetscCall(TSSetType(ts, TSSSP));
  PetscCall(TSSetMaxTime(ts, maxtime));
  PetscCall(TSSetMaxSteps(ts, 10));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, dgnet->cfl / dgnet->Mx / (2 * maxorder + 1)));
  PetscCall(TSSetFromOptions(ts)); /* Take runtime options */
  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      if (viewfullnet) {
        PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor_GLVis_NET, monitor_gl, NULL));
      } else {
        PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor_GLVis, monitor_gl, NULL));
      }
    } else {
      PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL));
    }
  }
  if (limit) {
    /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */
    PetscCall(DGNetlimiter(ts, 0, 0, &dgnet->X));
    PetscCall(TSSetPostStage(ts, DGNetlimiter));
  }

  PetscCall(TSSolve(ts, dgnet->X));
  if (savefinal) {
    PetscCall(PetscSNPrintf(outputfile, 256, "ex8output_P%i_%i", maxorder, dgnet->Mx));
    PetscCall(PetscViewerBinaryOpen(comm, outputfile, FILE_MODE_WRITE, &vecbinary));
    PetscCall(VecView(dgnet->X, vecbinary));
    PetscCall(PetscViewerDestroy(&vecbinary));
  }
  /* Clean up */
  if (dgnet->view && size == 1) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorDestroy_Glvis(&monitor_gl));
    } else {
      PetscCall(DGNetworkMonitorDestroy(&monitor));
    }
  }
  PetscCall(RiemannSolverDestroy(&dgnet->physics.rs));
  PetscCall(PetscFree(dgnet->physics.order));
  PetscCall(DGNetworkDestroy(dgnet)); /* Destroy all data within the network and within dgnet */
  PetscCall(DMDestroy(&dgnet->network));
  PetscCall(PetscFree(dgnet));

  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
}
