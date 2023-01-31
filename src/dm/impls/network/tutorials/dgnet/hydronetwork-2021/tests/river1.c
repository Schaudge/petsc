static char help[] = "This example demonstrates the use of DMNetwork \n\n";

/*
  Example:
    ./river1 -ts_max_steps 10 -case 0 -wash_dmnetwork_view
    ./river1 -ts_max_steps 10 -case 1 -wat_dmnetwork_view
    ./river1 -ts_max_steps 10 -case 2 -wat_dmnetwork_view
    mpiexec -n <np> ./river1 -ts_max_steps 10 -case -1 -dmnetwork_view_distributed
    mpiexec -n <np> ./river1 -ts_max_steps 10 -f ../cases/master_small.inp -nsubnet 3 -petscpartitioner_type simple -dmnetwork_view_distributed

    ./river1 -ts_max_steps 100 -case 0 -river_monitor 0-3 -wash_view
    ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 7.0 -river_monitor 0 -case 3 -subcase 1
    ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 2.5 -river_monitor 0 -case 3 -subcase 2
    ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 4.0 -river_monitor 0 -case 3 -subcase 3
    ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 4.0 -river_monitor 0 -case 3 -subcase 4
    ./river1 -ts_max_steps 4000 -ts_dt .001 -ts_max_time 5.0 -river_monitor 0 -case 3 -subcase 5

    ./river1 -river_monitor 0,1,2 -ts_max_steps 1000 -f ../cases/sample2.inp -qmin -1.0 -qmax 5.0 -hmin 0.0 -hmax 1.0
    ./river1 -river_monitor 2,3,4,5 -ts_max_steps 1500 -f ../cases/sample2.inp -qmin -1.0 -qmax 8.0 -hmin 0.0 -hmax 1.0 -river_monitor_q //fail at step 1407!

    ./river1 -test_mscale -ts_max_steps 10
*/

#include "../src/wash.h"

int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  Wash              wash;
  Junction          junctions, junction;
  PetscInt          KeyJunction, KeyRiver, KeyPump;
  PetscInt          i, e, vkey, type, k, eStart, eEnd;
  DM                networkdm;
  PetscMPIInt       size, rank;
  Vec               X;
  TS                ts;
  PetscBool         moni = PETSC_FALSE, view = PETSC_FALSE, parseflg, *riv, moni_q = PETSC_FALSE, moni_h = PETSC_FALSE;
  PetscInt          washCase, nv, ne, subnet, nsubnet = 1;
  DMNetworkMonitor  monitor;
  TSConvergedReason reason;
  River             river, rivers;
  PetscReal         ftime;
  Pump              pumps;
  PetscInt          ntmaxi, *riv_tmp, nriv;
  char              filename[100][PETSC_MAX_PATH_LEN], filename0[PETSC_MAX_PATH_LEN] = "", fmaster[PETSC_MAX_PATH_LEN] = "", fsmall[PETSC_MAX_PATH_LEN] = "";
  const PetscInt   *cone, *vtx, *edges;
  PetscInt          vfrom, vto;
  PetscReal         elevUs, elevDs;
  PetscInt         *numVertices, *numEdges, **edgelists;
  WashSubnet        washsubnet;
  PetscLogStage     stages[3];
  PetscReal         dx_min = 10.0, dx_max = 0.0, dx; /* used by wash->test_mscale */

  ierr = PetscInitialize(&argc, &argv, "", help);
  if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
  CHKERRQ(ierr);

  /* Register various stages for profiling */
  ierr = PetscLogStageRegister("Wash setup", &stages[0]);
  CHKERRQ(ierr);
  ierr = PetscLogStageRegister("TS step", &stages[1]);
  CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Free space", &stages[2]);
  CHKERRQ(ierr);

  /* Create and setup network */
  /*--------------------------*/
  /* Register various stages for profiling */
  ierr = PetscLogStagePush(stages[0]);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-nsubnet", &nsubnet, NULL);
  CHKERRQ(ierr);

  ierr = PetscCalloc3(nsubnet, &numEdges, nsubnet, &numVertices, nsubnet, &edgelists);
  CHKERRQ(ierr);

  ierr = DMNetworkCreate(PETSC_COMM_WORLD, &networkdm);
  CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)networkdm, "wash_");
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(networkdm);
  CHKERRQ(ierr);

  /* Register the physic components in the network */
  ierr = DMNetworkRegisterComponent(networkdm, "junctionstruct", sizeof(struct _p_Junction), &KeyJunction);
  CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm, "riverstruct", sizeof(struct _p_River), &KeyRiver);
  CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm, "pumpstruct", sizeof(struct _p_Pump), &KeyPump);
  CHKERRQ(ierr);

  /* Set the wash test case */
  washCase = 0; /* default case */
  ierr     = PetscOptionsGetInt(NULL, NULL, "-case", &washCase, NULL);
  CHKERRQ(ierr);

  /* Read files from a screen (runtime) or from ../cases/master_.inp */
  parseflg = PETSC_FALSE;
  ierr     = PetscOptionsGetString(NULL, NULL, "-f", filename0, PETSC_MAX_PATH_LEN, &parseflg);
  CHKERRQ(ierr);

  FILE *fp = fopen(filename0, "rb");
  if (parseflg && fp) {
    /* Get input filename[] from ../cases/master_.inp */
    PetscInt subcase = 0;
    washCase         = -1;
    ierr             = PetscOptionsGetInt(NULL, NULL, "-subcase", &subcase, NULL);
    CHKERRQ(ierr);

    ierr = PetscStrcpy(fsmall, "../cases/master_small.inp");
    CHKERRQ(ierr);
    ierr = PetscStrcpy(fmaster, "../cases/master.inp");
    CHKERRQ(ierr);
    ierr = PetscStrcpy(filename[0], filename0);
    CHKERRQ(ierr);

    /* All processes read filename[i], i=0,...,nsubnet-1 */
    if (strcmp(filename[0], fmaster) == 0) {
      ierr = WashReadInputFile(nsubnet, filename);
      CHKERRQ(ierr);
    } else if (strcmp(filename[0], fsmall) == 0) {
      ierr = WashReadInputFile(nsubnet, filename);
      CHKERRQ(ierr);
    }
    fclose(fp);
  }

  /* Create wash */
  ierr = WashCreate(PETSC_COMM_WORLD, nsubnet, 0, &wash);
  CHKERRQ(ierr);
  wash->dm = networkdm;

  /* Set wash subnetworks */
  for (i = 0; i < nsubnet; i++) {
    if (size < nsubnet) {
      /* proc[0] reads i-th subnetwork input */
      ierr = WashAddSubnet(i, washCase, filename[i], 0, wash);
      CHKERRQ(ierr);
    } else {
      /* proc[i] reads i-th subnetwork input */
      ierr = WashAddSubnet(i, washCase, filename[i], i, wash);
      CHKERRQ(ierr);
    }
    washsubnet     = (WashSubnet)wash->subnet[i];
    numEdges[i]    = washsubnet->nedge;
    numVertices[i] = washsubnet->nvertex;
    edgelists[i]   = washsubnet->edgelist;

    if (numEdges[i] || numVertices[i]) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]...%d -- washCase, %d-the Washsubnet(), numE %d, numV %d\n", rank, washCase, i, numEdges[i], numVertices[i]));
  }

  wash->keyJunction = KeyJunction;
  wash->keyRiver    = KeyRiver;
  wash->keyPump     = KeyPump;

  /* Set local number of vertices and edges */
  ierr = DMNetworkSetNumSubNetworks(networkdm, PETSC_DECIDE, nsubnet);
  CHKERRQ(ierr);
  for (i = 0; i < nsubnet; i++) {
    PetscInt netNum = -1;
    ierr            = DMNetworkAddSubnetwork(networkdm, NULL, numEdges[i], edgelists[i], &netNum);
    CHKERRQ(ierr);
  }
  if (size < 10) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] nV %d, nE %d, nsubnet %d\n", rank, numVertices[0], numEdges[0], nsubnet);
    CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);
  }

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);
  CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "...DMNetworkLayoutSetUp() is done\n");
    CHKERRQ(ierr);
  }

  ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
  CHKERRQ(ierr);
  ne = eEnd - eStart; /* Total local num of edges */

  /* Get wash runtime options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Wash options", "");
  if (size == 1) { /* Monitor selected rivers graphically */
    ierr = PetscMalloc2(ne, &riv, ne, &riv_tmp);
    CHKERRQ(ierr);
    nriv = ne;
    ierr = PetscOptionsGetIntArray(NULL, NULL, "-river_monitor", riv_tmp, &nriv, &moni);
    CHKERRQ(ierr);
    if (moni) {
      moni_q = PETSC_TRUE;
      moni_h = PETSC_TRUE;

      for (i = 0; i < ne; i++) riv[i] = PETSC_FALSE;
      for (i = 0; i < nriv; i++) riv[riv_tmp[i]] = PETSC_TRUE;
      ierr = DMNetworkMonitorCreate(networkdm, &monitor);
      CHKERRQ(ierr);

      /* enable viewing Q or H only */
      ierr = PetscOptionsGetBool(NULL, NULL, "-river_monitor_q", &moni_q, &parseflg);
      CHKERRQ(ierr);
      if (parseflg && moni_q) moni_h = PETSC_FALSE;
      ierr = PetscOptionsGetBool(NULL, NULL, "-river_monitor_h", &moni_h, &parseflg);
      CHKERRQ(ierr);
      if (parseflg && moni_h) moni_q = PETSC_FALSE;
    }
  }
  PetscOptionsEnd();

  /* Add network components - only process[0] has data to add */
  /* ---------------------------------------------------------*/
  k = 0;
  for (subnet = 0; subnet < nsubnet; subnet++) {
    washsubnet = (WashSubnet)wash->subnet[subnet];
    junctions  = washsubnet->junction;
    rivers     = washsubnet->river;
    pumps      = washsubnet->pump;

    ierr = DMNetworkGetSubnetwork(networkdm, subnet, &nv, &ne, &vtx, &edges);
    CHKERRQ(ierr);
    if (!rank) {
      if (ne != washsubnet->nriver) { printf("[%d] Warning: subnet[%d] ne %d != wash->nriver %d\n", rank, subnet, ne, washsubnet->nriver); }
    }

    for (i = 0; i < washsubnet->nriver; i++) {
      /* Add River component to edges, rivers=wash->river, and number of variables */
      ierr = DMNetworkAddComponent(networkdm, edges[i], KeyRiver, &rivers[i], 2 * rivers[i].ncells);
      CHKERRQ(ierr);

      if (size == 1 && moni && riv[k]) { /* Add monitor */
        PetscReal ymin, ymax;
        if (moni_q) {
          ymin = wash->QMin;
          ymax = wash->QMax;
          ierr = PetscOptionsGetReal(NULL, NULL, "-qmin", &ymin, NULL);
          CHKERRQ(ierr);
          ierr = PetscOptionsGetReal(NULL, NULL, "-qmax", &ymax, NULL);
          CHKERRQ(ierr);
          ierr = DMNetworkMonitorAdd(monitor, "River Q", edges[i], rivers[i].ncells, 0, 2, 0.0, rivers[i].length, ymin, ymax, PETSC_TRUE);
          CHKERRQ(ierr);
        }
        if (moni_h) {
          ymin = wash->HMin;
          ymax = wash->HMax;
          ierr = PetscOptionsGetReal(NULL, NULL, "-hmin", &ymin, NULL);
          CHKERRQ(ierr);
          ierr = PetscOptionsGetReal(NULL, NULL, "-hmax", &ymax, NULL);
          CHKERRQ(ierr);
          ierr = DMNetworkMonitorAdd(monitor, "River H", edges[i], rivers[i].ncells, 1, 2, 0.0, rivers[i].length, ymin, ymax, PETSC_TRUE);
          CHKERRQ(ierr);
        }
      }
      k++;
    }

    e = 0;
    while (i < ne) { /* Add Pump component to the remaining edges ??? */
      ierr = DMNetworkAddComponent(networkdm, edges[i], KeyPump, &pumps[e++], 0);
      CHKERRQ(ierr);
      i++;
    }

    /* Add Junction component to all vertices */
    for (i = 0; i < nv; i++) {
      ierr = DMNetworkAddComponent(networkdm, vtx[i], KeyJunction, &junctions[i], 2);
      CHKERRQ(ierr);
    }
  }
  if (size == 1) {
    ierr = PetscFree2(riv, riv_tmp);
    CHKERRQ(ierr);
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);
  CHKERRQ(ierr);
  wash->dm = networkdm;
  ierr     = WashCleanUp(wash, edgelists);
  CHKERRQ(ierr);
  ierr = PetscFree3(numEdges, numVertices, edgelists);
  CHKERRQ(ierr);

  /* Network partitioning and distribution of data */
  ierr = DMNetworkDistribute(&networkdm, 0);
  CHKERRQ(ierr);
  wash->dm = networkdm;
  ierr     = PetscOptionsGetBool(NULL, NULL, "-junction_view", &view, NULL);
  CHKERRQ(ierr);
  if (view) {
    /* Display JunctionType */
    ierr = WashJunctionView(wash);
    CHKERRQ(ierr);
  }

  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "...DMNetworkDistribute() is done by nproc %d\n", size);
    CHKERRQ(ierr);
  }

  /* SetUp physics (e.g., pipe or river) -- each process only sets its own physics */
  /*-------------------------------------------------------------------------------*/
  for (subnet = 0; subnet < nsubnet; subnet++) {
    ierr = DMNetworkGetSubnetwork(networkdm, subnet, &nv, &ne, &vtx, &edges);
    CHKERRQ(ierr);
    for (i = 0; i < ne; i++) {
      e    = edges[i];
      ierr = DMNetworkGetComponent(networkdm, e, 0, &type, (void **)&river, NULL);
      CHKERRQ(ierr);
      if (type != KeyRiver) continue; /* skip if pump */

      ierr = RiverSetParameters(river, e);
      CHKERRQ(ierr);
      ierr = RiverSetUp(river);
      CHKERRQ(ierr);

      if (wash->test_mscale) {
        /* compute dx_min and dx_max */
        dx = river->length / river->ncells;
        if (dx < dx_min) dx_min = dx;
        if (dx > dx_max) dx_max = dx;
      }

      wash->nnodes_loc += river->ncells; /* local total num of nodes, will be used by RiverView() */

      ierr = DMNetworkGetConnectedVertices(networkdm, e, &cone);
      CHKERRQ(ierr); /* get vertexes index */
      vfrom = cone[0];
      vto   = cone[1];

      ierr = DMNetworkGetComponent(networkdm, vfrom, 0, &vkey, (void **)&junction, NULL);
      CHKERRQ(ierr); /* get upsteam elevation */
      elevUs = junction->elev;
      ierr   = DMNetworkGetComponent(networkdm, vto, 0, &vkey, (void **)&junction, NULL);
      CHKERRQ(ierr); /* get downstream elevation */
      elevDs = junction->elev;
      ierr   = RiverSetElevation(river, elevUs, elevDs);
      CHKERRQ(ierr); /* set elevation at each cell */
    }
  }
  if (!rank && wash->test_mscale) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "...dx_min/max: %g, %g\n", dx_min, dx_max);
    CHKERRQ(ierr);
  }

  /* Create vectors */
  ierr = WashCreateVecs(wash);
  CHKERRQ(ierr);
  X    = wash->X;
  ierr = VecGetSize(X, &i);
  CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Total variables %d\n", i);
    CHKERRQ(ierr);
  }

  /* Set initial solution */
  ierr = WashSetInitialSolution(networkdm, wash);
  CHKERRQ(ierr);

  /* Setup solver ts                                       */
  /*-------------------------------------------------------*/
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);
  CHKERRQ(ierr);
  ierr = WashTSSetUp(wash, ts);
  CHKERRQ(ierr);
  if (size == 1 && moni) {
    ierr = TSMonitorSet(ts, TSDMNetworkMonitor, monitor, NULL);
    CHKERRQ(ierr);
  }

  ierr = TSSetSolution(ts, X);
  CHKERRQ(ierr);
  ierr = TSSetUp(ts);
  CHKERRQ(ierr);
  ierr = PetscLogStagePop();
  CHKERRQ(ierr);

  /* Time steps           */
  /* -------------------- */
  ierr = PetscLogStagePush(stages[1]);
  CHKERRQ(ierr);
  ierr = TSSolve(ts, X);
  CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &ftime);
  CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &ntmaxi);
  CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason);
  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %d steps\n", TSConvergedReasons[reason], (double)ftime, ntmaxi);
  CHKERRQ(ierr);

  /* View solution vector X */
  view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-X_view", &view, NULL);
  CHKERRQ(ierr);
  if (view) {
    if (!rank) printf("ts X:\n");
    ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);
  }

  /* View solution [q h u=q/h] */
  /* ------------------------- */
  view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-wash_view", &view, NULL);
  CHKERRQ(ierr);
  if (view) {
    ierr = WashVecView(wash, X);
    CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();
  CHKERRQ(ierr);

  /* Free spaces */
  /* ----------- */
  ierr = PetscLogStagePush(stages[2]);
  ierr = TSDestroy(&ts);
  CHKERRQ(ierr);
  ierr = WashDestroyVecs(wash);
  CHKERRQ(ierr);

  ierr = WashDestroy(wash);
  CHKERRQ(ierr);

  if (size == 1 && moni) {
    ierr = DMNetworkMonitorDestroy(&monitor);
    CHKERRQ(ierr);
  }
  ierr = DMDestroy(&networkdm);
  CHKERRQ(ierr);
  ierr = PetscLogStagePop();
  CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
