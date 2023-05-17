#include "dgnet.h"
#include <petscdraw.h>
#include "hydronetwork-2021/src/wash.h"
#include "petscdmnetwork.h"
#include "petscerror.h"
#include "petscmath.h"
#include "petscnetrs.h"
#include "petscsystypes.h"

PetscLogEvent  DGNET_SetUP;
PetscErrorCode WashDestroy_DGNet(Wash wash)
{
  PetscErrorCode ierr;
  PetscInt       subnet, nsubnet = wash->nsubnet;

  PetscFunctionBegin;
  for (subnet = 0; subnet < nsubnet; subnet++) {
    ierr = PetscFree(wash->subnet[subnet]);
    CHKERRQ(ierr);
  }
  ierr = PetscFree(wash->subnet);
  CHKERRQ(ierr);
  ierr = PetscFree(wash);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkCreate(DGNetwork dgnet, PetscInt networktype, PetscInt Mx)
{
  PetscInt      nfvedge;
  PetscMPIInt   rank;
  PetscInt      i, j, k, m, n, field, numVertices, numEdges;
  PetscInt     *edgelist;
  DGNETJunction junctions = NULL;
  EdgeFE        fvedges   = NULL;
  PetscInt      dof       = dgnet->physics.dof;

  PetscFunctionBegin;
  PetscCall(PetscLogEventRegister("DGNetRHS_Edge", TS_CLASSID, &DGNET_Edge_RHS));
  PetscCall(PetscLogEventRegister("DGNetRHS_Vert", TS_CLASSID, &DGNET_RHS_Vert));
  PetscCall(PetscLogEventRegister("DGNetRHS_Comm", TS_CLASSID, &DGNET_RHS_COMM));
  PetscCall(PetscLogEventRegister("DGNetLimiter", TS_CLASSID, &DGNET_Limiter));
  PetscCall(PetscLogEventRegister("DGNetSetUp", TS_CLASSID, &DGNET_SetUP));

  PetscCall(MPI_Comm_rank(dgnet->comm, &rank));
  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;

  /* proc[0] creates a sequential dgnet and edgelist    */
  /* Set global number of fvedges, edges, and junctions */
  /*-------------------------------------------------*/
  switch (networktype) {
  case -3: /* read from h5 file */
    /* EPANet Parser from hydronetwork-2021. Rework into proper parser later */
  case -2:
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    /* only read in on processor 0 */
    if (!rank) {
      Wash       wash;
      PetscBool  parseflg = PETSC_FALSE;
      char       filename[100][PETSC_MAX_PATH_LEN], filename0[PETSC_MAX_PATH_LEN] = "", fmaster[PETSC_MAX_PATH_LEN] = "", fsmall[PETSC_MAX_PATH_LEN] = "";
      PetscInt   washCase;
      WashSubnet washsubnet;

      /* Read files from a screen (runtime) or from ../cases/master_.inp */
      PetscCall(PetscOptionsGetString(NULL, NULL, "-f", filename0, PETSC_MAX_PATH_LEN, &parseflg));
      FILE *fp = fopen(filename0, "rb");
      if (parseflg && fp) {
        /* Get input filename[] from ../cases/master_.inp */
        PetscInt subcase = 0;
        washCase         = -1;
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-subcase", &subcase, NULL));

        PetscCall(PetscStrcpy(fsmall, "../hydronetwork-2021/cases/master_small.inp"));
        PetscCall(PetscStrcpy(fmaster, "../hydronetwork-2021/cases/master.inp"));
        PetscCall(PetscStrcpy(filename[0], filename0));

        /* All processes read filename[i], i=0,...,nsubnet-1 */
        if (strcmp(filename[0], fmaster) == 0) {
          PetscCall(WashReadInputFile(1, filename));
        } else if (strcmp(filename[0], fsmall) == 0) {
          PetscCall(WashReadInputFile(1, filename));
        }
        fclose(fp);
      }
      PetscCall(WashCreate(PETSC_COMM_WORLD, 1, 0, &wash));
      PetscCall(WashAddSubnet(0, washCase, filename[0], 0, wash));

      washsubnet  = wash->subnet[0];
      numEdges    = washsubnet->nedge;
      numVertices = washsubnet->nvertex;

      PetscCheck(washsubnet->npump == 0, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Can only handle EPANet files containing rivers, cannot handle pumps. Mesh contained %" PetscInt_FMT " pumps", washsubnet->npump);
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));
      PetscCall(PetscArraycpy(edgelist, washsubnet->edgelist, 2 * numEdges));
      for (i = 0; i < numEdges; i++) {
        fvedges[i].length = washsubnet->river[i].length;
        if (dgnet->dx <= 0) dgnet->dx = 1;
        fvedges[i].nnodes = (PetscInt)PetscCeilReal(washsubnet->river[i].length / dgnet->dx);
        if (fvedges[i].nnodes < 3) fvedges[i].nnodes = 3; /* minimum requirement for limiting to work */
      }

      PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d -- washCase, numE %d, numV %d\n", washCase, numEdges, numVertices));

      /* no coordinate from EPANET so do nothing with junction coordinates here */

      /* wash has really really silly destruction routines */
      PetscCall(WashCleanUp(wash, &wash->subnet[0]->edgelist));

      PetscCall(WashDestroy_DGNet(wash));
    }
    break;
  /* grid graph with entrance */
  /* ndaughters governs the depth of the network */
  case -1:
    m = dgnet->ndaughters;
    n = dgnet->ndaughters;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = m * n + 2;
      numEdges    = (m - 1) * n + (n - 1) * m + 2;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      /* Enter Branch */
      edgelist[0] = 0;
      edgelist[1] = 1;
      /* Exit Branch */
      edgelist[2 * numEdges - 1] = numVertices - 2;
      edgelist[2 * numEdges - 2] = numVertices - 1;

      /* Grid Graph Generation */
      k = 2;
      for (j = 0; j < n - 1; ++j) {
        for (i = 0; i < m - 1; ++i) {
          edgelist[k++] = i + j * m + 1;
          edgelist[k++] = i + j * m + 2;
          edgelist[k++] = i + j * m + 1;
          edgelist[k++] = i + (j + 1) * m + 1;
        }
      }
      for (j = 0; j < n - 1; j++) {
        edgelist[k++] = (j + 1) * m;
        edgelist[k++] = (j + 2) * m;
      }
      for (i = 0; i < m - 1; ++i) {
        edgelist[k++] = i + (n - 1) * m + 1;
        edgelist[k++] = i + (n - 1) * m + 2;
      }

      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      /* vertex */
      /* embed them as a shifted grid like 
                --v2--
        v0---v1<--v3-->v4---v5 

        for the depth 2 case.  */

      /* Edge */
      fvedges[0].nnodes = (m + 1) * Mx;
      fvedges[0].length = (m + 1) * dgnet->length;

      for (i = 1; i < numEdges; ++i) {
        fvedges[i].nnodes = Mx;
        fvedges[i].length = dgnet->length;
      }

      PetscReal xx, yy;
      for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
          xx                         = j * dgnet->length;
          yy                         = i * dgnet->length;
          junctions[i + j * m + 1].x = PetscCosReal(PETSC_PI / 4) * xx + PetscSinReal(PETSC_PI / 4) * yy;
          junctions[i + j * m + 1].y = -PetscSinReal(PETSC_PI / 4) * xx + PetscCosReal(PETSC_PI / 4) * yy;
        }
      }
      junctions[0].x               = -fvedges[0].length;
      junctions[0].y               = 0;
      junctions[numVertices - 1].x = junctions[numVertices - 2].x + dgnet->length;
      junctions[numVertices - 1].y = 0;
    }
    break;
  case 0:
    /* Case 0: */
    /* =================================================
      (OUTFLOW) v0 --E0--> v1--E1--> v2 --E2-->v3 (OUTFLOW)
      ====================================================  */
    nfvedge = 3;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = nfvedge;
      numEdges    = nfvedge + 1;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      edgelist[0] = 0;
      edgelist[1] = 1;
      edgelist[2] = 1;
      edgelist[3] = 2;
      edgelist[4] = 2;
      edgelist[5] = 3;
      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));

      for (i = 0; i < numVertices; i++) { junctions[i].x = i * 1.0 / 3.0 * 50.0; }
      /* Edge */
      fvedges[0].nnodes = Mx;
      fvedges[1].nnodes = Mx;
      fvedges[2].nnodes = Mx;

      for (i = 0; i < numEdges; i++) { fvedges[i].length = 50.0; }
    }
    break;
  case 1:
    /* Case 1: */
    /* =================================================
      (OUTFLOW) v0 --E0--> v1 (OUTFLOW)
      ====================================================  */
    nfvedge = 1;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = nfvedge;
      numEdges    = nfvedge + 1;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      for (i = 0; i < numEdges; i++) {
        edgelist[2 * i]     = i;
        edgelist[2 * i + 1] = i + 1;
      }
      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      /* vertex */

      for (i = 0; i < numVertices; i++) {
        junctions[i].x = i * 1.0 * 50.0;
        junctions[i].y = 0.;
      }
      /* Edge */
      fvedges[0].nnodes = Mx;

      for (i = 0; i < numEdges; i++) { fvedges[i].length = 50.0; }
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
    nfvedge = 3;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = nfvedge;
      numEdges    = nfvedge + 1;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      edgelist[0] = 1;
      edgelist[1] = 0;
      edgelist[2] = 2;
      edgelist[3] = 1;
      edgelist[4] = 3;
      edgelist[5] = 2;
      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));

      for (i = 0; i < numVertices; i++) {
        junctions[i].x = (3 - i) * 1.0 / 3.0 * 50.0;
        junctions[i].y = 0.;
      }
      /* Edge */
      fvedges[0].nnodes = Mx;
      fvedges[1].nnodes = dgnet->hratio * Mx;
      fvedges[2].nnodes = Mx;

      for (i = 0; i < numEdges; i++) { fvedges[i].length = 50.0; }
    }
    break;
  case 3:
    /* Case 3: (Image is for the case we ndaughers = 2. The number of out branches is given by dgnet->ndaughers */
    /* =================================================
    (OUTFLOW) v1 --E0--> v0-E1--> v2  (OUTFLOW)
                          |
                          E2
                          |
                          \/
                          v3 (OUTFLOW) 
    ====================================================
    This tests the coupling condition for the simple case */
    nfvedge = dgnet->ndaughters + 1;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = nfvedge;
      numEdges    = nfvedge + 1;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      /* Parent Branch (pointing in) */
      edgelist[0] = 0;
      edgelist[1] = 1;
      /* Daughter Branches (pointing out from v1) */
      for (i = 1; i < dgnet->ndaughters + 1; ++i) {
        edgelist[2 * i]     = 0;
        edgelist[2 * i + 1] = i + 1;
      }
      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      /* vertex */

      /* place them equispaced on the circle of radius length */
      PetscReal theta;
      theta = 2. * PETSC_PI / (dgnet->ndaughters + 1);
      /*daughters */
      for (i = 1; i < dgnet->ndaughters + 2; ++i) {
        junctions[i].x = dgnet->length * PetscCosReal(theta * (i - 1) + PETSC_PI);
        junctions[i].y = dgnet->length * PetscSinReal(theta * (i - 1) + PETSC_PI);
      }
      /* center */
      junctions[0].x = 0.0;
      junctions[0].y = 0.0;

      /* Edge */
      fvedges[0].nnodes = Mx;
      for (i = 1; i < dgnet->ndaughters + 1; ++i) { fvedges[i].nnodes = Mx; }

      for (i = 0; i < numEdges; i++) { fvedges[i].length = dgnet->length; }
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
    =================================================
    */
    break;
  case 6:
    /* Case 6: Periodic Boundary conditions
    =================================================
       v1 --E1--> v0--E0--> v1
    ================================================
          used for convergence tests */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = 2;
      numEdges    = 2;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      edgelist[0] = 0;
      edgelist[1] = 1;
      edgelist[2] = 1;
      edgelist[3] = 0;

      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      /* vertex */

      junctions[0].x = -5.0;
      junctions[1].x = 5.0;
      /* Edge */
      for (i = 0; i < numEdges; ++i) {
        fvedges[i].nnodes = Mx;
        fvedges[i].length = 5.0;
      }
    }
    break;
  case 7:

    /* double linked grid graph. Same as -1 but double linked and no entrance/exit 

      */
    m = dgnet->ndaughters;
    n = dgnet->ndaughters;
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = m * n;
      numEdges    = 2 * ((m - 1) * n + (n - 1) * m);
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      /* Grid Graph Generation */
      k = 0;
      for (j = 0; j < n - 1; ++j) {
        for (i = 0; i < m - 1; ++i) {
          edgelist[k++] = i + j * m;
          edgelist[k++] = i + j * m + 1;

          edgelist[k++] = i + j * m + 1;
          edgelist[k++] = i + j * m;

          edgelist[k++] = i + j * m;
          edgelist[k++] = i + (j + 1) * m;

          edgelist[k++] = i + (j + 1) * m;
          edgelist[k++] = i + j * m;
        }
      }
      for (j = 0; j < n - 1; j++) {
        edgelist[k++] = (j + 1) * m - 1;
        edgelist[k++] = (j + 2) * m - 1;

        edgelist[k++] = (j + 2) * m - 1;
        edgelist[k++] = (j + 1) * m - 1;
      }
      for (i = 0; i < m - 1; ++i) {
        edgelist[k++] = i + (n - 1) * m;
        edgelist[k++] = i + (n - 1) * m + 1;

        edgelist[k++] = i + (n - 1) * m + 1;
        edgelist[k++] = i + (n - 1) * m;
      }

      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      for (i = 0; i < numEdges; ++i) {
        fvedges[i].nnodes = Mx;
        fvedges[i].length = dgnet->length;
      }

      PetscReal xx, yy;
      for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
          xx                     = j * dgnet->length;
          yy                     = i * dgnet->length;
          junctions[i + j * m].x = xx;
          junctions[i + j * m].y = yy;
        }
      }
    }
    break;
  case 8:
    /* Traffic Circle (Pattern from Jingmei and Bennedettos paper ) */

    /* each "spoke" alterantes between in and out and the inner loop goes counter-clockwise */
    n = dgnet->ndaughters; /* number of inner edges */
    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = 2 * n;
      numEdges    = 2 * n;
      PetscCall(PetscCalloc1(2 * numEdges, &edgelist));

      /* inner loop  */
      k = 0;
      for (i = 0; i < n - 1; ++i) {
        edgelist[k++] = i;
        edgelist[k++] = i + 1;
      }
      edgelist[k++] = n - 1;
      edgelist[k++] = 0;

      /* spokes */
      for (i = 0; i < n; ++i) {
        if (i % 2) {
          edgelist[k++] = i;
          edgelist[k++] = i + n;
        } else {
          edgelist[k++] = i + n;
          edgelist[k++] = i;
        }
      }

      /* Add network components */
      /*------------------------*/
      PetscCall(PetscCalloc2(numVertices, &junctions, numEdges, &fvedges));
      for (i = 0; i < numEdges; ++i) {
        fvedges[i].nnodes = Mx;
        fvedges[i].length = dgnet->length;
      }

      PetscReal xx, yy;
      PetscReal circumradius = dgnet->length / PetscSinReal(PETSC_PI / (PetscReal)n);
      PetscReal angle        = 2 * PETSC_PI / (PetscReal)n;
      PetscReal norm;

      for (i = 0; i < n; ++i) {
        xx             = circumradius * PetscCosReal(angle * i + PETSC_PI / 2);
        yy             = circumradius * PetscSinReal(angle * i + PETSC_PI / 2);
        junctions[i].x = xx;
        junctions[i].y = yy;
      }
      norm = PetscSqrtReal(PetscSqr(junctions[0].x) + PetscSqr(junctions[0].y));

      for (i = 0; i < n; ++i) {
        xx                 = junctions[i].x / norm * dgnet->length + junctions[i].x;
        yy                 = junctions[i].y / norm * dgnet->length + junctions[i].y;
        junctions[i + n].x = xx;
        junctions[i + n].y = yy;
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "not done yet");
  }

  dgnet->junction = junctions;
  dgnet->edgefe   = fvedges;

  PetscCall(DMNetworkCreate(dgnet->comm, &dgnet->network));
  PetscCall(DMNetworkSetNumSubNetworks(dgnet->network, PETSC_DECIDE, 1));
  PetscCall(DMNetworkAddSubnetwork(dgnet->network, NULL, numEdges, edgelist, NULL));
  PetscCall(DMNetworkLayoutSetUp(dgnet->network));
  PetscCall(PetscFree(edgelist));
  /*
    TODO : Make all this stuff its own class

    NOTE: Should I have tensor interface for petsc? Would be really useful for all the tabulation tensors an d
    etc I'm using. Not to mention that effectively the DG solution is a tensor (3rd order )
    element x basis x field

    something to consider....
  */

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkSetComponents(DGNetwork dgnet)
{
  PetscInt      f, e, v, eStart, eEnd, vStart, vEnd, dof = dgnet->physics.dof;
  PetscInt      KeyEdge, KeyJunction;
  PetscInt      dmsize = 0, numdof = 0;
  EdgeFE        edgefe;
  DGNETJunction junction;
  MPI_Comm      comm = dgnet->comm;
  PetscMPIInt   size, rank;

  PetscFunctionBegin;
  PetscLogEventBegin(DGNET_SetUP, 0, 0, 0, 0);
  PetscCall(MPI_Comm_rank(comm, &rank));
  PetscCall(MPI_Comm_size(comm, &size));
  for (f = 0; f < dof; f++) { numdof += dgnet->physics.order[f] + 1; }

  /* Set up the network layout */

  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
  PetscCall(DMNetworkRegisterComponent(dgnet->network, "junctionstruct", sizeof(struct _p_DGNETJunction), &KeyJunction));
  PetscCall(DMNetworkRegisterComponent(dgnet->network, "fvedgestruct", sizeof(struct _p_EdgeFE), &KeyEdge));

  /* Add FVEdge component to all local edges. Note that as we have
     yet to distribute the network, all data is on proc[0]. */
  for (e = eStart; e < eEnd; e++) {
    /*
      TODO : Remove EdgeFE from DGNet, refactor how to construct the FE network. THis is definitely a hacky way to do it.
    */
    edgefe = &dgnet->edgefe[e - eStart];
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
    junction = &dgnet->junction[v - vStart];
    PetscCall(DMNetworkAddComponent(dgnet->network, v, KeyJunction, junction, 0));
  }
  PetscCall(DMSetUp(dgnet->network));
  PetscLogEventEnd(DGNET_SetUP, 0, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkAddMonitortoEdges(DGNetwork dgnet, DGNetworkMonitor monitor)
{
  PetscInt e, eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  if (monitor) {
    for (e = eStart; e < eEnd; e++) { PetscCall(DGNetworkMonitorAdd(monitor, e, PETSC_DECIDE, PETSC_DECIDE, dgnet->ymin, dgnet->ymax, PETSC_FALSE)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkAddMonitortoEdges_Glvis(DGNetwork dgnet, DGNetworkMonitor_Glvis monitor, PetscViewerGLVisType type)
{
  PetscInt e, eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  if (monitor) {
    for (e = eStart; e < eEnd; e++) { PetscCall(DGNetworkMonitorAdd_Glvis(monitor, e, "localhost", type)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkAddMonitortoEdges_Glvis_3D(DGNetwork dgnet, DGNetworkMonitor_Glvis monitor, PetscViewerGLVisType type)
{
  PetscInt e, eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  if (monitor) {
    for (e = eStart; e < eEnd; e++) { PetscCall(DGNetworkMonitorAdd_Glvis_3D(monitor, e, "localhost", type)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Only call after network is distirbuted. Rework some stuff otherwise... */
PetscErrorCode DGNetworkBuildDynamic(DGNetwork dgnet)
{
  PetscFunctionBegin;
  PetscCall(DGNetworkBuildEdgeDM(dgnet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkBuildEdgeDM(DGNetwork dgnet)
{
  PetscInt     e, i, dof = dgnet->physics.dof;
  PetscInt     eStart, eEnd, *numComp, *numDof, dim = 1, f;
  EdgeFE       edgefe;
  PetscReal    low[3] = {0, 0, 0}, upper[3] = {1, 1, 1};
  PetscSection section;

  PetscFunctionBegin;
  PetscLogEventBegin(DGNET_SetUP, 0, 0, 0, 0);
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  /* iterate through the edges and build the dmplex mesh for each edge */
  PetscCall(PetscMalloc2(dof, &numComp, dof * (dim + 1), &numDof));
  for (i = 0; i < dof * (dim + 1); ++i) numDof[i] = 0;
  for (i = 0; i < dof; ++i) numComp[i] = 1;

  /* all variables are stored at the cell level for DG (i.e edges in the 1d case here) */
  for (f = 0; f < dof; ++f) { numDof[f * (dim + 1) + dim] = dgnet->physics.order[f] + 1; }
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    upper[0] = edgefe->length;

    /* Anyway to turn off options for this? it will only work with dim 1 for the rest of the code */
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 1, PETSC_FALSE, &edgefe->nnodes, low, upper, NULL, PETSC_TRUE, &edgefe->dm));

    /* Create Field section */
    PetscCall(DMSetNumFields(edgefe->dm, dof));
    PetscCall(DMPlexCreateSection(edgefe->dm, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, &section));
    /*
      NOTE: I do not assign names to the field variables as I don't want every edge storing copies of the same field names.
      These are instead stored in the user provided physics ctx. Anywhere a name is needed, look there, they will be stored in the same
      order as the field order in this section.
    */
    PetscCall(DMSetLocalSection(edgefe->dm, section));
    PetscCall(PetscSectionDestroy(&section));
    PetscCall(DMSetUp(edgefe->dm));
  }
  PetscCall(PetscFree2(numComp, numDof));
  PetscLogEventEnd(DGNET_SetUP, 0, 0, 0, 0);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkBuildTabulation(DGNetwork dgnet)
{
  PetscInt   n, j, i, dof = dgnet->physics.dof, numunique, dim = 1;
  PetscInt  *deg, *temp_taborder;
  PetscReal *xnodes, *w, bdry[2] = {-1, 1}, *viewnodes;
  PetscBool  unique;

  PetscFunctionBegin;
  /* Iterate through the user provided orders for each field and build the taborder and fieldtotab arrays */
  PetscCall(PetscMalloc1(dof, &dgnet->fieldtotab));
  PetscCall(PetscMalloc1(dof, &temp_taborder));
  /* count number of unique field orders */
  numunique = 0;
  for (i = 0; i < dof; i++) {
    PetscCheck(dgnet->physics.order[i] >= 0, PetscObjectComm((PetscObject)dgnet), PETSC_ERR_USER_INPUT, "Order for Field %" PetscInt_FMT " is %" PetscInt_FMT ". Order cannot be negative", i, dgnet->physics.order[i]);
    /* Search through the current unique orders for a match */
    unique = PETSC_TRUE;
    for (j = 0; j < numunique; j++) {
      if (dgnet->physics.order[i] == temp_taborder[j]) {
        unique               = PETSC_FALSE;
        dgnet->fieldtotab[i] = j;
        break;
      }
    }
    if (unique) {
      dgnet->fieldtotab[i]     = numunique;
      temp_taborder[numunique] = dgnet->physics.order[i];
      numunique++;
    }
  }
  /* now we have the number of unique orders and what they are in fieldtotab (which is being reused here) */
  PetscCall(PetscMalloc1(numunique, &dgnet->taborder));
  dgnet->tabordersize = numunique;
  for (i = 0; i < dgnet->tabordersize; i++) { dgnet->taborder[i] = temp_taborder[i]; }
  PetscCheck(dgnet->tabordersize > 0, PetscObjectComm((PetscObject)dgnet), PETSC_ERR_COR, "Tabordersize is %" PetscInt_FMT "<1, this should not happen", dgnet->tabordersize);
  PetscCall(PetscFree(temp_taborder));
  PetscCall(PetscMalloc4(dgnet->tabordersize, &dgnet->LegEval, dgnet->tabordersize, &dgnet->Leg_L2, dgnet->tabordersize, &dgnet->LegEvalD, dgnet->tabordersize, &dgnet->LegEvaL_bdry));
  /* Internal Viewer Storage stuff (to be migrated elsewhere) */
  PetscCall(PetscMalloc2(dgnet->tabordersize, &dgnet->LegEval_equispaced, dgnet->tabordersize, &dgnet->numviewpts));
  /* Build Reference Quadrature (Single Quadrature for all fields (maybe generalize but not now) */
  PetscCall(PetscQuadratureCreate(dgnet->comm, &dgnet->quad));
  /* Find maximum ordeer */
  n = 0;
  for (i = 0; i < dgnet->tabordersize; i++) {
    if (n < dgnet->taborder[i] + 1) n = dgnet->taborder[i] + 1;
  }
  PetscCall(PetscMalloc1(n, &xnodes));
  PetscCall(PetscMalloc1(n, &w));
  PetscCall(PetscDTGaussQuadrature(n, -1, 1, xnodes, w));
  PetscCall(PetscQuadratureSetData(dgnet->quad, dim, 1, n, xnodes, w));
  PetscCall(PetscQuadratureSetOrder(dgnet->quad, 2 * n));
  PetscCall(PetscMalloc2(dof, &dgnet->pteval, dof * n, &dgnet->fluxeval));
  for (i = 0; i < dgnet->tabordersize; i++) {
    /* Build Reference Legendre Evaluations */
    PetscCall(PetscMalloc1(dgnet->taborder[i] + 1, &deg));
    PetscCall(PetscMalloc2(n * (dgnet->taborder[i] + 1), &dgnet->LegEval[i], n * (dgnet->taborder[i] + 1), &dgnet->LegEvalD[i]));
    for (j = 0; j <= dgnet->taborder[i]; j++) { deg[j] = j; }
    PetscCall(PetscDTLegendreEval(n, xnodes, dgnet->taborder[i] + 1, deg, dgnet->LegEval[i], dgnet->LegEvalD[i], NULL));
    PetscCall(PetscMalloc1(2 * (dgnet->taborder[i] + 1), &dgnet->LegEvaL_bdry[i]));
    PetscCall(PetscDTLegendreEval(2, bdry, dgnet->taborder[i] + 1, deg, dgnet->LegEvaL_bdry[i], NULL, NULL));
    PetscCall(PetscMalloc1(dgnet->taborder[i] + 1, &dgnet->Leg_L2[i]));
    for (j = 0; j <= dgnet->taborder[i]; j++) { dgnet->Leg_L2[i][j] = (2.0 * j + 1.) / (2.); }
    /* Viewer evaluations to be migrated */
    dgnet->numviewpts[i] = (dgnet->taborder[i] + 1);
    PetscCall(PetscMalloc1(dgnet->numviewpts[i], &viewnodes));
    for (j = 0; j < dgnet->numviewpts[i]; j++) viewnodes[j] = 2. * j / (dgnet->numviewpts[i]) - 1.;
    PetscCall(PetscMalloc1(dgnet->numviewpts[i] * (dgnet->taborder[i] + 1), &dgnet->LegEval_equispaced[i]));
    PetscCall(PetscDTLegendreEval(dgnet->numviewpts[i], viewnodes, dgnet->taborder[i] + 1, deg, dgnet->LegEval_equispaced[i], NULL, NULL));
    PetscCall(PetscFree(viewnodes));
    PetscCall(PetscFree(deg));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LegendreTabulationViewer_Internal(PetscInt npoints, PetscInt ndegree, PetscViewer viewer, PetscReal *LegEval)
{
  PetscInt  deg, qpoint;
  PetscReal viewerarray[npoints]; /* For some reason malloc was giving me memory corruption, but this works ... */

  PetscFunctionBegin;
  /* View each row individually (makes more sense to view) */
  for (deg = 0; deg <= ndegree; deg++) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Degree %i Evaluations \n", deg));
    for (qpoint = 0; qpoint < npoints; qpoint++) { *(viewerarray + qpoint) = LegEval[qpoint * (ndegree + 1) + deg]; }
    PetscCall(PetscRealView(npoints, viewerarray, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 TODO refactor as a petsc_____view function ?
*/
PetscErrorCode ViewDiscretizationObjects(DGNetwork dgnet, PetscViewer viewer)
{
  PetscInt i, quadsize;
  PetscInt ndegree;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Tab Order size: %i \n", dgnet->tabordersize));
  /* call standard viewers for discretization objects if available */
  PetscCall(PetscQuadratureView(dgnet->quad, viewer));
  PetscCall(PetscQuadratureGetData(dgnet->quad, NULL, NULL, &quadsize, NULL, NULL));
  /* View the tabulation arrays
    TODO as per other comments, these arrays should be petsctabulation objects and this should be its dedicated viewing routine
  */
  PetscCall(PetscViewerASCIIPrintf(viewer, "Quadsize: %i \n", quadsize));

  /* Iterate through the tabulation Orders */
  for (i = 0; i < dgnet->tabordersize; i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Legendre Tabulation Order: %i \n \n", dgnet->taborder[i]));
    /* Hack to make use of PetscRealViewer function */
    /* Maybe should be redone to have everything stored as Matrices, or custom storage? Idk man, either
       way it will work for now, though involves silly copying of data to get the arrays in the right format
       for viewing. Basically transposing the induced matrix from this data */
    ndegree = dgnet->taborder[i];

    PetscCall(PetscViewerASCIIPrintf(viewer, "Legendre Evaluations at Quadrature Points \n"));
    PetscCall(LegendreTabulationViewer_Internal(quadsize, ndegree, viewer, dgnet->LegEval[i]));

    PetscCall(PetscViewerASCIIPrintf(viewer, "Legendre Derivative Evaluations at Quadrature Points \n"));
    PetscCall(LegendreTabulationViewer_Internal(quadsize, ndegree, viewer, dgnet->LegEvalD[i]));

    PetscCall(PetscViewerASCIIPrintf(viewer, "Legendre Evaluations at Boundary Quadrature \n"));
    /* Fix hard coded 1D code here. We assume that the boundary evaluation quadrature has only two points */
    PetscCall(LegendreTabulationViewer_Internal(2, ndegree, viewer, dgnet->LegEvaL_bdry[i]));

    PetscCall(PetscViewerASCIIPrintf(viewer, "Legendre Normalization\n"));
    PetscCall(PetscRealView(ndegree + 1, dgnet->Leg_L2[i], viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TODO : Refactor as PetscView Function

  Function for Viewing the Mesh information inside of the dgnet (just calls dmview for each
  dmplex inside the edges)
*/
PetscErrorCode DGNetworkViewEdgeDMs(DGNetwork dgnet, PetscViewer viewer)
{
  PetscInt e, eStart, eEnd;
  EdgeFE   edgefe;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n Mesh on Edge %i \n \n ", e));
    PetscCall(DMView(edgefe->dm, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Just prints the jacobian and inverse jacobians to screen for dms inside the edgee

ONLY WORKS FOR 1D MESHES FOR NOW !!!! */
PetscErrorCode DGNetworkViewEdgeGeometricInfo(DGNetwork dgnet, PetscViewer viewer)
{
  PetscInt  e, eStart, eEnd, c, cStart, cEnd;
  EdgeFE    edgefe;
  PetscReal J, Jinv, Jdet;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n \n Geometric Info on Edge %i \n \n \n ", e));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; c++) {
      PetscCall(DMPlexComputeCellGeometryAffineFEM(edgefe->dm, c, NULL, &J, &Jinv, &Jdet));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Cell %i: J: %e  - Jinv: %e - Jdet: %e \n  ", c, J, Jinv, Jdet));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* WIP, builds the NetRP objects and assigns to verrtices of the NetRS, make cleaner later  */
PetscErrorCode DGNetworkAssignNetRS(DGNetwork dgnet)
{
  PetscInt v, vStart, vEnd, vdeg;
  NetRP    netrpbdry, netrpcouple;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrpbdry));
  PetscCall(NetRPSetType(netrpbdry, NETRPOUTFLOW));
  PetscCall(NetRPSetFlux(netrpbdry, dgnet->physics.rs));

  PetscCall(DMNetworkGetVertexRange(dgnet->network, &vStart, &vEnd));
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrpcouple));
  if (dgnet->linearcoupling) {
    PetscCall(NetRPSetType(netrpcouple, NETRPLINEARIZED));
  } else {
    PetscCall(NetRPSetType(netrpcouple, NETRPEXACTSWE));
  }
  PetscCall(NetRPSetFlux(netrpcouple, dgnet->physics.rs));

  PetscCall(NetRSCreate(PETSC_COMM_WORLD, &dgnet->netrs));
  PetscCall(NetRSSetFromOptions(dgnet->netrs));
  PetscCall(NetRSSetFlux(dgnet->netrs, dgnet->physics.rs));
  PetscCall(NetRSSetNetwork(dgnet->netrs, dgnet->network));
  PetscCall(NetRSSetUp(dgnet->netrs));

  for (v = vStart; v < vEnd; v++) {
    /*
      type dispatching depending on number of edges
    */
    PetscCall(NetRSGetVertexDegree(dgnet->netrs, v, &vdeg));
    if (vdeg == 1) {
      PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpbdry));
    } else {
      PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpcouple));
    }
  }
  PetscCall(NetRSCreateLocalVec(dgnet->netrs, &dgnet->Flux));
  PetscCall(NetRSCreateLocalVec(dgnet->netrs, &dgnet->RiemannData));
  PetscCall(NetRPDestroy(&netrpbdry));
  PetscCall(NetRPDestroy(&netrpcouple));
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

PetscErrorCode DGNetworkAssignNetRS_Traffic(DGNetwork dgnet)
{
  PetscInt v, vStart, vEnd, vdeg, indeg, outdeg;
  NetRP    netrpbdry, netrpcouple, netrpcouple_priority;

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
      PetscCall(NetRSAddNetRPatVertex(dgnet->netrs, v, netrpbdry));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkCleanUp(DGNetwork dgnet)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(MPI_Comm_rank(dgnet->comm, &rank));
  if (!rank) { PetscCall(PetscFree2(dgnet->junction, dgnet->edgefe)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkCreateVectors(DGNetwork dgnet)
{
  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(dgnet->network, &dgnet->X));
  PetscCall(DMCreateLocalVector(dgnet->network, &dgnet->localX));
  PetscCall(DMCreateLocalVector(dgnet->network, &dgnet->localF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkDestroyTabulation(DGNetwork dgnet)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < dgnet->tabordersize; i++) {
    PetscCall(PetscFree2(dgnet->LegEval[i], dgnet->LegEvalD[i]));
    PetscCall(PetscFree(dgnet->Leg_L2[i]));
    PetscCall(PetscFree(dgnet->LegEvaL_bdry[i]));
    PetscCall(PetscFree(dgnet->LegEval_equispaced[i]));
  }
  PetscCall(PetscQuadratureDestroy(&dgnet->quad));
  PetscCall(PetscFree4(dgnet->LegEval, dgnet->Leg_L2, dgnet->LegEvalD, dgnet->LegEvaL_bdry));
  PetscCall(PetscFree(dgnet->taborder));
  PetscCall(PetscFree(dgnet->fieldtotab));
  PetscCall(PetscFree2(dgnet->pteval, dgnet->fluxeval));
  PetscCall(PetscFree2(dgnet->LegEval_equispaced, dgnet->numviewpts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkDestroyPhysics(DGNetwork dgnet)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall((*dgnet->physics.destroy)(dgnet->physics.user));
  for (i = 0; i < dgnet->physics.dof; i++) { PetscCall(PetscFree(dgnet->physics.fieldname[i])); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkDestroy(DGNetwork dgnet)
{
  PetscInt e, eStart, eEnd;
  EdgeFE   edgefe;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetEdgeRange(dgnet->network, &eStart, &eEnd));
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(dgnet->network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(DMDestroy(&edgefe->dm));
  }

  PetscCall(PetscFree2(dgnet->R, dgnet->Rinv));
  PetscCall(PetscFree5(dgnet->cuLR, dgnet->uLR, dgnet->flux, dgnet->speeds, dgnet->uPlus));
  PetscCall(PetscFree5(dgnet->limitactive, dgnet->charcoeff, dgnet->cbdryeval_L, dgnet->cbdryeval_R, dgnet->cuAvg));
  PetscCall(PetscFree2(dgnet->uavgs, dgnet->cjmpLR));
  PetscCall(DGNetworkDestroyTabulation(dgnet));
  PetscCall(DGNetworkDestroyPhysics(dgnet));
  PetscCall(VecDestroy(&dgnet->X));
  PetscCall(VecDestroy(&dgnet->localX));
  PetscCall(VecDestroy(&dgnet->localF));
  PetscCall(VecDestroy(&dgnet->Flux));
  PetscCall(VecDestroy(&dgnet->RiemannData));
  PetscCall(NetRSDestroy(&dgnet->netrs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal evalviewpt_internal(DGNetwork dgnet, PetscInt field, PetscInt viewpt, const PetscReal *comp)
{
  PetscInt  deg, tab = dgnet->fieldtotab[field], ndegree = dgnet->taborder[tab];
  PetscReal eval = 0.0;

  for (deg = 0; deg <= ndegree; deg++) { eval += comp[deg] * dgnet->LegEval_equispaced[tab][viewpt * (ndegree + 1) + deg]; }
  return eval;
}

PetscErrorCode DGNetworkMonitorCreate(DGNetwork dgnet, DGNetworkMonitor *monitorptr)
{
  DGNetworkMonitor monitor;
  MPI_Comm         comm;
  PetscMPIInt      size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dgnet->network, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel DGNetworkMonitor is not supported yet");

  PetscCall(PetscMalloc1(1, &monitor));
  monitor->comm      = comm;
  monitor->dgnet     = dgnet;
  monitor->firstnode = NULL;

  *monitorptr = monitor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorPop(DGNetworkMonitor monitor)
{
  DGNetworkMonitorList node;

  PetscFunctionBegin;
  if (monitor->firstnode) {
    /* Update links */
    node               = monitor->firstnode;
    monitor->firstnode = node->next;

    /* Free list node */
    PetscCall(PetscViewerDestroy(&(node->viewer)));
    PetscCall(VecDestroy(&(node->v)));
    PetscCall(PetscFree(node));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorDestroy(DGNetworkMonitor *monitor)
{
  PetscFunctionBegin;
  while ((*monitor)->firstnode) { PetscCall(DGNetworkMonitorPop(*monitor)); }
  PetscCall(PetscFree(*monitor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ymax and ymin must be removed by the caller */
PetscErrorCode DGNetworkMonitorAdd(DGNetworkMonitor monitor, PetscInt element, PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax, PetscBool hold)
{
  PetscDrawLG          drawlg;
  PetscDrawAxis        axis;
  PetscMPIInt          rank, size;
  DGNetworkMonitorList node;
  char                 titleBuffer[64];
  PetscInt             vStart, vEnd, eStart, eEnd, viewsize, field, cStart, cEnd;
  DM                   network = monitor->dgnet->network;
  DGNetwork            dgnet   = monitor->dgnet;
  PetscInt             dof     = dgnet->physics.dof;
  EdgeFE               edgefe;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(DMNetworkGetVertexRange(network, &vStart, &vEnd));
  PetscCall(DMNetworkGetEdgeRange(network, &eStart, &eEnd));
  /* make a viewer for each field on the componenent */
  for (field = 0; field < dof; field++) {
    /* Make window title */
    if (vStart <= element && element < vEnd) {
      /* Nothing to view on the vertices for DGNetwork (for now) so skip */
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (eStart <= element && element < eEnd) {
      PetscCall(PetscSNPrintf(titleBuffer, 64, "%s @ edge %d [%d / %d]", dgnet->physics.fieldname[field], element - eStart, rank, size - 1));
    } else {
      /* vertex / edge is not on local machine, so skip! */
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscMalloc1(1, &node));
    /* Setup viewer. */
    PetscCall(PetscViewerDrawOpen(monitor->comm, NULL, titleBuffer, PETSC_DECIDE, PETSC_DECIDE, PETSC_DRAW_QUARTER_SIZE, PETSC_DRAW_QUARTER_SIZE, &(node->viewer)));
    PetscCall(PetscViewerPushFormat(node->viewer, PETSC_VIEWER_DRAW_LG_XRANGE));
    PetscCall(PetscViewerDrawGetDrawLG(node->viewer, 0, &drawlg));
    PetscCall(PetscDrawLGGetAxis(drawlg, &axis));
    if (xmin != PETSC_DECIDE && xmax != PETSC_DECIDE) {
      PetscCall(PetscDrawAxisSetLimits(axis, xmin, xmax, ymin, ymax));
    } else {
      PetscCall(PetscDrawAxisSetLimits(axis, 0, 1, ymin, ymax));
    }
    PetscCall(PetscDrawAxisSetHoldLimits(axis, hold));

    /* Setup vector storage for drawing. */
    PetscCall(DMNetworkGetComponent(network, element, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
    viewsize = dgnet->numviewpts[dgnet->fieldtotab[field]] * (cEnd - cStart);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v)));

    node->element      = element;
    node->field        = field;
    node->next         = monitor->firstnode;
    node->vsize        = viewsize;
    monitor->firstnode = node;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorView(DGNetworkMonitor monitor, Vec x)
{
  PetscInt             edgeoff, fieldoff, cStart, cEnd, c, tab, q, viewpt;
  const PetscScalar   *xx;
  PetscScalar         *vv;
  DGNetworkMonitorList node;
  DM                   network = monitor->dgnet->network;
  DGNetwork            dgnet   = monitor->dgnet;
  EdgeFE               edgefe;
  PetscSection         section;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &xx));
  for (node = monitor->firstnode; node; node = node->next) {
    PetscCall(DMNetworkGetLocalVecOffset(network, node->element, FVEDGE, &edgeoff));
    PetscCall(DMNetworkGetComponent(dgnet->network, node->element, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(VecGetArray(node->v, &vv));

    PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
    PetscCall(DMGetSection(edgefe->dm, &section));
    tab = dgnet->fieldtotab[node->field];
    /* Evaluate at the eqiudistant point evalutions */
    viewpt = 0;
    for (c = cStart; c < cEnd; c++) {
      PetscCall(PetscSectionGetFieldOffset(section, c, node->field, &fieldoff));
      for (q = 0; q < dgnet->numviewpts[tab]; q++) { vv[viewpt++] = evalviewpt_internal(dgnet, node->field, q, xx + edgeoff + fieldoff); }
    }
    PetscCall(VecRestoreArray(node->v, &vv));
    PetscCall(VecView(node->v, node->viewer));
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorCreate_Glvis(DGNetwork dgnet, DGNetworkMonitor_Glvis *monitorptr)
{
  DGNetworkMonitor_Glvis monitor;
  MPI_Comm               comm;
  PetscMPIInt            size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dgnet->network, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel DGNetworkMonitor is not supported yet");

  PetscCall(PetscMalloc1(1, &monitor));
  monitor->comm      = comm;
  monitor->dgnet     = dgnet;
  monitor->firstnode = NULL;

  *monitorptr = monitor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitor_g2l_internal(PetscObject V, PetscInt nfields, PetscObject Vfield[], void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis)ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe;
  PetscInt                   c, cStart, cEnd, field, tab, dof = dgnet->physics.dof, i, fieldoff, deg, ndegree;
  PetscSection               section;
  const PetscReal           *v;
  PetscReal                 *vwork;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetComponent(network, node->element, FVEDGE, NULL, (void **)&edgefe, NULL));
  PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
  PetscCall(DMGetSection(edgefe->dm, &section));
  PetscCall(VecGetArrayRead((Vec)V, &v));
  /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
  for (field = 0; field < dof; field++) {
    i = 0;
    PetscCall(VecGetArray((Vec)Vfield[field], &vwork));
    for (c = cStart; c < cEnd; c++) {
      PetscCall(PetscSectionGetFieldOffset(section, c, field, &fieldoff));
      tab     = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      for (deg = 0; deg <= ndegree; deg++) { vwork[i++] = evalviewpt_internal(dgnet, field, deg, v + fieldoff); }
    }
    PetscCall(VecRestoreArray((Vec)Vfield[field], &vwork));
  }
  PetscCall(VecRestoreArrayRead((Vec)V, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitor_destroyctx_internal(void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorAdd_Glvis(DGNetworkMonitor_Glvis monitor, PetscInt element, const char hostname[], PetscViewerGLVisType type)
{
  PetscMPIInt                rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt                   viewsize, field, cStart, cEnd, tab, Dim = 1;
  ;
  DM        network = monitor->dgnet->network;
  DGNetwork dgnet   = monitor->dgnet;
  PetscInt  dof     = dgnet->physics.dof;
  EdgeFE    edgefe;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(PetscMalloc1(1, &node));
  PetscCall(PetscMalloc3(dof, &node->dim, dof, &node->v_work, dof, &node->fec_type));

  PetscCall(PetscViewerGLVisOpen(monitor->comm, type, hostname, PETSC_DECIDE, &node->viewer));

  PetscCall(DMNetworkGetComponent(network, element, FVEDGE, NULL, (void **)&edgefe, NULL));
  PetscCall(DMClone(edgefe->dm, &node->viewdm));
  PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
  /* make the work vector for each field */
  for (field = 0; field < dof; field++) {
    /* Setup vector storage for drawing. */
    tab      = dgnet->fieldtotab[field];
    viewsize = (cEnd - cStart) * (dgnet->taborder[tab] + 1); /* number of variables for the given field */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field])));
    PetscCall(PetscObjectCompose((PetscObject)node->v_work[field], "__PETSc_dm", (PetscObject)edgefe->dm)); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    PetscCall(PetscMalloc(64, &node->fec_type[field]));
    PetscCall(PetscSNPrintf(node->fec_type[field], 64, "FiniteElementCollection: L2_T4_%iD_P%i", Dim, dgnet->taborder[tab]));
    node->dim[field] = Dim;
  }
  PetscCall(DMCreateGlobalVector(edgefe->dm, &node->v));

  node->element      = element;
  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  node->snapid       = 0;
  monitor->firstnode = node;

  PetscCall(PetscViewerGLVisSetFields(node->viewer, dof, (const char **)node->fec_type, node->dim, DGNetworkMonitor_g2l_internal, (PetscObject *)node->v_work, (void *)node, DGNetworkMonitor_destroyctx_internal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorPop_Glvis(DGNetworkMonitor_Glvis monitor)
{
  DGNetworkMonitorList_Glvis node;
  PetscInt                   field, dof = monitor->dgnet->physics.dof;

  PetscFunctionBegin;
  if (monitor->firstnode) {
    /* Update links */
    node               = monitor->firstnode;
    monitor->firstnode = node->next;
    /* Free list node */
    if (node->v) PetscCall(VecDestroy(&(node->v)));
    for (field = 0; field < dof; field++) {
      PetscCall(VecDestroy(&node->v_work[field]));
      PetscCall(PetscFree(node->fec_type[field]));
    }
    PetscCall(PetscFree3(node->v_work, node->dim, node->fec_type));
    PetscCall(PetscViewerDestroy(&(node->viewer)));
    if (node->viewdm) PetscCall(DMDestroy(&node->viewdm));
    PetscCall(PetscFree(node));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorDestroy_Glvis(DGNetworkMonitor_Glvis *monitor)
{
  PetscFunctionBegin;
  while ((*monitor)->firstnode) { PetscCall(DGNetworkMonitorPop_Glvis(*monitor)); }
  PetscCall(PetscFree(*monitor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorView_Glvis(DGNetworkMonitor_Glvis monitor, Vec x)
{
  PetscInt                   edgeoff, i, vecsize;
  const PetscScalar         *xx;
  PetscScalar               *vv;
  DGNetworkMonitorList_Glvis node;
  DM                         network = monitor->dgnet->network;
  DGNetwork                  dgnet   = monitor->dgnet;
  EdgeFE                     edgefe;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &xx));
  for (node = monitor->firstnode; node; node = node->next) {
    PetscCall(PetscViewerGLVisSetSnapId(node->viewer, node->snapid++));

    PetscCall(DMNetworkGetLocalVecOffset(network, node->element, FVEDGE, &edgeoff));
    PetscCall(DMNetworkGetComponent(dgnet->network, node->element, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(VecGetArray(node->v, &vv));
    PetscCall(VecGetSize(node->v, &vecsize));
    for (i = 0; i < vecsize; i++) { vv[i] = xx[edgeoff + i]; }
    PetscCall(VecRestoreArray(node->v, &vv));
    PetscCall(VecView(node->v, node->viewer));
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_l(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy = 2 * x[1] - 1, zz = 2 * x[2] - 1;

  xp[1] = yy * PetscSqrtReal(1 - PetscPowReal(zz, 2) / 2.) / 10.;
  xp[2] = zz * PetscSqrtReal(1 - PetscPowReal(yy, 2) / 2.) / 10.;
  xp[0] = 2. * x[0] + 0.1;
}

/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy = 2 * x[1] - 1, zz = 2 * x[2] - 1;

  xp[1] = yy * PetscSqrtReal(1 - PetscPowReal(zz, 2) / 2.);
  xp[2] = zz * PetscSqrtReal(1 - PetscPowReal(yy, 2) / 2.);
  xp[0] = x[0];
}

/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_r(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy = 2 * x[1] - 1, zz = 2 * x[2] - 1;

  xp[1] = yy * PetscSqrtReal(1 - PetscPowReal(zz, 2) / 2.) / 10.0;
  xp[2] = zz * PetscSqrtReal(1 - PetscPowReal(yy, 2) / 2.) / 10.0;
  xp[0] = x[0] * 2. - 2.1; /*hack for presentation */
}

/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_t(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy = 2 * x[1] - 1, zz = 2 * x[2] - 1;

  xp[0] = yy * PetscSqrtReal(1 - PetscPowReal(zz, 2) / 2.) / 10.;
  xp[2] = zz * PetscSqrtReal(1 - PetscPowReal(yy, 2) / 2.) / 10.;
  xp[1] = 2. * x[0] + 0.1;
}

/* 3d visualization of a network element, transformation of unit cube to unit cylinder element. */
static void f0_circle_b(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xp[])
{
  const PetscReal yy = 2 * x[1] - 1, zz = 2 * x[2] - 1;

  xp[0] = yy * PetscSqrtReal(1 - PetscPowReal(zz, 2) / 2.) / 10.;
  xp[2] = zz * PetscSqrtReal(1 - PetscPowReal(yy, 2) / 2.) / 10.;
  xp[1] = -2. * x[0] - 0.1;
}

static PetscErrorCode DGNetworkCreateViewDM(DM dm)
{
  DM             cdm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, dE, cStart, size;
  PetscBool      simplex;
  PetscReal     *coord;
  Vec            Coord;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMPlexGetHeightStratum(cdm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 3, PETSC_DECIDE, &fe));
  PetscCall(DMProjectCoordinates(dm, fe));
  PetscCall(DMGetCoordinates(dm, &Coord));
  PetscCall(VecGetSize(Coord, &size));
  PetscCall(VecGetArray(Coord, &coord));
  PetscCall(VecRestoreArray(Coord, &coord));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMPlexRemapGeometry(dm, 0.0, f0_circle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DGNetworkCreateViewDM2(DM dm)
{
  DM             cdm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, dE, cStart;
  PetscBool      simplex;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMPlexGetHeightStratum(cdm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 1, PETSC_DECIDE, &fe));
  PetscCall(DMProjectCoordinates(dm, fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitor_3D_g2l_internal(PetscObject V, PetscInt nfields, PetscObject Vfield[], void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis)ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe;
  PetscInt                   copy, c, cStart, cEnd, field, tab, dof = dgnet->physics.dof, i, fieldoff, deg, ndegree;
  PetscSection               section;
  const PetscReal           *v;
  PetscReal                 *vwork;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetComponent(network, node->element, FVEDGE, NULL, (void **)&edgefe, NULL));
  PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
  PetscCall(DMGetSection(edgefe->dm, &section));
  PetscCall(VecGetArrayRead((Vec)V, &v));
  /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
  for (field = 0; field < dof; field++) {
    i = 0;
    PetscCall(VecGetArray((Vec)Vfield[field], &vwork));
    for (c = cStart; c < cEnd; c++) {
      PetscCall(PetscSectionGetFieldOffset(section, c, field, &fieldoff));
      tab     = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      for (deg = 0; deg <= ndegree; deg++) {
        vwork[i] = evalviewpt_internal(dgnet, field, deg, v + fieldoff);
        for (copy = 1; copy < (ndegree + 1) * (ndegree + 1); copy++) { vwork[i + copy * (ndegree + 1)] = vwork[i]; }
        i++;
      }
      i += (ndegree + 1) * ((ndegree + 1) * (ndegree + 1) - 1);
    }
    PetscCall(VecRestoreArray((Vec)Vfield[field], &vwork));
  }
  PetscCall(VecRestoreArrayRead((Vec)V, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorAdd_Glvis_3D(DGNetworkMonitor_Glvis monitor, PetscInt element, const char hostname[], PetscViewerGLVisType type)
{
  PetscMPIInt                rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt                   viewsize, field, cStart, cEnd, tab, Dim = 3;
  DM                         network = monitor->dgnet->network;
  DGNetwork                  dgnet   = monitor->dgnet;
  PetscInt                   dof     = dgnet->physics.dof;
  EdgeFE                     edgefe;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(PetscMalloc1(1, &node));
  PetscCall(PetscMalloc3(dof, &node->dim, dof, &node->v_work, dof, &node->fec_type));

  PetscCall(PetscViewerGLVisOpen(monitor->comm, type, hostname, PETSC_DECIDE, &node->viewer));

  PetscCall(DMNetworkGetComponent(network, element, FVEDGE, NULL, (void **)&edgefe, NULL));
  PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
  PetscInt faces[3] = {cEnd - cStart, 1, 1};
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 3, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &node->viewdm));
  PetscCall(DGNetworkCreateViewDM(node->viewdm));
  /* make the work vector for each field */
  for (field = 0; field < dof; field++) {
    /* Setup vector storage for drawing. */
    tab      = dgnet->fieldtotab[field];
    viewsize = (cEnd - cStart) * PetscPowInt((dgnet->taborder[tab] + 1), Dim); /* number of variables for the given field */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field])));
    PetscCall(PetscObjectCompose((PetscObject)node->v_work[field], "__PETSc_dm", (PetscObject)node->viewdm)); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    PetscCall(PetscMalloc(64, &node->fec_type[field]));
    PetscCall(PetscSNPrintf(node->fec_type[field], 64, "FiniteElementCollection: L2_T4_%iD_P%i", Dim, dgnet->taborder[tab]));
    node->dim[field] = Dim;
  }
  PetscCall(DMCreateGlobalVector(edgefe->dm, &node->v));

  node->element      = element;
  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  monitor->firstnode = node;
  node->snapid       = 0;

  PetscCall(PetscViewerGLVisSetFields(node->viewer, dof, (const char **)node->fec_type, node->dim, DGNetworkMonitor_3D_g2l_internal, (PetscObject *)node->v_work, (void *)node, DGNetworkMonitor_destroyctx_internal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Convience create from DAG function that only creates the topology, leaving the geometry dm and section uncreated */
PetscErrorCode DMPlexCreateFromDAG_Topological(DM dm, PetscInt depth, const PetscInt numPoints[], const PetscInt coneSize[], const PetscInt cones[], const PetscInt coneOrientations[])
{
  PetscInt firstVertex = -1, pStart = 0, pEnd = 0, p, dim, dimEmbed, d, off;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  if (dimEmbed < dim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Embedding dimension %D cannot be less than intrinsic dimension %d", dimEmbed, dim);
  for (d = 0; d <= depth; ++d) pEnd += numPoints[d];
  PetscCall(DMPlexSetChart(dm, pStart, pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscCall(DMPlexSetConeSize(dm, p, coneSize[p - pStart]));
    if (firstVertex < 0 && !coneSize[p - pStart]) { firstVertex = p - pStart; }
  }
  if (firstVertex < 0 && numPoints[0]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected %D vertices but could not find any", numPoints[0]);
  PetscCall(DMSetUp(dm)); /* Allocate space for cones */
  for (p = pStart, off = 0; p < pEnd; off += coneSize[p - pStart], ++p) {
    PetscCall(DMPlexSetCone(dm, p, &cones[off]));
    PetscCall(DMPlexSetConeOrientation(dm, p, &coneOrientations[off]));
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
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

PetscErrorCode DMPlexAdd_Disconnected(DM *dmlist, PetscInt numdm, DM *dmsum, PetscSection *stratumoffsets)
{
  PetscInt           p, i, j, k, depth, depth_temp, dim, dim_prev, dim_top, dim_top_temp, pStart, pEnd, chartsize, stratum, totalconesize;
  PetscInt          *numpoints_g, *coneSize_g, *cones_g, *coneOrientations_g, coneSize, off, prevtotal;
  const PetscInt    *cone, *coneOrientation;
  const PetscScalar *vertexcoords;
  DMType             dmtype;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dmlist[0]);
  PetscSection       offsets;
  char               fieldname[64]; /* Should be long enough unless we get crazy deep complexs */
  DM                 dm_sum;
  PetscBool          flag;

  PetscFunctionBegin;
  /* input checks */
  if (numdm <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDim(dmlist[0], &dim_prev));
  for (i = 0; i < numdm; i++) {
    PetscCall(DMGetType(dmlist[i], &dmtype));
    PetscCall(PetscStrncmp(dmtype, DMPLEX, 64, &flag));
    if (!flag) SETERRQ(PetscObjectComm((PetscObject)dmlist[i]), PETSC_ERR_ARG_WRONG, "Wrong DM Object can only be DMPlex");
    PetscCall(DMGetCoordinateDim(dmlist[i], &dim));
    if (dim_prev != dim) SETERRQ(PetscObjectComm((PetscObject)dmlist[i]), PETSC_ERR_ARG_WRONG, "All Input DM objects must have the same Geometric Dimension");
  }

  /* Acquire maximum depth size across all dms and maximum topologial dimension chartsize */
  depth     = 0;
  dim_top   = 0;
  chartsize = 0;
  for (i = 0; i < numdm; i++) {
    PetscCall(DMPlexGetDepth(dmlist[i], &depth_temp));
    if (depth < depth_temp) depth = depth_temp;
    PetscCall(DMGetDimension(dmlist[i], &dim_top_temp));
    if (dim_top < dim_top_temp) dim_top = dim_top_temp;
    PetscCall(DMPlexGetChart(dmlist[i], &pStart, &pEnd));
    chartsize += (pEnd - pStart);
  }

  PetscCall(PetscMalloc1(chartsize, &coneSize_g));
  PetscCall(PetscCalloc1(depth + 1, &numpoints_g));
  /* set up the stratum offset section */
  PetscCall(PetscSectionCreate(comm, &offsets));
  PetscCall(PetscSectionSetNumFields(offsets, depth + 1)); /* one field per stratum */
  PetscCall(PetscSectionSetChart(offsets, 0, numdm));
  for (j = 0; j <= depth; j++) {
    PetscCall(PetscSectionSetFieldComponents(offsets, j, 1));
    PetscCall(PetscSNPrintf(fieldname, 64, "Stratum Depth %D", j));
    PetscCall(PetscSectionSetFieldName(offsets, j, fieldname));
  }
  /* Iterate through the meshes and compute the number of points at each stratum */

  for (i = 0; i < numdm; i++) {
    PetscCall(DMPlexGetDepth(dmlist[i], &depth_temp));
    PetscCall(PetscSectionSetDof(offsets, i, depth_temp + 1));
    for (stratum = 0; stratum <= depth_temp; stratum++) {
      PetscCall(PetscSectionSetFieldDof(offsets, i, stratum, 1));
      PetscCall(DMPlexGetDepthStratum(dmlist[i], stratum, &pStart, &pEnd));
      /* manually specify the section offset information, as the domain chart is not the same
         as the range chart, and is not an onto mapbrping */
      PetscCall(PetscSectionSetFieldOffset(offsets, i, stratum, numpoints_g[stratum] - pStart));
      numpoints_g[stratum] += (pEnd - pStart);
    }
  }
  /* Now we have the offset information for the input dm stratum into the new dm stratum */

  /* Create the cone size information */
  totalconesize = 0;
  for (i = 0; i < numdm; i++) {
    PetscCall(DMPlexGetDepth(dmlist[i], &depth_temp));
    for (stratum = 0; stratum <= depth_temp; stratum++) {
      PetscCall(DMPlexGetDepthStratum(dmlist[i], stratum, &pStart, &pEnd));
      prevtotal = 0;
      for (j = 0; j < stratum; j++) prevtotal += numpoints_g[j];
      PetscCall(PetscSectionGetFieldOffset(offsets, i, stratum, &off));
      PetscCall(PetscSectionSetFieldOffset(offsets, i, stratum, off + prevtotal));
      PetscCall(PetscSectionGetFieldOffset(offsets, i, stratum, &off));
      for (p = pStart; p < pEnd; p++) {
        PetscCall(DMPlexGetConeSize(dmlist[i], p, &coneSize));
        coneSize_g[p + off] = coneSize;
        totalconesize += coneSize;
      }
    }
  }

  /* create the cone and cone orientations */
  PetscCall(PetscMalloc2(totalconesize, &cones_g, totalconesize, &coneOrientations_g));
  k = 0;
  for (stratum = 0; stratum <= depth; stratum++) {
    for (i = 0; i < numdm; i++) {
      PetscCall(DMPlexGetDepth(dmlist[i], &depth_temp));
      if (stratum <= depth_temp) {
        PetscCall(DMPlexGetDepthStratum(dmlist[i], stratum, &pStart, &pEnd));
        if (stratum > 0) { /* stratum = 0 doesn't matter as the cones for stratum = 0 are empty */
          PetscCall(PetscSectionGetFieldOffset(offsets, i, stratum - 1, &off));
        }
        for (p = pStart; p < pEnd; p++) {
          PetscCall(DMPlexGetCone(dmlist[i], p, &cone));
          PetscCall(DMPlexGetConeOrientation(dmlist[i], p, &coneOrientation));
          PetscCall(DMPlexGetConeSize(dmlist[i], p, &coneSize));
          for (j = 0; j < coneSize; j++) {
            coneOrientations_g[k] = coneOrientation[j];
            cones_g[k++]          = cone[j] + off; /* account for the offset in the cone stratum (stratum -1) */
          }
        }
      }
    }
  }
  /* Hack to make geometry work. I associate a a zero vector for the geometry field, in order the have all the
  sections and etc built automatically. To be redone when I am more skilled */

  /* In theory we have everything ready to create the new global dm */
  PetscCall(DMPlexCreate(comm, &dm_sum));
  PetscCall(DMSetDimension(dm_sum, dim_top));
  PetscCall(DMSetCoordinateDim(dm_sum, dim));

  PetscCall(PetscCalloc1(numpoints_g[0] * dim, &vertexcoords));

  PetscCall(DMPlexCreateFromDAG(dm_sum, depth, numpoints_g, coneSize_g, cones_g, coneOrientations_g, vertexcoords));
  PetscCall(PetscFree(numpoints_g));
  PetscCall(PetscFree(coneSize_g));
  PetscCall(PetscFree(vertexcoords));
  PetscCall(PetscFree2(cones_g, coneOrientations_g));

  /* Now we map the coordinates ... somehow */
  *dmsum          = dm_sum;
  *stratumoffsets = offsets;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkCreateNetworkDMPlex_3D(DGNetwork dgnet, const PetscInt edgelist[], PetscInt edgelistsize, DM *dmsum, PetscSection *stratumoffset, DM **dm_list, PetscInt *numdm)
{
  PetscInt       i = 0, j, e, eStart, eEnd, cStart, cEnd, dim, dE, pStart, pEnd, dof, p, off, off_g, off_stratum, secStart, secEnd, depth, stratum;
  DM            *dmlist, network = dgnet->network, cdm;
  EdgeFE         edgefe;
  PetscSection   coordsec, coordsec_g;
  PetscBool      simplex;
  PetscFE        fe;
  DMPolytopeType ct;
  Vec            Coord_g, Coord;
  PetscReal     *coord_g, *coord;

  PetscFunctionBegin;
  if (edgelist == NULL) { /* Assume the entire network is used */
    PetscCall(DMNetworkGetEdgeRange(network, &eStart, &eEnd));
    PetscCall(PetscMalloc1(eEnd - eStart, &dmlist));
    for (e = eStart; e < eEnd; e++) {
      PetscCall(DMNetworkGetComponent(network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
      PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
      PetscInt faces[3] = {cEnd - cStart, 1, 1};
      PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 3, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dmlist[i]));
      PetscCall(DGNetworkCreateViewDM2(dmlist[i]));
      if (e == eStart) {
        PetscCall(DMPlexRemapGeometry(dmlist[i++], 0, f0_circle_r));
      } else if (e == eStart + 2) {
        PetscCall(DMPlexRemapGeometry(dmlist[i++], 0, f0_circle_t));
      } else if (e == eStart + 1) {
        PetscCall(DMPlexRemapGeometry(dmlist[i++], 0, f0_circle_l));
      } else {
        PetscCall(DMPlexRemapGeometry(dmlist[i++], 0, f0_circle_b));
      }
    }
    *numdm = i;
    PetscCall(DMPlexAdd_Disconnected(dmlist, *numdm, dmsum, stratumoffset));
    PetscCall(DMGetCoordinateDM(*dmsum, &cdm));
    PetscCall(DMGetDimension(*dmsum, &dim));
    PetscCall(DMGetCoordinateDim(*dmsum, &dE));
    PetscCall(DMPlexGetHeightStratum(cdm, 0, &cStart, NULL));
    PetscCall(DMPlexGetCellType(*dmsum, cStart, &ct));
    simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 1, PETSC_DECIDE, &fe));
    PetscCall(DMProjectCoordinates(*dmsum, fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMGetCoordinateSection(*dmsum, &coordsec_g));
    PetscCall(DMGetCoordinates(*dmsum, &Coord_g));
    PetscCall(VecGetArray(Coord_g, &coord_g));
    /* Now map the coordinate data */
    for (i = 0; i < *numdm; i++) {
      PetscCall(DMGetCoordinates(dmlist[i], &Coord));
      PetscCall(VecGetArray(Coord, &coord));
      PetscCall(DMGetCoordinateSection(dmlist[i], &coordsec));

      PetscCall(PetscSectionGetChart(coordsec, &secStart, &secEnd));
      /* Iterate through the stratum */
      PetscCall(DMPlexGetDepth(dmlist[i], &depth));
      for (stratum = 0; stratum <= depth; stratum++) {
        PetscCall(DMPlexGetDepthStratum(dmlist[i], stratum, &pStart, &pEnd));
        PetscCall(PetscSectionGetFieldOffset(*stratumoffset, i, stratum, &off_stratum));
        /* there is a better way of doing this ... for later */
        for (p = pStart; p < pEnd && p < secEnd; p++) {
          if (p >= secStart) {
            PetscCall(PetscSectionGetFieldOffset(coordsec, p, 0, &off)); /* domain offset */
            PetscCall(PetscSectionGetFieldDof(coordsec, p, 0, &dof));
            PetscCall(PetscSectionGetFieldOffset(coordsec_g, p + off_stratum, 0, &off_g)); /*range offset */
            for (j = 0; j < dof; j++) { coord_g[off_g + j] = coord[off + j]; }
          }
        }
      }
      PetscCall(VecRestoreArray(Coord, &coord));
    }
    PetscCall(VecRestoreArray(Coord_g, &coord_g));
    PetscCall(DMSetCoordinatesLocal(*dmsum, Coord_g));

    /* in theory the coordinates are now mapped correctly ... we shall see */
    *dm_list = dmlist;
  } else {
    /* TODO */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* More viewer stuff */
/* Here we assume we are viewing the entire DGNetwork vector */
PetscErrorCode DGNetworkMonitor_3D_NET_g2l_internal(PetscObject V, PetscInt nfields, PetscObject Vfield[], void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis)ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe;
  PetscInt                   copy, c, cStart, cEnd, field, tab, dof = dgnet->physics.dof, i, fieldoff, deg, ndegree, e, eStart, eEnd, cCount, off_e;
  PetscSection               section;
  const PetscReal           *v;
  PetscReal                 *vwork;
  PetscInt                   Dim = 3;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead((Vec)V, &v));
  PetscCall(DMNetworkGetEdgeRange(network, &eStart, &eEnd));
  cCount = 0;
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(DMNetworkGetLocalVecOffset(network, e, FVEDGE, &off_e));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
    PetscCall(DMGetSection(edgefe->dm, &section));
    /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
    for (field = 0; field < dof; field++) {
      tab     = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      i       = cCount * PetscPowInt(ndegree + 1, Dim);
      PetscCall(VecGetArray((Vec)Vfield[field], &vwork));
      for (c = cStart; c < cEnd; c++) {
        PetscCall(PetscSectionGetFieldOffset(section, c, field, &fieldoff));
        for (deg = 0; deg <= ndegree; deg++) {
          vwork[i] = evalviewpt_internal(dgnet, field, deg, v + fieldoff + off_e);
          for (copy = 1; copy < (ndegree + 1) * (ndegree + 1); copy++) { vwork[i + copy * (ndegree + 1)] = vwork[i]; }
          i++;
        }
        i += (ndegree + 1) * ((ndegree + 1) * (ndegree + 1) - 1);
      }
      PetscCall(VecRestoreArray((Vec)Vfield[field], &vwork));
    }
    cCount += cEnd - cStart;
  }

  PetscCall(VecRestoreArrayRead((Vec)V, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorAdd_Glvis_3D_NET(DGNetworkMonitor_Glvis monitor, const char hostname[], PetscViewerGLVisType type)
{
  PetscMPIInt                rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt                   viewsize, field, cStart, cEnd, tab, Dim = 3, i;
  DGNetwork                  dgnet = monitor->dgnet;
  PetscInt                   dof   = dgnet->physics.dof;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(PetscMalloc1(1, &node));
  PetscCall(PetscMalloc3(dof, &node->dim, dof, &node->v_work, dof, &node->fec_type));

  PetscCall(PetscViewerGLVisOpen(monitor->comm, type, hostname, PETSC_DECIDE, &node->viewer));
  PetscCall(DGNetworkCreateNetworkDMPlex_3D(dgnet, NULL, 0, &node->viewdm, &node->stratumoffset, &node->dmlist, &node->numdm));
  /* delete the unneeded dms */
  for (i = 0; i < node->numdm; i++) { PetscCall(DMDestroy(&node->dmlist[i])); }
  PetscCall(PetscFree(node->dmlist));
  /* Create the network mesh */
  PetscCall(DMPlexGetHeightStratum(node->viewdm, 0, &cStart, &cEnd));
  /* make the work vector for each field */
  for (field = 0; field < dof; field++) {
    /* Setup vector storage for drawing. */
    tab      = dgnet->fieldtotab[field];
    viewsize = (cEnd - cStart) * PetscPowInt((dgnet->taborder[tab] + 1), Dim); /* number of variables for the given field */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field])));
    PetscCall(PetscObjectCompose((PetscObject)node->v_work[field], "__PETSc_dm", (PetscObject)node->viewdm)); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    PetscCall(PetscMalloc(64, &node->fec_type[field]));
    PetscCall(PetscSNPrintf(node->fec_type[field], 64, "FiniteElementCollection: L2_T4_%iD_P%i", Dim, dgnet->taborder[tab]));
    node->dim[field] = Dim;
  }

  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  node->v            = NULL;
  monitor->firstnode = node;

  PetscCall(PetscViewerGLVisSetFields(node->viewer, dof, (const char **)node->fec_type, node->dim, DGNetworkMonitor_3D_NET_g2l_internal, (PetscObject *)node->v_work, (void *)node, DGNetworkMonitor_destroyctx_internal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorView_Glvis_NET(DGNetworkMonitor_Glvis monitor, Vec x)
{
  DGNetworkMonitorList_Glvis node;

  PetscFunctionBegin;
  for (node = monitor->firstnode; node; node = node->next) {
    PetscCall(PetscViewerGLVisSetSnapId(node->viewer, node->snapid++));
    PetscCall(VecView(x, node->viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 2D FULL NETWORK VIEWING HERE */

/* Here we assume we are viewing the entire DGNetwork vector */
PetscErrorCode DGNetworkMonitor_2D_NET_g2l_internal(PetscObject V, PetscInt nfields, PetscObject Vfield[], void *ctx)
{
  DGNetworkMonitorList_Glvis node    = (DGNetworkMonitorList_Glvis)ctx;
  DGNetwork                  dgnet   = node->dgnet;
  DM                         network = dgnet->network;
  EdgeFE                     edgefe;
  PetscInt                   copy, c, cStart, cEnd, field, tab, dof = dgnet->physics.dof, i, fieldoff, deg, ndegree, e, eStart, eEnd, cCount, off_e;
  PetscSection               section;
  const PetscReal           *v;
  PetscReal                 *vwork;
  PetscInt                   Dim = 2;
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead((Vec)V, &v));
  PetscCall(DMNetworkGetEdgeRange(network, &eStart, &eEnd));
  cCount = 0;
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
    PetscCall(DMNetworkGetLocalVecOffset(network, e, FVEDGE, &off_e));
    PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
    PetscCall(DMGetSection(edgefe->dm, &section));
    /* Deep copy the data from Field field from V to Vfield. Also changing basis to closed  uniform evaluation basis */
    for (field = 0; field < dof; field++) {
      tab     = dgnet->fieldtotab[field];
      ndegree = dgnet->taborder[tab];
      i       = cCount * PetscPowInt(ndegree + 1, Dim);
      PetscCall(VecGetArray((Vec)Vfield[field], &vwork));
      for (c = cStart; c < cEnd; c++) {
        PetscCall(PetscSectionGetFieldOffset(section, c, field, &fieldoff));
        for (deg = 0; deg <= ndegree; deg++) {
          vwork[i] = evalviewpt_internal(dgnet, field, deg, v + fieldoff + off_e);
          for (copy = 1; copy < (ndegree + 1); copy++) { vwork[i + copy * (ndegree + 1)] = vwork[i]; }
          i++;
        }
        i += (ndegree + 1) * ((ndegree + 1) - 1);
      }
      PetscCall(VecRestoreArray((Vec)Vfield[field], &vwork));
    }
    cCount += cEnd - cStart;
  }

  PetscCall(VecRestoreArrayRead((Vec)V, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkCreateNetworkDMPlex_2D(DGNetwork dgnet, const PetscInt edgelist[], PetscInt edgelistsize, DM *dmsum, PetscSection *stratumoffset, DM **dm_list, PetscInt *numdm)
{
  PetscInt        i = 0, e, eStart, eEnd, cStart, cEnd;
  PetscInt        vfrom, vto;
  DM             *dmlist, network = dgnet->network, dmunion, dmtemp;
  const PetscInt *cone;
  EdgeFE          edgefe;
  DGNETJunction   junct;
  PetscReal       lower[2], upper[2];
  PetscReal       thickness, z[2], n[2], norm = 0.0;
  PetscSection    stratumoff;

  PetscFunctionBegin;
  if (edgelist == NULL) { /* Assume the entire network is used */
    PetscCall(DMNetworkGetEdgeRange(network, &eStart, &eEnd));
    PetscCall(PetscMalloc1(eEnd - eStart, &dmlist));

    thickness = dgnet->edgethickness <= 0 ? 0.05 * dgnet->length : dgnet->edgethickness;
    for (e = eStart; e < eEnd; e++) {
      PetscCall(DMNetworkGetComponent(network, e, FVEDGE, NULL, (void **)&edgefe, NULL));
      PetscCall(DMPlexGetHeightStratum(edgefe->dm, 0, &cStart, &cEnd));
      PetscCall(DMNetworkGetConnectedVertices(network, e, &cone));
      vto               = cone[0];
      vfrom             = cone[1];
      PetscInt faces[2] = {cEnd - cStart, 1};

      PetscCall(DMNetworkGetComponent(network, vfrom, DGNETJUNCTION, NULL, (void **)&junct, NULL));
      z[1]     = junct->y;
      z[0]     = junct->x;
      upper[0] = junct->x;
      upper[1] = junct->y;
      PetscCall(DMNetworkGetComponent(network, vto, DGNETJUNCTION, NULL, (void **)&junct, NULL));
      z[1] -= junct->y;
      z[0] -= junct->x;
      norm = PetscSqrtReal(z[1] * z[1] + z[0] * z[0]);
      z[0] /= norm;
      z[1] /= norm;
      n[0] = -z[1];
      n[1] = z[0];

      lower[0] = junct->x;
      lower[1] = junct->y;
      lower[0] -= thickness * n[0];
      lower[1] -= thickness * n[1];
      upper[0] -= thickness * n[0];
      upper[1] -= thickness * n[1];

      PetscCall(DMPlexCreateEmbeddedLineMesh(PETSC_COMM_SELF, 2, faces[0], lower, upper, &dmtemp));
      PetscCall(DGNetworkCreateViewDM2(dmtemp));
      PetscCall(DMPlexExtrude(dmtemp, 1, thickness * 2, PETSC_FALSE, PETSC_TRUE, NULL, NULL, &dmlist[i]));
      PetscCall(DMDestroy(&dmtemp));
      i++;
    }
    *numdm = i;
    PetscCall(DMPlexDisjointUnion_Geometric_Section(dmlist, i, &dmunion, &stratumoff));
    /* in theory the coordinates are now mapped correctly ... we shall see */
    *dm_list = dmlist;
    *dmsum   = dmunion;
    if (stratumoff) {
      *stratumoffset = stratumoff;
    } else {
      PetscCall(PetscSectionDestroy(&stratumoff));
    }
  } else {
    /* TODO */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DGNetworkMonitorAdd_Glvis_2D_NET(DGNetworkMonitor_Glvis monitor, const char hostname[], PetscViewerGLVisType type)
{
  PetscMPIInt                rank, size;
  DGNetworkMonitorList_Glvis node;
  PetscInt                   viewsize, field, cStart, cEnd, tab, Dim = 2, i;
  DGNetwork                  dgnet = monitor->dgnet;
  PetscInt                   dof   = dgnet->physics.dof;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(PetscMalloc1(1, &node));
  PetscCall(PetscMalloc3(dof, &node->dim, dof, &node->v_work, dof, &node->fec_type));

  PetscCall(PetscViewerGLVisOpen(monitor->comm, type, hostname, PETSC_DECIDE, &node->viewer));
  /* Create the network mesh */
  PetscCall(DGNetworkCreateNetworkDMPlex_2D(dgnet, NULL, 0, &node->viewdm, &node->stratumoffset, &node->dmlist, &node->numdm));
  /* delete the unneeded dms */
  for (i = 0; i < node->numdm; i++) { PetscCall(DMDestroy(&node->dmlist[i])); }
  PetscCall(PetscFree(node->dmlist));
  PetscCall(PetscSectionDestroy(&node->stratumoffset));
  PetscCall(DMPlexGetHeightStratum(node->viewdm, 0, &cStart, &cEnd));
  /* make the work vector for each field */
  for (field = 0; field < dof; field++) {
    /* Setup vector storage for drawing. */
    tab      = dgnet->fieldtotab[field];
    viewsize = (cEnd - cStart) * PetscPowInt((dgnet->taborder[tab] + 1), Dim); /* number of variables for the given field */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, viewsize, &(node->v_work[field])));
    PetscCall(PetscObjectSetName((PetscObject)node->v_work[field], dgnet->physics.fieldname[field]));         /* set the name of the vector for file writing viewing */
    PetscCall(PetscObjectCompose((PetscObject)node->v_work[field], "__PETSc_dm", (PetscObject)node->viewdm)); /* Hack to associate the viewing dm with each work vector for glvis visualization */
    PetscCall(PetscMalloc(64, &node->fec_type[field]));
    PetscCall(PetscSNPrintf(node->fec_type[field], 64, "FiniteElementCollection: L2_T4_%iD_P%i", Dim, dgnet->taborder[tab]));
    node->dim[field] = Dim;
  }

  node->next         = monitor->firstnode;
  node->dgnet        = monitor->dgnet;
  node->snapid       = 0;
  node->v            = NULL;
  monitor->firstnode = node;

  PetscCall(PetscViewerGLVisSetFields(node->viewer, dof, (const char **)node->fec_type, node->dim, DGNetworkMonitor_2D_NET_g2l_internal, (PetscObject *)node->v_work, (void *)node, DGNetworkMonitor_destroyctx_internal));
  PetscFunctionReturn(PETSC_SUCCESS);
}
