
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

/* TSHistory is an helper object that allows inquiring
   the TSTrajectory by time and not by the step number only
   this can be moved to TS when consolidated, or at
   least shared by the different TSTrajectory implementations */
struct _n_TSHistory {
  PetscReal   *hist;    /* time history */
  PetscInt    *hist_id; /* stores the stepid in time history */
  PetscInt    n;        /* current number of steps stored */
  PetscBool   sorted;   /* if the history is sorted in ascending order */
  PetscReal   c;        /* current capacity of hist */
  PetscReal   s;        /* reallocation size */
};
typedef struct _n_TSHistory* TSHistory;

static PetscErrorCode TSHistoryUpdate(TSHistory tsh, PetscInt id, PetscReal time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tsh->n == tsh->c) { /* reallocation */
    tsh->c += tsh->s;
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist),&tsh->hist);CHKERRQ(ierr);
    ierr = PetscRealloc(tsh->c*sizeof(*tsh->hist_id),&tsh->hist_id);CHKERRQ(ierr);
  }
  tsh->sorted = (PetscBool)(tsh->sorted && (tsh->n ? time >= tsh->hist[tsh->n-1] : PETSC_TRUE));
#if defined(PETSC_USE_DEBUG)
  if (tsh->n) { /* id should be unique */
    PetscInt loc,*ids;

    ierr = PetscMalloc1(tsh->n,&ids);CHKERRQ(ierr);
    ierr = PetscMemcpy(ids,tsh->hist_id,tsh->n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscSortInt(tsh->n,ids);CHKERRQ(ierr);
    ierr = PetscFindInt(id,tsh->n,ids,&loc);CHKERRQ(ierr);
    ierr = PetscFree(ids);CHKERRQ(ierr);
    if (loc >=0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"History id should be unique");
  }
#endif
  tsh->hist[tsh->n]    = time;
  tsh->hist_id[tsh->n] = id;
  tsh->n += 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHistoryGetLocFromTime(TSHistory tsh, PetscReal time, PetscInt *loc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tsh->sorted) {
    ierr = PetscSortRealWithArrayInt(tsh->n,tsh->hist,tsh->hist_id);CHKERRQ(ierr);
    tsh->sorted = PETSC_TRUE;
  }
  ierr = PetscFindReal(time,tsh->n,tsh->hist,PETSC_SMALL,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHistoryDestroy(TSHistory tsh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tsh->hist);CHKERRQ(ierr);
  ierr = PetscFree(tsh->hist_id);CHKERRQ(ierr);
  ierr = PetscFree(tsh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* these two functions are stolen from bdf.c */
PETSC_STATIC_INLINE void LagrangeBasisVals(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar L[])
{
  PetscInt k,j;
  for (k=0; k<n; k++)
    for (L[k]=1, j=0; j<n; j++)
      if (j != k)
        L[k] *= (t - T[j])/(T[k] - T[j]);
}

PETSC_STATIC_INLINE void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
{
  PetscInt  k,j,i;
  for (k=0; k<n; k++)
    for (dL[k]=0, j=0; j<n; j++)
      if (j != k) {
        PetscReal L = 1/(T[k] - T[j]);
        for (i=0; i<n; i++)
          if (i != j && i != k)
            L *= (t - T[i])/(T[k] - T[i]);
        dL[k] += L;
      }
}

PETSC_STATIC_INLINE PetscInt LagrangeGetId(PetscReal t, PetscInt n, const PetscReal T[], const PetscBool Taken[])
{
  PetscInt _tid = 0;
  while (_tid < n && PetscAbsReal(t-T[_tid]) > PETSC_SMALL) _tid++;
  if (_tid < n && !Taken[_tid]) {
    return _tid;
  } else { /* we get back a negative id, where the maximum time is stored, since we use usually reconstruct backward in time */
    PetscReal max = PETSC_MIN_REAL;
    PetscInt maxloc = n;
    _tid = 0;
    while (_tid < n) { maxloc = (max < T[_tid] && !Taken[_tid]) ? (max = T[_tid],_tid) : maxloc; _tid++; }
    return -maxloc-1;
  }
}

typedef struct {
  /* output */
  PetscViewer viewer;
  char        *folder;
  char        *basefilename;
  char        *ext;
  PetscBool   dumpstages;

  /* strategy for computing from missing time */
  PetscInt    order;  /* interpolation order. if negative, recompute */
  Vec         *W;     /* work vectors */
  PetscScalar *L;     /* workspace for Lagrange basis */
  PetscReal   *T;     /* Lagrange times (stored) */
  Vec         *WW;    /* just an array of pointers */
  PetscBool   *TT;    /* workspace for Lagrange */
  PetscReal   *TW;    /* Lagrange times (workspace) */

  /* history */
  TSHistory   tsh;
} TSTrajectory_Basic;

static PetscErrorCode TSTrajectoryDestroy_Basic(TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->folder);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->basefilename);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic->ext);CHKERRQ(ierr);
  ierr = TSHistoryDestroy(tjbasic->tsh);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tjbasic->order+1,&tjbasic->W);CHKERRQ(ierr);
  ierr = PetscFree5(tjbasic->L,tjbasic->T,tjbasic->WW,tjbasic->TT,tjbasic->TW);CHKERRQ(ierr);
  ierr = PetscFree(tjbasic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tj,&comm);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscRMTree(tjbasic->folder);CHKERRQ(ierr);
      ierr = PetscMkdir(tjbasic->folder);CHKERRQ(ierr);
    }
    ierr = PetscBarrier((PetscObject)tj);CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tjbasic->viewer,filename);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(tjbasic->viewer);CHKERRQ(ierr);
  ierr = VecView(X,tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(tjbasic->viewer,&time,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);

  if (stepnum && tjbasic->dumpstages) {
    Vec       *Y;
    PetscReal tprev;
    PetscInt  ns,i;

    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
      ierr = VecView(Y[i],tjbasic->viewer);CHKERRQ(ierr);
    }

    ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(tjbasic->viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = TSHistoryUpdate(tjbasic->tsh,stepnum,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGetVecs_Basic(TSTrajectory,TS,PetscInt,PetscReal*,Vec,Vec);

static PetscErrorCode TSTrajectoryBasicReconstruct_Private(TSTrajectory tj,PetscReal t,Vec U,Vec Udot)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  TSHistory          tsh = tjbasic->tsh;
  PetscInt           id, cnt, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TSHistoryGetLocFromTime(tsh,t,&id);CHKERRQ(ierr);
  if (id == -1 || id == -tsh->n - 1) {
    PetscReal t0 = tsh->n ? tsh->hist[0]        : 0.0;
    PetscReal tf = tsh->n ? tsh->hist[tsh->n-1] : 0.0;
    SETERRQ4(PetscObjectComm((PetscObject)tj),PETSC_ERR_PLIB,"Requested time %g is outside the history interval [%g, %g] (%d)",t,t0,tf,tsh->n);
  }
  if (!tjbasic->T) {
    PetscInt o = tjbasic->order+1;
    ierr = PetscMalloc5(o,&tjbasic->L,o,&tjbasic->T,o,&tjbasic->WW,o,&tjbasic->TT,o,&tjbasic->TW);CHKERRQ(ierr);
    for (i = 0; i < o; i++) tjbasic->T[i] = PETSC_MAX_REAL;
    ierr = VecDuplicateVecs(U,o,&tjbasic->W);CHKERRQ(ierr);
  }
  cnt = 0;
  ierr = PetscMemzero(tjbasic->TT,(tjbasic->order+1)*sizeof(PetscBool));CHKERRQ(ierr);
  if (id < 0 || Udot) {
    PetscInt s,nid = id < 0 ? -(id+1) : id;

    PetscInt up = PetscMin(nid + tjbasic->order/2+1,tsh->n);
    PetscInt low = PetscMax(up-tjbasic->order-1,0);
    up = PetscMin(PetscMax(low + tjbasic->order + 1,up),tsh->n);
    for (s = up-1; s >= low; s--) {
      PetscReal t = tsh->hist[s];
      PetscInt tid = LagrangeGetId(t,tjbasic->order+1,tjbasic->T,tjbasic->TT);
      if (tid < 0) {
        tid  = -tid-1;
        ierr = TSTrajectoryGetVecs_Basic(tj,NULL,tsh->hist_id[s],&t,tjbasic->W[tid],NULL);CHKERRQ(ierr);
        tjbasic->T[tid] = t;
      }
      tjbasic->TT[tid] = PETSC_TRUE;
      tjbasic->WW[cnt] = tjbasic->W[tid];
      tjbasic->TW[cnt] = t;
      cnt++;
    }
  }
  ierr = PetscMemzero(tjbasic->TT,(tjbasic->order+1)*sizeof(PetscBool));CHKERRQ(ierr);
  if (id >=0 && U) { /* requested time match */
    PetscInt tid = LagrangeGetId(t,tjbasic->order+1,tjbasic->T,tjbasic->TT);
    if (tid < 0) {
      tid  = -tid-1;
      ierr = TSTrajectoryGetVecs_Basic(tj,NULL,tsh->hist_id[id],&t,tjbasic->W[tid],NULL);CHKERRQ(ierr);
      tjbasic->T[tid] = t;
    }
    ierr = VecCopy(tjbasic->W[tid],U);CHKERRQ(ierr);
  }
  if (id < 0 && U) {
    LagrangeBasisVals(cnt,t,tjbasic->TW,tjbasic->L);
    ierr = VecZeroEntries(U);CHKERRQ(ierr);
    ierr = VecMAXPY(U,cnt,tjbasic->L,tjbasic->WW);CHKERRQ(ierr);
  }
  if (Udot) {
    LagrangeBasisDers(cnt,t,tjbasic->TW,tjbasic->L);
    ierr = VecZeroEntries(Udot);CHKERRQ(ierr);
    ierr = VecMAXPY(Udot,cnt,tjbasic->L,tjbasic->WW);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,TSTrajectory tj)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"TS trajectory options for Basic type");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ts_trajectory_basic_order","Interpolation order for reconstruction",NULL,tjbasic->order,&tjbasic->order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_basic_dumpstages","Dump stages during TSTrajectorySet",NULL,tjbasic->dumpstages,&tjbasic->dumpstages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGetVecs_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t,Vec U,Vec Udot)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  PetscErrorCode     ierr;
  PetscViewer        viewer;
  char               filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (stepnum < 0 || Udot) { /* reverse search for requested time in TSHistory */
    ierr = TSTrajectoryBasicReconstruct_Private(tj,*t,U,Udot);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* we were asked to load from stepnum */
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(U,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  TSTrajectory_Basic *tjbasic = (TSTrajectory_Basic*)tj->data;
  PetscViewer        viewer;
  char               filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;
  Vec                Sol;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"%s/%s-%06d.%s",tjbasic->folder,tjbasic->basefilename,stepnum,tjbasic->ext);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  if (stepnum != 0) {
    Vec         *Y;
    PetscInt    Nr,i;
    PetscReal   timepre;

    ierr = TSGetStages(ts,&Nr,&Y);CHKERRQ(ierr);
    for (i=0;i<Nr ;i++) {
      ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryRead(viewer,&timepre,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-(*t)+timepre);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a seperate file in SA-data/SA-%06d.bin

      This version saves the solutions at all the stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m can read in files created with this format

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj,TS ts)
{
  TSTrajectory_Basic *tjbasic;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&tjbasic);CHKERRQ(ierr);

  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)tj),&tjbasic->viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(tjbasic->viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(tjbasic->viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscStrallocpy("./SA-data",&tjbasic->folder);CHKERRQ(ierr);
  ierr = PetscStrallocpy("SA",&tjbasic->basefilename);CHKERRQ(ierr);
  ierr = PetscStrallocpy("bin",&tjbasic->ext);CHKERRQ(ierr);

  ierr = PetscNew(&tjbasic->tsh);CHKERRQ(ierr);
  tjbasic->tsh->n      = 0;
  tjbasic->tsh->c      = 1000;
  tjbasic->tsh->s      = 1000;
  tjbasic->tsh->sorted = PETSC_TRUE;
  ierr = PetscMalloc1(tjbasic->tsh->c,&tjbasic->tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tjbasic->tsh->c,&tjbasic->tsh->hist_id);CHKERRQ(ierr);

  tjbasic->dumpstages = PETSC_TRUE;
  tjbasic->order      = 1;

  tj->data = tjbasic;

  tj->ops->set            = TSTrajectorySet_Basic;
  tj->ops->get            = TSTrajectoryGet_Basic;
  tj->ops->getvecs        = TSTrajectoryGetVecs_Basic;
  tj->ops->destroy        = TSTrajectoryDestroy_Basic;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Basic;
  PetscFunctionReturn(0);
}
