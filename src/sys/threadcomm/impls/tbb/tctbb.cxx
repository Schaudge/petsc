#include <../src/sys/threadcomm/impls/tbb/tctbbimpl.h>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

using namespace tbb;

class TBBRunKernel {
  PetscThreadCommJobCtx job;

public:
  void operator()(blocked_range<size_t>& r) const {
    PetscInt trank= r.begin();
    job->job_status = THREAD_JOB_RECIEVED;
    PetscRunKernel(trank,job->nargs,job);
    job->job_status = THREAD_JOB_COMPLETED;
  }

  TBBRunKernel(PetscThreadCommJobCtx ijob) : job(ijob) {}
};

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_TBB"
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(pool->model==THREAD_MODEL_AUTO || pool->model==THREAD_MODEL_USER) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to use auto or user thread model with TBB. Use loop model with TBB");

  ierr = PetscStrcpy(pool->type,TBB);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_TBB;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_TBB"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  tcomm->ops->runkernel = PetscThreadCommRunKernel_TBB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_TBB"
PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscFunctionBegin;
  task_scheduler_init init(tcomm->ncommthreads);
  parallel_for(blocked_range<size_t>(0,tcomm->ncommthreads,1),TBBRunKernel(job));
  PetscFunctionReturn(0);
}
