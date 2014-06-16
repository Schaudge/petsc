/* Define feature test macros to make sure CPU_SET and other functions are available
 */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined PETSC_HAVE_MALLOC_H
#include <malloc.h>
#endif

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

static PetscBool PetscPThreadCommInitializeCalled = PETSC_FALSE;

static PetscInt ptcommcrtct = 0; /* PThread communicator creation count. Incremented whenever a pthread
                                    communicator is created and decremented when it is destroyed. On the
                                    last pthread communicator destruction, the thread pool is also terminated
                                  */

PetscErrorCode PetscThreadCommGetRank_PThread(PetscInt *trank)
{
#if defined(PETSC_PTHREAD_LOCAL)
  *trank = PetscPThreadRank;
#else
  *trank = *((PetscInt*)pthread_getspecific(PetscPThreadRankkey));
#endif
  return 0;
}

/* Sets the attributes for threads */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_PThread"
PetscErrorCode PetscThreadCommSetAffinity_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  pthread_attr_t          *attr =ptcomm->attr;
  PetscBool               set;
  PetscThreadPool         pool = PETSC_THREAD_POOL;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t               *cpuset;
#endif
  PetscInt                i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  /* Set affinity for workers */
  ierr = PetscMalloc1(pool->npoolthreads,&cpuset);CHKERRQ(ierr);
  for (i=tcomm->thread_start; i<pool->npoolthreads; i++) {
    ierr = pthread_attr_init(&attr[i]);CHKERRQ(ierr);
    PetscThreadPoolSetAffinity(pool,&cpuset[i],i,&set);
    if(set) pthread_attr_setaffinity_np(&attr[i],sizeof(cpu_set_t),&cpuset[i]);
  }

  /* Set affinity for main thread */
  if (pool->ismainworker) {
    PetscThreadPoolSetAffinity(pool,&cpuset[0],0,&set);
    sched_setaffinity(0,sizeof(cpu_set_t),&cpuset[0]);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!ptcomm) PetscFunctionReturn(0);
  ptcommcrtct--;
  if (!ptcommcrtct) {
    /* Terminate the thread pool */
    if(pool->model==THREAD_MODEL_LOOP) {
      ierr = PetscThreadCommFinalize_PThread(tcomm);CHKERRQ(ierr);
      ierr = PetscFree(ptcomm->tid);CHKERRQ(ierr);
      ierr = PetscFree(ptcomm->attr);CHKERRQ(ierr);
    }
    if(pool->model==THREAD_MODEL_USER) {
      ierr = pthread_barrier_destroy(&ptcomm->barr);CHKERRQ(ierr);
      ierr = pthread_mutex_destroy(&ptcomm->threadmutex);CHKERRQ(ierr);
    }
    PetscPThreadCommInitializeCalled = PETSC_FALSE;
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadLoop"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadComm tcomm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread Loop\n");
  ptcommcrtct++;
  ierr = PetscStrcpy(pool->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  tcomm->data              = (void*)ptcomm;
  pool->ops->destroy      = PetscThreadCommDestroy_PThread;
  pool->ops->runkernel    = PetscThreadCommRunKernel_PThread;
  pool->ops->kernelbarrier = PetscThreadPoolBarrier;
  pool->ops->getrank      = PetscThreadCommGetRank_PThread;

  if (!PetscPThreadCommInitializeCalled) { /* Only done for PETSC_THREAD_COMM_WORLD */
    PetscPThreadCommInitializeCalled = PETSC_TRUE;

    if (pool->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
      PetscPThreadRank=0; /* Main thread rank */
#else
      ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
      ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->pool->granks[0]);CHKERRQ(ierr);
#endif
    }

    /* Create array holding pthread ids */
    ierr = PetscMalloc1(tcomm->ncommthreads,&ptcomm->tid);CHKERRQ(ierr);
    /* Create thread attributes */
    ierr = PetscMalloc1(tcomm->ncommthreads,&ptcomm->attr);CHKERRQ(ierr);
    ierr = PetscThreadCommSetAffinity_PThread(tcomm);CHKERRQ(ierr);

    /* Initialize thread pool */
    ierr = PetscThreadCommInitialize_PThread(tcomm);CHKERRQ(ierr);

  } else {
    PetscThreadComm         gtcomm;
    PetscThreadComm_PThread gptcomm;

    ierr        = PetscCommGetThreadComm(PETSC_COMM_WORLD,&gtcomm);CHKERRQ(ierr);
    gptcomm     = (PetscThreadComm_PThread)tcomm->data;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadUser"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadComm tcomm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread User\n");
  ptcommcrtct++;
  ierr = PetscStrcpy(pool->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  pthread_barrier_init(&ptcomm->barr,NULL,tcomm->ncommthreads);
  pthread_mutex_init(&ptcomm->threadmutex,NULL);

  tcomm->data              = (void*)ptcomm;
  pool->ops->destroy      = PetscThreadCommDestroy_PThread;
  pool->ops->runkernel    = PetscThreadCommRunKernel_PThread;
  pool->ops->kernelbarrier = PetscThreadPoolBarrier;
  pool->ops->globalbarrier = PetscThreadCommBarrier_PThread;
  pool->ops->atomicincrement = PetscThreadCommAtomicIncrement_PThread;
  pool->ops->getrank      = PetscThreadCommGetRank_PThread;

  if (pool->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->pool->granks[0]);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread"
PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread ptcomm;
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  if (pool->ismainworker) {
    job->job_status[0]   = THREAD_JOB_RECIEVED;
    jobqueue->tinfo[0]->data = job;
    PetscRunKernel(0,job->nargs, jobqueue->tinfo[0]->data);
    job->job_status[0]   = THREAD_JOB_COMPLETED;
  }
  if (pool->synchronizeafter) {
    ierr = (*pool->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize_PThread"
PetscErrorCode PetscThreadCommInitialize_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;

  PetscFunctionBegin;
  /* Create threads */
  for (i=tcomm->thread_start; i < tcomm->ncommthreads; i++) {
    printf("Creating thread=%d\n",i);
    jobqueue->tinfo[i]->status = THREAD_CREATED;
    jobqueue->tinfo[i]->rank = pool->granks[i];
    jobqueue->tinfo[i]->tcomm = tcomm;
    ierr = pthread_create(&ptcomm->tid[i],&ptcomm->attr[i],&PetscThreadPoolFunc,&jobqueue->tinfo[i]);CHKERRQ(ierr);
  }

  if (pool->ismainworker) jobqueue->tinfo[0]->status = THREAD_INITIALIZED;

  PetscInt threads_initialized=0;
  /* Wait till all threads have been initialized */
  while (threads_initialized != tcomm->ncommthreads) {
    threads_initialized=0;
    for (i=0; i<tcomm->ncommthreads; i++) {
      if (!jobqueue->tinfo[pool->granks[i]]->status) break;
      threads_initialized++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalize_PThread"
PetscErrorCode PetscThreadCommFinalize_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  void                    *jstatus;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;
  PetscInt                i;

  PetscFunctionBegin;
  ierr = (*pool->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
  for (i=tcomm->thread_start; i < tcomm->ncommthreads; i++) {
    printf("Terminating thread=%d\n",i);
    jobqueue->tinfo[i]->status = THREAD_TERMINATE;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread"
PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;
  pthread_barrier_wait(&ptcomm->barr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAtomicIncrement_PThread"
PetscErrorCode PetscThreadCommAtomicIncrement_PThread(PetscThreadComm tcomm,PetscInt *val,PetscInt inc)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;
  pthread_mutex_lock(&ptcomm->threadmutex);
  (*val)+=inc;
  pthread_mutex_unlock(&ptcomm->threadmutex);
  PetscFunctionReturn(0);
}
