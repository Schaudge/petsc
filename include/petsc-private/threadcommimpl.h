
#ifndef __THREADCOMMIMPL_H
#define __THREADCOMMIMPL_H

#include <petscthreadcomm.h>
#include <petsc-private/petscimpl.h>

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif
#if defined(PETSC_HAVE_WINDOWS_H)
#include <windows.h>
#endif

PETSC_EXTERN PetscMPIInt Petsc_ThreadComm_keyval;

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10

/* Reduction status of threads */
#define THREADCOMM_THREAD_WAITING_FOR_NEWRED 0
#define THREADCOMM_THREAD_POSTED_LOCALRED    1
/* Status of the reduction */
#define THREADCOMM_REDUCTION_NONE           -1
#define THREADCOMM_REDUCTION_NEW             0
#define THREADCOMM_REDUCTION_COMPLETE        1

/* Job status for threads */
#define THREAD_JOB_NONE       -1
#define THREAD_JOB_POSTED      1
#define THREAD_JOB_RECIEVED    2
#define THREAD_JOB_COMPLETED   0

/* Thread status */
#define THREAD_TERMINATE      0
#define THREAD_INITIALIZED    1
#define THREAD_CREATED        0

/* Thread model */
#define THREAD_MODEL_LOOP   0
#define THREAD_MODEL_AUTO   1
#define THREAD_MODEL_USER   2

#define PetscReadOnce(type,val) (*(volatile type *)&val)

#if defined(PETSC_MEMORY_BARRIER)
#define PetscMemoryBarrier() do {PETSC_MEMORY_BARRIER();} while(0)
#else
#define PetscMemoryBarrier()
#endif
#if defined(PETSC_READ_MEMORY_BARRIER)
#define PetscReadMemoryBarrier() do {PETSC_READ_MEMORY_BARRIER();} while(0)
#else
#define PetscReadMemoryBarrier()
#endif
#if defined(PETSC_WRITE_MEMORY_BARRIER)
#define PetscWriteMemoryBarrier() do {PETSC_WRITE_MEMORY_BARRIER();} while(0)
#else
#define PetscWriteMemoryBarrier()
#endif

#if defined(PETSC_CPU_RELAX)
#define PetscCPURelax() do {PETSC_CPU_RELAX();} while (0)
#else
#define PetscCPURelax() do { } while (0)
#endif

typedef enum {PTHREADAFFPOLICY_ALL,PTHREADAFFPOLICY_ONECORE,PTHREADAFFPOLICY_NONE} PetscPThreadCommAffinityPolicyType;
extern const char *const PetscPTheadCommAffinityPolicyTypes[];

typedef enum {PTHREADPOOLSPARK_SELF} PetscThreadPoolSparkType;
extern const char *const PetscThreadPoolSparkTypes[];

typedef struct _p_PetscThreadCommRedCtx *PetscThreadCommRedCtx;
struct _p_PetscThreadCommRedCtx{
  PetscThreadComm               tcomm;          /* The associated threadcomm */
  PetscInt                      red_status;     /* Reduction status */
  PetscInt                      *thread_status; /* Reduction status of each thread */
  void                          *local_red;     /* Array to hold local reduction contribution from each thread */
  PetscThreadCommReductionOp    op;             /* The reduction operation */
  PetscDataType                 type;           /* The reduction data type */
};

struct _p_PetscThreadCommReduction{
  PetscInt              nreds;                              /* Number of reductions in operation */
  PetscThreadCommRedCtx redctx;                             /* Reduction objects */
  PetscInt               ctr;                               /* Global Reduction counter */
  PetscInt              *thread_ctr;                        /* Reduction counter for each thread */
};

typedef struct _p_PetscThreadCommJobCtx* PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscThreadComm   tcomm;                         /* The thread communicator */
  PetscInt          commrank;                      /* Rank of thread in communicator */
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
  PetscScalar       scalars[3];                    /* Array to hold three scalar values */
  PetscInt          ints[3];                       /* Array to hold three integer values */
  PetscInt          job_status;                   /* Thread job status */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue* PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt ctr;                      /* Job counter */
  PetscInt kernel_ctr;               /* Kernel counter .. need this otherwise race conditions are unavoidable */
  PetscThreadCommJobCtx jobs;        /* Queue of jobs */
};

typedef struct _PetscThreadCommOps* PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(PetscThreadComm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*kernelbarrier)(PetscThreadComm);
  PetscErrorCode (*globalbarrier)();
  PetscErrorCode (*atomicincrement)(PetscThreadComm,PetscInt*,PetscInt);
  PetscErrorCode (*getrank)(PetscInt*);
  // Create threads and put in ThreadPool
  PetscErrorCode (*createthreads)(PetscThreadComm);
  PetscErrorCode (*destroythreads)(PetscThreadComm);
};

typedef struct _p_PetscThread* PetscThread;
struct _p_PetscThread{
  PetscInt              grank;   /* Thread rank in pool */
  PetscThreadComm       tcomm;   /* Thread comm for current thread */
  PetscInt              status;  /* Status of current job for each thread */
  PetscThreadCommJobCtx jobdata; /* Data for current job for each thread */

  // Job information
  PetscThreadCommJobQueue jobqueue;     /* Job queue */
  PetscInt                job_ctr;      /* which job is this threadcomm running in the job queue */
  PetscInt                my_job_counter;
  PetscInt                my_kernel_ctr;
  PetscInt                glob_kernel_ctr;
};

struct _p_PetscThreadPool{
  PetscInt                refct;           /* Number of ThreadComm references */
  PetscInt                npoolthreads;    /* Max number of threads pool can hold */
  PetscInt                *affinities;     /* Core affinity of each thread */
  void                    *data;           /* Implementation specific data */
  PetscThread             *poolthreads;    /* Array of all threads */
};

struct _p_PetscThreadComm{
  // General threadcomm information
  PetscInt                 refct;        /* Number of MPI_Comm references */
  PetscInt                 leader;       /* Rank of the leader thread. This thread manages
                                           the synchronization for collective operatons like reductions. */
  PetscBool                isnothread;   /* No threading model used */
  PetscInt                 thread_start; /* Index for the first created thread (=1 if main thread is a worker, else 0 */
  PetscInt                 model;        /* Threading model used */
  PetscThreadCommReduction red;          /* Reduction context */
  PetscBool                active;       /* Does this threadcomm have access to the threads? */
  PetscThreadCommOps       ops;          /* Operations table */
  char                     type[256];    /* Thread model type */

  // User input options
  PetscThreadPoolSparkType spark;            /* Type for sparking threads */
  PetscPThreadCommAffinityPolicyType  aff;   /* affinity policy */
  PetscBool                synchronizeafter; /* Whether the main thread should block until all threads complete kernel */
  PetscInt                 nkernels;     /* Maximum kernels launched */
  PetscBool                ismainworker; /* Is the main thread also a work thread? */

  // Thread information
  PetscThreadPool         pool;        /* Threadpool containing threads for this comm */
  PetscInt                ncommthreads; /* Max threads comm can use */
  PetscInt                nthreads;     /* Number of active threads available to comm */
  PetscThread             *commthreads; /* Threads that this comm can use */

  // User barrier
  PetscInt                barrier_threads;
  PetscBool               wait1, wait2;
};

/* register thread communicator models */
PETSC_EXTERN PetscErrorCode PetscThreadPoolModelRegister(const char[],PetscErrorCode(*)(PetscThreadPool));
PETSC_EXTERN PetscErrorCode PetscThreadPoolTypeRegister(const char[],PetscErrorCode(*)(PetscThreadPool));
PETSC_EXTERN PetscErrorCode PetscThreadPoolRegisterAllModels(void);
PETSC_EXTERN PetscErrorCode PetscThreadPoolRegisterAllTypes(PetscThreadPool pool);

#undef __FUNCT__
#define __FUNCT__
PETSC_STATIC_INLINE PetscErrorCode PetscRunKernel(PetscInt trank,PetscInt nargs,PetscThreadCommJobCtx job)
{
  printf("Running kernel with trank=%d\n",trank);
  switch(nargs) {
  case 0:
    (*job->pfunc)(trank);
    break;
  case 1:
    (*job->pfunc)(trank,job->args[0]);
    break;
  case 2:
    (*job->pfunc)(trank,job->args[0],job->args[1]);
    break;
  case 3:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2]);
    break;
  case 4:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3]);
    break;
  case 5:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4]);
    break;
  case 6:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5]);
    break;
  case 7:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6]);
    break;
  case 8:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7]);
    break;
  case 9:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8]);
    break;
  case 10:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8],job->args[9]);
    break;
  }
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommReduction);

PETSC_EXTERN PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;
#endif
