
#if !defined(__TCPTHREADIMPLH)
#define __TCPTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

#if defined(PETSC_HAVE_PTHREAD_H)
#include <pthread.h>
#elif defined(PETSC_HAVE_WINPTHREADS_H)
#include <winpthreads.h>       /* http://locklessinc.com/downloads/winpthreads.h */
#endif

/*
   PetscThreadComm_PThread - The main data structure to manage the thread
   communicator using pthreads. This data structure is shared by NONTHREADED
   and PTHREAD threading models. For NONTHREADED threading model, no extra
   pthreads are created
*/
struct _p_PetscThreadComm_PThread {
  pthread_t      *tid;                       /* thread ids */
  pthread_attr_t *attr;                      /* thread attributes */
  pthread_barrier_t barr;                    /* pthread barrier */
  pthread_mutex_t threadmutex;               /* mutex for nthreads variable */
};
typedef struct _p_PetscThreadComm_PThread *PetscThreadComm_PThread;

#if defined(PETSC_PTHREAD_LOCAL)
extern PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank; /* Rank of the calling thread ... thread local variable */
#else
extern pthread_key_t PetscPThreadRankkey;
#endif

PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadAuto(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadPool);
extern PetscErrorCode PetscThreadCommInitialize_PThreadUser(PetscThreadComm);
extern PetscErrorCode PetscThreadCommFinalize_PThread(PetscThreadComm);
extern PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm,PetscThreadCommJobCtx);
extern PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm);
extern PetscErrorCode PetscThreadCommAtomicIncrement_PThread(PetscThreadComm,PetscInt*,PetscInt);

#endif
