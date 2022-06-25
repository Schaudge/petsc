/*
   Implements the sequential ViennaCL vectors.
*/

#include <petscconf.h>
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/ocl/backend.hpp"
#endif

PETSC_EXTERN PetscErrorCode VecViennaCLGetArray(Vec v, ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  *a = 0;
  PetscCall(VecViennaCLCopyToGPU(v));
  *a = ((Vec_ViennaCL *)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArray(Vec v, ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayRead(Vec v, const ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  *a = 0;
  PetscCall(VecViennaCLCopyToGPU(v));
  *a = ((Vec_ViennaCL *)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayWrite(Vec v, ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  *a = 0;
  PetscCall(VecViennaCLAllocateCheck(v));
  *a = ((Vec_ViennaCL *)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, ViennaCLVector **a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscViennaCLInit() {
  char      string[20];
  PetscBool flg, flg_cuda, flg_opencl, flg_openmp;

  PetscFunctionBegin;
  /* ViennaCL backend selection: CUDA, OpenCL, or OpenMP */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-viennacl_backend", string, sizeof(string), &flg));
  if (flg) {
    try {
      PetscCall(PetscStrcasecmp(string, "cuda", &flg_cuda));
      PetscCall(PetscStrcasecmp(string, "opencl", &flg_opencl));
      PetscCall(PetscStrcasecmp(string, "openmp", &flg_openmp));

      /* A default (sequential) CPU backend is always available - even if OpenMP is not enabled. */
      if (flg_openmp) viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);
#if defined(PETSC_HAVE_CUDA)
      else if (flg_cuda) viennacl::backend::default_memory_type(viennacl::CUDA_MEMORY);
#endif
#if defined(PETSC_HAVE_OPENCL)
      else if (flg_opencl) viennacl::backend::default_memory_type(viennacl::OPENCL_MEMORY);
#endif
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: Backend not recognized or available: %s.\n Pass -viennacl_view to see available backends for ViennaCL.", string);
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
  }

  if (PetscDefined(HAVE_CUDA)) {
    /* For CUDA event timers */
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  }

#if defined(PETSC_HAVE_OPENCL)
  /* ViennaCL OpenCL device type configuration */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-viennacl_opencl_device_type", string, sizeof(string), &flg));
  if (flg) {
    try {
      PetscCall(PetscStrcasecmp(string, "cpu", &flg));
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_CPU);

      PetscCall(PetscStrcasecmp(string, "gpu", &flg));
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_GPU);

      PetscCall(PetscStrcasecmp(string, "accelerator", &flg));
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_ACCELERATOR);
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
  }
#endif

  /* Print available backends */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-viennacl_view", &flg));
  if (flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ViennaCL backends available: "));
#if defined(PETSC_HAVE_CUDA)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CUDA, "));
#endif
#if defined(PETSC_HAVE_OPENCL)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenCL, "));
#endif
#if defined(PETSC_HAVE_OPENMP)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenMP "));
#else
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenMP (1 thread) "));
#endif
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

    /* Print selected backends */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ViennaCL backend  selected: "));
#if defined(PETSC_HAVE_CUDA)
    if (viennacl::backend::default_memory_type() == viennacl::CUDA_MEMORY) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CUDA ")); }
#endif
#if defined(PETSC_HAVE_OPENCL)
    if (viennacl::backend::default_memory_type() == viennacl::OPENCL_MEMORY) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenCL ")); }
#endif
#if defined(PETSC_HAVE_OPENMP)
    if (viennacl::backend::default_memory_type() == viennacl::MAIN_MEMORY) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenMP ")); }
#else
    if (viennacl::backend::default_memory_type() == viennacl::MAIN_MEMORY) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenMP (sequential - consider reconfiguration: --with-openmp=1) ")); }
#endif
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscFunctionReturn(0);
}

/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscViennaCLFlag for the vector
    Does NOT zero the ViennaCL array
 */
PETSC_EXTERN PetscErrorCode VecViennaCLAllocateCheckHost(Vec v) {
  PetscScalar *array;
  Vec_Seq     *s;
  PetscInt     n = v->map->n;

  PetscFunctionBegin;
  s = (Vec_Seq *)v->data;
  PetscCall(VecViennaCLAllocateCheck(v));
  if (s->array == 0) {
    PetscCall(PetscMalloc1(n, &array));
    PetscCall(PetscLogObjectMemory((PetscObject)v, n * sizeof(PetscScalar)));
    s->array           = array;
    s->array_allocated = array;
  }
  PetscFunctionReturn(0);
}

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscViennaCLFlag for the vector
    Does NOT zero the ViennaCL array

 */
PetscErrorCode VecViennaCLAllocateCheck(Vec v) {
  PetscFunctionBegin;
  if (!v->spptr) {
    try {
      v->spptr                                       = new Vec_ViennaCL;
      ((Vec_ViennaCL *)v->spptr)->GPUarray_allocated = new ViennaCLVector((PetscBLASInt)v->map->n);
      ((Vec_ViennaCL *)v->spptr)->GPUarray           = ((Vec_ViennaCL *)v->spptr)->GPUarray_allocated;

    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecViennaCLCopyToGPU(Vec v) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLAllocateCheck(v));
  if (v->map->n > 0) {
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      PetscCall(PetscLogEventBegin(VEC_ViennaCLCopyToGPU, v, 0, 0, 0));
      try {
        ViennaCLVector *vec = ((Vec_ViennaCL *)v->spptr)->GPUarray;
        viennacl::fast_copy(*(PetscScalar **)v->data, *(PetscScalar **)v->data + v->map->n, vec->begin());
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogCpuToGpu((v->map->n) * sizeof(PetscScalar)));
      PetscCall(PetscLogEventEnd(VEC_ViennaCLCopyToGPU, v, 0, 0, 0));
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
  }
  PetscFunctionReturn(0);
}

/*
     VecViennaCLCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecViennaCLCopyFromGPU(Vec v) {
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLAllocateCheckHost(v));
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    PetscCall(PetscLogEventBegin(VEC_ViennaCLCopyFromGPU, v, 0, 0, 0));
    try {
      ViennaCLVector *vec = ((Vec_ViennaCL *)v->spptr)->GPUarray;
      viennacl::fast_copy(vec->begin(), vec->end(), *(PetscScalar **)v->data);
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuToCpu((v->map->n) * sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(VEC_ViennaCLCopyFromGPU, v, 0, 0, 0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/* Copy on CPU */
static PetscErrorCode VecCopy_SeqViennaCL_Private(Vec xin, Vec yin) {
  PetscScalar       *ya;
  const PetscScalar *xa;

  PetscFunctionBegin;
  PetscCall(VecViennaCLAllocateCheckHost(xin));
  PetscCall(VecViennaCLAllocateCheckHost(yin));
  if (xin != yin) {
    PetscCall(VecGetArrayRead(xin, &xa));
    PetscCall(VecGetArray(yin, &ya));
    PetscCall(PetscArraycpy(ya, xa, xin->map->n));
    PetscCall(VecRestoreArrayRead(xin, &xa));
    PetscCall(VecRestoreArray(yin, &ya));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetRandom_SeqViennaCL_Private(Vec xin, PetscRandom r) {
  PetscInt     n = xin->map->n, i;
  PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArray(xin, &xx));
  for (i = 0; i < n; i++) PetscCall(PetscRandomGetValue(r, &xx[i]));
  PetscCall(VecRestoreArray(xin, &xx));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDestroy_SeqViennaCL_Private(Vec v) {
  Vec_Seq *vs = (Vec_Seq *)v->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsViewOff(v));
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v, "Length=%" PetscInt_FMT, v->map->n);
#endif
  if (vs->array_allocated) PetscCall(PetscFree(vs->array_allocated));
  PetscCall(PetscFree(vs));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecResetArray_SeqViennaCL_Private(Vec vin) {
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

/*MC
   VECSEQVIENNACL - VECSEQVIENNACL = "seqviennacl" - The basic sequential vector, modified to use ViennaCL

   Options Database Keys:
. -vec_type seqviennacl - sets the vector type to VECSEQVIENNACL during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateSeqWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`, `VecCreateSeq()`
M*/

PetscErrorCode VecAYPX_SeqViennaCL(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx) {
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;

  PetscFunctionBegin;
  PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
  PetscCall(VecViennaCLGetArray(yin, &ygpu));
  PetscCall(PetscLogGpuTimeBegin());
  try {
    if (!PetscManagedScalarKnownAndEqual(alpha, 0.0) && xin->map->n > 0) {
      PetscScalar *aptr;

      PetscCall(PetscManagedScalarGetValues(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
      *ygpu = *xgpu + (*aptr) * *ygpu;
      PetscCall(PetscLogGpuFlops(2.0 * yin->map->n));
    } else {
      *ygpu = *xgpu;
    }
    ViennaCLWaitForGPU();
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
  PetscCall(VecViennaCLRestoreArray(yin, &ygpu));
  PetscFunctionReturn(0);
}

// this exists because it is too inefficient to do PetscManagedSubRange() in VecMAXPY_SeqViennaCL()
static PetscErrorCode VecAXPY_SeqViennaCL_Private(Vec yin, PetscScalar alpha, Vec xin) {
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;

  PetscFunctionBegin;
  if (alpha != 0.0 && xin->map->n > 0) {
    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLGetArray(yin, &ygpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      *ygpu += alpha * *xgpu;
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArray(yin, &ygpu));
    PetscCall(PetscLogGpuFlops(2.0 * yin->map->n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqViennaCL(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx) {
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
  PetscCall(VecAXPY_SeqViennaCL_Private(yin, *aptr, xin));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqViennaCL(Vec win, Vec xin, Vec yin, PetscDeviceContext) {
  const ViennaCLVector *xgpu, *ygpu;
  ViennaCLVector       *wgpu;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLGetArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLGetArrayWrite(win, &wgpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      *wgpu = viennacl::linalg::element_div(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(win->map->n));
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLRestoreArrayWrite(win, &wgpu));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqViennaCL(Vec win, PetscManagedScalar ascal, Vec xin, Vec yin, PetscDeviceContext dctx) {
  const ViennaCLVector *xgpu, *ygpu;
  ViennaCLVector       *wgpu;

  PetscFunctionBegin;
  if (PetscManagedScalarKnownAndEqual(ascal, 0.0) && xin->map->n > 0) {
    PetscCall(VecCopy_SeqViennaCL(yin, win, dctx));
  } else {
    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLGetArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLGetArrayWrite(win, &wgpu));
    PetscCall(PetscLogGpuTimeBegin());
    if (PetscManagedScalarKnownAndEqual(ascal, 1.0)) {
      try {
        *wgpu = *ygpu + *xgpu;
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuFlops(win->map->n));
    } else if (PetscManagedScalarKnownAndEqual(ascal, -1.0)) {
      try {
        *wgpu = *ygpu - *xgpu;
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuFlops(win->map->n));
    } else {
      PetscScalar *aptr;
      PetscCall(PetscManagedScalarGetValues(dctx, ascal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
      try {
        *wgpu = *ygpu + (*aptr) * *xgpu;
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuFlops(2 * win->map->n));
    }
    ViennaCLWaitForGPU();
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLRestoreArrayWrite(win, &wgpu));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqViennaCL(Vec xin, Vec yin, PetscManagedScalar zscal, PetscDeviceContext dctx) {
  PetscScalar           z;
  const ViennaCLVector *xgpu, *ygpu;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLGetArrayRead(yin, &ygpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      z = viennacl::linalg::inner_prod(*xgpu, *ygpu);
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    ViennaCLWaitForGPU();
    PetscCall(PetscLogGpuTimeEnd());
    if (xin->map->n > 0) { PetscCall(PetscLogGpuFlops(2.0 * xin->map->n - 1)); }
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArrayRead(yin, &ygpu));
  } else z = 0.0;
  PetscCall(PetscManagedScalarSetValues(dctx, zscal, PETSC_MEMTYPE_HOST, &z, 1));
  PetscFunctionReturn(0);
}

/*
 * Operation z[j] = dot(x, y[j])
 *
 * We use an iterated application of dot() for each j. For small ranges of j this is still faster than an allocation of extra memory in order to use gemv().
 */
PetscErrorCode VecMDot_SeqViennaCL(Vec xin, PetscManagedInt nvscal, const Vec yin[], PetscManagedScalar zscal, PetscDeviceContext dctx) {
  PetscInt    *nvptr;
  PetscScalar *z;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx, nvscal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx, zscal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &z));
  if (const auto n = xin->map->n) {
    auto                                                    yyin = const_cast<Vec *>(yin);
    const ViennaCLVector                                   *xgpu, *ygpu;
    std::vector<viennacl::vector_base<PetscScalar> const *> ygpu_array(*nvptr);

    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    for (PetscInt i = 0; i < *nvptr; ++i) {
      PetscCall(VecViennaCLGetArrayRead(yyin[i], &ygpu));
      ygpu_array[i] = ygpu;
    }
    PetscCall(PetscLogGpuTimeBegin());
    viennacl::vector_tuple<PetscScalar> y_tuple(ygpu_array);
    ViennaCLVector                      result = viennacl::linalg::inner_prod(*xgpu, y_tuple);
    viennacl::copy(result.begin(), result.end(), z);
    for (PetscInt i = 0; i < *nvptr; ++i) PetscCall(VecViennaCLRestoreArrayRead(yyin[i], &ygpu));
    ViennaCLWaitForGPU();
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(PetscLogGpuFlops(PetscMax((*nvptr) * (2.0 * n - 1), 0.0)));
  } else {
    PetscCall(PetscArrayzero(z, *nvptr));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMTDot_SeqViennaCL(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  /* Since complex case is not supported at the moment, this is the same as VecMDot_SeqViennaCL */
  PetscCall(VecMDot_SeqViennaCL(xin, nv, yin, z, dctx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_SeqViennaCL(Vec xin, PetscManagedScalar ascal, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ViennaCLVector *xgpu;
    PetscScalar    *aptr;

    PetscCall(PetscManagedScalarGetValues(dctx, ascal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
    PetscCall(VecViennaCLGetArrayWrite(xin, &xgpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      *xgpu = viennacl::scalar_vector<PetscScalar>(xgpu->size(), *aptr);
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayWrite(xin, &xgpu));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqViennaCL(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  if (const auto n = xin->map->n) {
    if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
      PetscCall(VecSet_SeqViennaCL(xin, alpha, dctx));
    } else if (!PetscManagedScalarKnownAndEqual(alpha, 1.0)) {
      ViennaCLVector *xgpu;
      PetscScalar    *aptr;

      PetscCall(PetscManagedScalarGetValues(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
      PetscCall(VecViennaCLGetArray(xin, &xgpu));
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *xgpu *= *aptr;
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecViennaCLRestoreArray(xin, &xgpu));
      PetscCall(PetscLogGpuFlops(n));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqViennaCL(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  /* Since complex case is not supported at the moment, this is the same as VecDot_SeqViennaCL */
  PetscCall(VecDot_SeqViennaCL(xin, yin, z, dctx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqViennaCL(Vec xin, Vec yin, PetscDeviceContext) {
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
      PetscCall(VecViennaCLGetArrayWrite(yin, &ygpu));
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *ygpu = *xgpu;
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
      PetscCall(VecViennaCLRestoreArrayWrite(yin, &ygpu));

    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU*/
      PetscCall(VecCopy_SeqViennaCL_Private(xin, yin));
      ViennaCLWaitForGPU();
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        PetscCall(VecCopy_SeqViennaCL_Private(xin, yin));
        ViennaCLWaitForGPU();
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
        PetscCall(VecViennaCLGetArrayWrite(yin, &ygpu));
        PetscCall(PetscLogGpuTimeBegin());
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
        PetscCall(VecViennaCLRestoreArrayWrite(yin, &ygpu));
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
        PetscCall(VecViennaCLGetArrayWrite(yin, &ygpu));
        PetscCall(PetscLogGpuTimeBegin());
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
        PetscCall(VecViennaCLRestoreArrayWrite(yin, &ygpu));
      } else {
        PetscCall(VecCopy_SeqViennaCL_Private(xin, yin));
        ViennaCLWaitForGPU();
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqViennaCL(Vec xin, Vec yin, PetscDeviceContext) {
  ViennaCLVector *xgpu, *ygpu;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    PetscCall(VecViennaCLGetArray(xin, &xgpu));
    PetscCall(VecViennaCLGetArray(yin, &ygpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      viennacl::swap(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArray(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArray(yin, &ygpu));
  }
  PetscFunctionReturn(0);
}

// y = alpha * x + beta * y
PetscErrorCode VecAXPBY_SeqViennaCL(Vec yin, PetscManagedScalar alpha, PetscManagedScalar beta, Vec xin, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  if (const auto n = xin->map->n) {
    if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
      PetscCall(VecScale_SeqViennaCL(yin, beta, dctx));
    } else if (PetscManagedScalarKnownAndEqual(beta, 1.0)) {
      PetscCall(VecAXPY_SeqViennaCL(yin, alpha, xin, dctx));
    } else if (PetscManagedScalarKnownAndEqual(alpha, 1.0)) {
      PetscCall(VecAYPX_SeqViennaCL(yin, beta, xin, dctx));
    } else {
      PetscScalar          *aptr;
      const ViennaCLVector *xgpu;
      ViennaCLVector       *ygpu;

      PetscCall(PetscManagedScalarGetValues(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
      PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
      PetscCall(VecViennaCLGetArray(yin, &ygpu));
      if (PetscManagedScalarKnownAndEqual(beta, 0.0)) {
        PetscCall(PetscLogGpuTimeBegin());
        try {
          *ygpu = *xgpu * (*aptr);
          ViennaCLWaitForGPU();
        } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(PetscLogGpuFlops(n));
      } else {
        PetscScalar *bptr;

        PetscCall(PetscManagedScalarGetValues(dctx, beta, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &bptr));
        PetscCall(PetscLogGpuTimeBegin());
        try {
          *ygpu = *xgpu * (*aptr) + *ygpu * (*bptr);
          ViennaCLWaitForGPU();
        } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(PetscLogGpuFlops(3 * n));
      }
      PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
      PetscCall(VecViennaCLRestoreArray(yin, &ygpu));
    }
  }
  PetscFunctionReturn(0);
}

// this exists because it is too inefficient to do PetscManagedSubRange() in VecMAXPY_SeqViennaCL()
static PetscErrorCode VecAXPBYPCZ_SeqViennaCL_Private(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin) {
  PetscInt              n = zin->map->n;
  const ViennaCLVector *xgpu, *ygpu;
  ViennaCLVector       *zgpu;

  PetscFunctionBegin;
  PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
  PetscCall(VecViennaCLGetArrayRead(yin, &ygpu));
  PetscCall(VecViennaCLGetArray(zin, &zgpu));
  if (alpha == 0.0 && xin->map->n > 0) {
    PetscCall(PetscLogGpuTimeBegin());
    try {
      if (beta == 0.0) {
        *zgpu = gamma * *zgpu;
        ViennaCLWaitForGPU();
        PetscCall(PetscLogGpuFlops(1.0 * n));
      } else if (gamma == 0.0) {
        *zgpu = beta * *ygpu;
        ViennaCLWaitForGPU();
        PetscCall(PetscLogGpuFlops(1.0 * n));
      } else {
        *zgpu = beta * *ygpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        PetscCall(PetscLogGpuFlops(3.0 * n));
      }
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(3.0 * n));
  } else if (beta == 0.0 && xin->map->n > 0) {
    PetscCall(PetscLogGpuTimeBegin());
    try {
      if (gamma == 0.0) {
        *zgpu = alpha * *xgpu;
        ViennaCLWaitForGPU();
        PetscCall(PetscLogGpuFlops(1.0 * n));
      } else {
        *zgpu = alpha * *xgpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        PetscCall(PetscLogGpuFlops(3.0 * n));
      }
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
  } else if (gamma == 0.0 && xin->map->n > 0) {
    PetscCall(PetscLogGpuTimeBegin());
    try {
      *zgpu = alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(3.0 * n));
  } else if (xin->map->n > 0) {
    PetscCall(PetscLogGpuTimeBegin());
    try {
      /* Split operation into two steps. This is not completely ideal, but avoids temporaries (which are far worse) */
      if (gamma != 1.0) *zgpu *= gamma;
      *zgpu += alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArray(zin, &zgpu));
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArrayRead(yin, &ygpu));
    PetscCall(PetscLogGpuFlops(5.0 * n));
  }
  PetscFunctionReturn(0);
}

/* operation  z = alpha * x + beta *y + gamma *z*/
PetscErrorCode VecAXPBYPCZ_SeqViennaCL(Vec zin, PetscManagedScalar ascal, PetscManagedScalar bscal, PetscManagedScalar gscal, Vec xin, Vec yin, PetscDeviceContext dctx) {
  PetscScalar *aptr, *bptr, *gptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx, ascal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
  PetscCall(PetscManagedScalarGetValues(dctx, bscal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &bptr));
  PetscCall(PetscManagedScalarGetValues(dctx, gscal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &gptr));
  PetscCall(VecAXPBYPCZ_SeqViennaCL_Private(zin, *aptr, *bptr, *gptr, xin, yin));
  PetscFunctionReturn(0);
}

/*
 * Operation x = x + sum_i alpha_i * y_i for vectors x, y_i and scalars alpha_i
 *
 * ViennaCL supports a fast evaluation of x += alpha * y and x += alpha * y + beta * z,
 * hence there is an iterated application of these until the final result is obtained
 */
PetscErrorCode VecMAXPY_SeqViennaCL(Vec xin, PetscManagedInt nvscal, PetscManagedScalar ascal, Vec *y, PetscDeviceContext dctx) {
  PetscInt    *nvptr;
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx, nvscal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx, ascal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &aptr));
  for (PetscInt j = 0, nvval = *nvptr; j < nvval; ++j) {
    if (j + 1 < *nvptr) {
      PetscCall(VecAXPBYPCZ_SeqViennaCL_Private(xin, aptr[j], aptr[j + 1], 1.0, y[j], y[j + 1]));
      ++j;
    } else {
      PetscCall(VecAXPY_SeqViennaCL_Private(xin, aptr[j], y[j]));
    }
  }
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqViennaCL(Vec win, Vec xin, Vec yin, PetscDeviceContext) {
  PetscFunctionBegin;
  if (const auto n = win->map->n) {
    const ViennaCLVector *xgpu, *ygpu;
    ViennaCLVector       *wgpu;

    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLGetArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLGetArray(win, &wgpu));
    PetscCall(PetscLogGpuTimeBegin());
    try {
      *wgpu = viennacl::linalg::element_prod(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
    PetscCall(VecViennaCLRestoreArrayRead(yin, &ygpu));
    PetscCall(VecViennaCLRestoreArray(win, &wgpu));
    PetscCall(PetscLogGpuFlops(n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_SeqViennaCL(Vec xin, NormType type, PetscManagedReal zscal, PetscDeviceContext dctx) {
  PetscReal             z[2];
  PetscInt              n = xin->map->n;
  PetscBLASInt          bn;
  const ViennaCLVector *xgpu;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(VecViennaCLGetArrayRead(xin, &xgpu));
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *z = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops(PetscMax(2.0 * n - 1, 0.0)));
    } else if (type == NORM_INFINITY) {
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *z = viennacl::linalg::norm_inf(*xgpu);
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
    } else if (type == NORM_1) {
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *z = viennacl::linalg::norm_1(*xgpu);
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops(PetscMax(n - 1.0, 0.0)));
    } else if (type == NORM_1_AND_2) {
      PetscCall(PetscLogGpuTimeBegin());
      try {
        *z       = viennacl::linalg::norm_1(*xgpu);
        *(z + 1) = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops(PetscMax(2.0 * n - 1, 0.0)));
      PetscCall(PetscLogGpuFlops(PetscMax(n - 1.0, 0.0)));
    }
    PetscCall(VecViennaCLRestoreArrayRead(xin, &xgpu));
  } else if (type == NORM_1_AND_2) {
    *z       = 0.0;
    *(z + 1) = 0.0;
  } else *z = 0.0;
  PetscCall(PetscManagedRealSetValues(dctx, zscal, PETSC_MEMTYPE_HOST, z, 1 + (type == NORM_1_AND_2)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqViennaCL(Vec xin, PetscRandom r, PetscDeviceContext) {
  PetscFunctionBegin;
  PetscCall(VecSetRandom_SeqViennaCL_Private(xin, r));
  xin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqViennaCL(Vec vin, PetscDeviceContext) {
  PetscFunctionBegin;
  PetscCheckTypeNames(vin, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLCopyFromGPU(vin));
  PetscCall(VecResetArray_SeqViennaCL_Private(vin));
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqViennaCL(Vec vin, const PetscScalar *a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCheckTypeNames(vin, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLCopyFromGPU(vin));
  PetscCall(VecPlaceArray_Seq(vin, a, dctx));
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqViennaCL(Vec vin, const PetscScalar *a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCheckTypeNames(vin, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLCopyFromGPU(vin));
  PetscCall(VecReplaceArray_Seq(vin, a, dctx));
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCL - Creates a standard, sequential array-style vector.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
-  n - the vector length

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

.seealso: `VecCreateMPI()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`
@*/
PetscErrorCode VecCreateSeqViennaCL(MPI_Comm comm, PetscInt n, Vec *v) {
  PetscFunctionBegin;
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, n, n));
  PetscCall(VecSetType(*v, VECSEQVIENNACL));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCLWithArray - Creates a viennacl sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - viennacl array where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecViennaCLPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: `VecCreateMPIViennaCLWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `VecCreateGhost()`, `VecCreateSeq()`, `VecCUDAPlaceArray()`, `VecCreateSeqWithArray()`,
          `VecCreateMPIWithArray()`
@*/
PETSC_EXTERN PetscErrorCode VecCreateSeqViennaCLWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const ViennaCLVector *array, Vec *V) {
  PetscMPIInt        size;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(VecCreate(comm, V));
  PetscCall(VecSetSizes(*V, n, n));
  PetscCall(VecSetBlockSize(*V, bs));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQ on more than one process");
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(VecCreate_SeqViennaCL_Private(*V, array, dctx));
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCLWithArrays - Creates a ViennaCL sequential vector, where
   the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  cpuarray - CPU memory where the vector elements are to be stored.
-  viennaclvec - ViennaCL vector where the Vec entries are to be stored on the device.

   Output Parameter:
.  V - the vector

   Notes:
   If both cpuarray and viennaclvec are provided, the caller must ensure that
   the provided arrays have identical values.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: `VecCreateMPIViennaCLWithArrays()`, `VecCreate()`, `VecCreateSeqWithArray()`,
          `VecViennaCLPlaceArray()`, `VecPlaceArray()`, `VecCreateSeqCUDAWithArrays()`,
          `VecViennaCLAllocateCheckHost()`
@*/
PetscErrorCode VecCreateSeqViennaCLWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const ViennaCLVector *viennaclvec, Vec *V) {
  PetscMPIInt size;

  PetscFunctionBegin;

  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQ on more than one process");

  // set V's viennaclvec to be viennaclvec, do not allocate memory on host yet.
  PetscCall(VecCreateSeqViennaCLWithArray(comm, bs, n, viennaclvec, V));

  if (cpuarray && viennaclvec) {
    Vec_Seq *s        = (Vec_Seq *)((*V)->data);
    s->array          = (PetscScalar *)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_Seq *s        = (Vec_Seq *)((*V)->data);
    s->array          = (PetscScalar *)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_CPU;
  } else if (viennaclvec) {
    (*V)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*V)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}

/*@C
   VecViennaCLPlaceArray - Replace the viennacl vector in a Vec with
   the one provided by the user. This is useful to avoid a copy.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the ViennaCL vector

   Notes:
   You can return to the original viennacl vector with a call to
   VecViennaCLResetArray() It is not possible to use VecViennaCLPlaceArray()
   and VecPlaceArray() at the same time on the same vector.

   Level: intermediate

.seealso: `VecPlaceArray()`, `VecSetValues()`, `VecViennaCLResetArray()`,
          `VecCUDAPlaceArray()`,

@*/
PETSC_EXTERN PetscErrorCode VecViennaCLPlaceArray(Vec vin, const ViennaCLVector *a) {
  PetscFunctionBegin;
  PetscCheckTypeNames(vin, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLCopyToGPU(vin));
  PetscCheck(!((Vec_Seq *)vin->data)->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "VecViennaCLPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecViennaCLResetArray()/VecResetArray()");
  ((Vec_Seq *)vin->data)->unplacedarray  = (PetscScalar *)((Vec_ViennaCL *)vin->spptr)->GPUarray; /* save previous GPU array so reset can bring it back */
  ((Vec_ViennaCL *)vin->spptr)->GPUarray = (ViennaCLVector *)a;
  vin->offloadmask                       = PETSC_OFFLOAD_GPU;
  PetscCall(PetscObjectStateIncrease((PetscObject)vin));
  PetscFunctionReturn(0);
}

/*@C
   VecViennaCLResetArray - Resets a vector to use its default memory. Call this
   after the use of VecViennaCLPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: `VecViennaCLPlaceArray()`, `VecResetArray()`, `VecCUDAResetArray()`, `VecPlaceArray()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLResetArray(Vec vin) {
  PetscFunctionBegin;
  PetscCheckTypeNames(vin, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLCopyToGPU(vin));
  ((Vec_ViennaCL *)vin->spptr)->GPUarray = (ViennaCLVector *)((Vec_Seq *)vin->data)->unplacedarray;
  ((Vec_Seq *)vin->data)->unplacedarray  = 0;
  vin->offloadmask                       = PETSC_OFFLOAD_GPU;
  PetscCall(PetscObjectStateIncrease((PetscObject)vin));
  PetscFunctionReturn(0);
}

/*  VecDotNorm2 - computes the inner product of two vectors and the 2-norm squared of the second vector
 *
 *  Simply reuses VecDot() and VecNorm(). Performance improvement through custom kernel (kernel generator) possible.
 */
PetscErrorCode VecDotNorm2_SeqViennaCL(Vec s, Vec t, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  // complex is supposedly not supported (as per earlier comments) so the cast to
  // PetscManagedReal is valid, but lets double-check
  PetscAssert(!PetscDefined(USE_COMPLEX), PetscObjectComm((PetscObject)s), PETSC_ERR_SUP, "Complex is not supported for " PetscStringize(VECVIENNACL));
  PetscCall(VecDot_SeqViennaCL(s, t, dp, dctx));
  PetscCall(VecNorm_SeqViennaCL(t, NORM_2, (PetscManagedReal)nm, dctx));
  {
    PetscScalar *nmptr;

    PetscCall(PetscManagedScalarGetValues(dctx, nm, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &nmptr));
    *nmptr *= *nmptr; //squared norm required
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_SeqViennaCL(Vec win, Vec *V, PetscDeviceContext) {
  PetscFunctionBegin;
  PetscCall(VecCreateSeqViennaCL(PetscObjectComm((PetscObject)win), win->map->n, V));
  PetscCall(PetscLayoutReference(win->map, &(*V)->map));
  PetscCall(PetscObjectListDuplicate(((PetscObject)win)->olist, &((PetscObject)(*V))->olist));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)win)->qlist, &((PetscObject)(*V))->qlist));
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqViennaCL(Vec v, PetscDeviceContext) {
  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_ViennaCL *)v->spptr)->GPUarray_allocated;
      delete (Vec_ViennaCL *)v->spptr;
    }
  } catch (char *ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex); }
  PetscCall(VecDestroy_SeqViennaCL_Private(v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArray_SeqViennaCL(Vec v, PetscScalar **a, PetscDeviceContext) {
  PetscFunctionBegin;
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    PetscCall(VecViennaCLCopyFromGPU(v));
  } else {
    PetscCall(VecViennaCLAllocateCheckHost(v));
  }
  *a = *((PetscScalar **)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArray_SeqViennaCL(Vec v, PetscScalar **a, PetscDeviceContext) {
  PetscFunctionBegin;
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWrite_SeqViennaCL(Vec v, PetscScalar **a, PetscDeviceContext) {
  PetscFunctionBegin;
  PetscCall(VecViennaCLAllocateCheckHost(v));
  *a = *((PetscScalar **)v->data);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecBindToCPU_SeqAIJViennaCL(Vec V, PetscBool flg, PetscDeviceContext) {
  PetscFunctionBegin;
  V->boundtocpu = flg;
  if (flg) {
    PetscCall(VecViennaCLCopyFromGPU(V));
    V->offloadmask          = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dot             = VecDot_Seq;
    V->ops->norm            = VecNorm_Seq;
    V->ops->tdot            = VecTDot_Seq;
    V->ops->scale           = VecScale_Seq;
    V->ops->copy            = VecCopy_Seq;
    V->ops->set             = VecSet_Seq;
    V->ops->swap            = VecSwap_Seq;
    V->ops->axpy            = VecAXPY_Seq;
    V->ops->axpby           = VecAXPBY_Seq;
    V->ops->axpbypcz        = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult   = VecPointwiseMult_Seq;
    V->ops->pointwisedivide = VecPointwiseDivide_Seq;
    V->ops->setrandom       = VecSetRandom_Seq;
    V->ops->dot_local       = VecDot_Seq;
    V->ops->tdot_local      = VecTDot_Seq;
    V->ops->norm_local      = VecNorm_Seq;
    V->ops->mdot_local      = VecMDot_Seq;
    V->ops->mtdot_local     = VecMTDot_Seq;
    V->ops->maxpy           = VecMAXPY_Seq;
    V->ops->mdot            = VecMDot_Seq;
    V->ops->mtdot           = VecMTDot_Seq;
    V->ops->aypx            = VecAYPX_Seq;
    V->ops->waxpy           = VecWAXPY_Seq;
    V->ops->dotnorm2        = NULL;
    V->ops->placearray      = VecPlaceArray_Seq;
    V->ops->replacearray    = VecReplaceArray_Seq;
    V->ops->resetarray      = VecResetArray_Seq;
    V->ops->duplicate       = VecDuplicate_Seq;
  } else {
    V->ops->dot             = VecDot_SeqViennaCL;
    V->ops->norm            = VecNorm_SeqViennaCL;
    V->ops->tdot            = VecTDot_SeqViennaCL;
    V->ops->scale           = VecScale_SeqViennaCL;
    V->ops->copy            = VecCopy_SeqViennaCL;
    V->ops->set             = VecSet_SeqViennaCL;
    V->ops->swap            = VecSwap_SeqViennaCL;
    V->ops->axpy            = VecAXPY_SeqViennaCL;
    V->ops->axpby           = VecAXPBY_SeqViennaCL;
    V->ops->axpbypcz        = VecAXPBYPCZ_SeqViennaCL;
    V->ops->pointwisemult   = VecPointwiseMult_SeqViennaCL;
    V->ops->pointwisedivide = VecPointwiseDivide_SeqViennaCL;
    V->ops->setrandom       = VecSetRandom_SeqViennaCL;
    V->ops->dot_local       = VecDot_SeqViennaCL;
    V->ops->tdot_local      = VecTDot_SeqViennaCL;
    V->ops->norm_local      = VecNorm_SeqViennaCL;
    V->ops->mdot_local      = VecMDot_SeqViennaCL;
    V->ops->mtdot_local     = VecMTDot_SeqViennaCL;
    V->ops->maxpy           = VecMAXPY_SeqViennaCL;
    V->ops->mdot            = VecMDot_SeqViennaCL;
    V->ops->mtdot           = VecMTDot_SeqViennaCL;
    V->ops->aypx            = VecAYPX_SeqViennaCL;
    V->ops->waxpy           = VecWAXPY_SeqViennaCL;
    V->ops->dotnorm2        = VecDotNorm2_SeqViennaCL;
    V->ops->placearray      = VecPlaceArray_SeqViennaCL;
    V->ops->replacearray    = VecReplaceArray_SeqViennaCL;
    V->ops->resetarray      = VecResetArray_SeqViennaCL;
    V->ops->destroy         = VecDestroy_SeqViennaCL;
    V->ops->duplicate       = VecDuplicate_SeqViennaCL;
    V->ops->getarraywrite   = VecGetArrayWrite_SeqViennaCL;
    V->ops->getarray        = VecGetArray_SeqViennaCL;
    V->ops->restorearray    = VecRestoreArray_SeqViennaCL;
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec V, PetscDeviceContext dctx) {
  PetscMPIInt        size;
  PetscScalar        zero = 0;
  PetscManagedScalar zeroscal;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)V), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQVIENNACL on more than one process");
  PetscCall(VecCreate_Seq_Private(V, 0, dctx));
  PetscCall(PetscObjectChangeTypeName((PetscObject)V, VECSEQVIENNACL));

  PetscCall(VecBindToCPU_SeqAIJViennaCL(V, PETSC_FALSE, dctx));
  V->ops->bindtocpu = VecBindToCPU_SeqAIJViennaCL;

  PetscCall(VecViennaCLAllocateCheck(V));
  PetscCall(PetscManageHostScalar(dctx, &zero, 1, &zeroscal));
  PetscCall(VecSet_SeqViennaCL(V, zeroscal, dctx));
  PetscCall(PetscManagedScalarDestroy(dctx, &zeroscal));
  PetscFunctionReturn(0);
}

/*@C
  VecViennaCLGetCLContext - Get the OpenCL context in which the Vec resides.

  Caller should cast (*ctx) to (const cl_context). Caller is responsible for
  invoking clReleaseContext().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  ctx - pointer to the underlying CL context

  Level: intermediate

.seealso: `VecViennaCLGetCLQueue()`, `VecViennaCLGetCLMemRead()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLContext(Vec v, PETSC_UINTPTR_T *ctx) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to get the associated cl_context.");
#else

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  const ViennaCLVector *v_vcl;
  PetscCall(VecViennaCLGetArrayRead(v, &v_vcl));
  try {
    viennacl::ocl::context vcl_ctx = v_vcl->handle().opencl_handle().context();
    const cl_context ocl_ctx = vcl_ctx.handle().get();
    clRetainContext(ocl_ctx);
    *ctx = (PETSC_UINTPTR_T)(ocl_ctx);
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLQueue - Get the OpenCL command queue to which all
  operations of the Vec are enqueued.

  Caller should cast (*queue) to (const cl_command_queue). Caller is
  responsible for invoking clReleaseCommandQueue().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  ctx - pointer to the CL command queue

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLGetCLMemRead()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLQueue(Vec v, PETSC_UINTPTR_T *queue) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to get the associated cl_command_queue.");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  const ViennaCLVector *v_vcl;
  PetscCall(VecViennaCLGetArrayRead(v, &v_vcl));
  try {
    viennacl::ocl::context vcl_ctx = v_vcl->handle().opencl_handle().context();
    const viennacl::ocl::command_queue &vcl_queue = vcl_ctx.current_queue();
    const cl_command_queue ocl_queue = vcl_queue.handle().get();
    clRetainCommandQueue(ocl_queue);
    *queue = (PETSC_UINTPTR_T)(ocl_queue);
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMemRead - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLGetCLMemWrite()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemRead(Vec v, PETSC_UINTPTR_T *mem) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  const ViennaCLVector *v_vcl;
  PetscCall(VecViennaCLGetArrayRead(v, &v_vcl));
  try {
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }
  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMemWrite - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  The device pointer has to be released by calling
  VecViennaCLRestoreCLMemWrite().  Upon restoring the vector data the data on
  the host will be marked as out of date.  A subsequent access of the host data
  will thus incur a data transfer from the device to the host.

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLRestoreCLMemWrite()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemWrite(Vec v, PETSC_UINTPTR_T *mem) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  ViennaCLVector *v_vcl;
  PetscCall(VecViennaCLGetArrayWrite(v, &v_vcl));
  try {
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLRestoreCLMemWrite - Restores a CL buffer pointer previously
  acquired with VecViennaCLGetCLMemWrite().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

  Input Parameters:
.  v    - the vector

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLGetCLMemWrite()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMemWrite(Vec v) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to restore a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLRestoreArrayWrite(v, PETSC_NULL));

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMem - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  The device pointer has to be released by calling VecViennaCLRestoreCLMem().
  Upon restoring the vector data the data on the host will be marked as out of
  date.  A subsequent access of the host data will thus incur a data transfer
  from the device to the host.

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLRestoreCLMem()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMem(Vec v, PETSC_UINTPTR_T *mem) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  ViennaCLVector *v_vcl;
  PetscCall(VecViennaCLGetArray(v, &v_vcl));
  try {
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const &ex) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ViennaCL error: %s", ex.what()); }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLRestoreCLMem - Restores a CL buffer pointer previously
  acquired with VecViennaCLGetCLMem().

   This marks the host data as out of date. Subsequent access to the vector
   data on the host side with for instance VecGetArray() incurs a data
   transfer.

  Input Parameters:
.  v    - the vector

  Level: intermediate

.seealso: `VecViennaCLGetCLContext()`, `VecViennaCLGetCLMem()`
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMem(Vec v) {
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "PETSc must be configured with --with-opencl to restore a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);
  PetscCall(VecViennaCLRestoreArray(v, PETSC_NULL));

  PetscFunctionReturn(0);
#endif
}

PetscErrorCode VecCreate_SeqViennaCL_Private(Vec V, const ViennaCLVector *array, PetscDeviceContext dctx) {
  Vec_ViennaCL *vecviennacl;
  PetscMPIInt   size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)V), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot create VECSEQVIENNACL on more than one process");
  PetscCall(VecCreate_Seq_Private(V, 0, dctx));
  PetscCall(PetscObjectChangeTypeName((PetscObject)V, VECSEQVIENNACL));
  PetscCall(VecBindToCPU_SeqAIJViennaCL(V, PETSC_FALSE, dctx));
  V->ops->bindtocpu = VecBindToCPU_SeqAIJViennaCL;

  if (array) {
    if (!V->spptr) V->spptr = new Vec_ViennaCL;
    vecviennacl                     = (Vec_ViennaCL *)V->spptr;
    vecviennacl->GPUarray_allocated = 0;
    vecviennacl->GPUarray           = (ViennaCLVector *)array;
    V->offloadmask                  = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}
