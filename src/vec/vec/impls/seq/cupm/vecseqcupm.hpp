#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/veccupmimpl.h>
#include <petsc/private/randomimpl.h> // for _p_PetscRandom
#include "../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp"
#include "../src/sys/objects/device/impls/cupm/cupmallocator.hpp"
#include "../src/sys/objects/device/impls/cupm/kernels.hpp"

#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// TODO
// - refactor the AXPY's for code reuse
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - touch up the docs for both implementations
// - pick one of the VecGetArray<modifier>() to explain data movement semantics in the docs and
//   have everyone else refer to it
// - do rocblas instead of hipblas

namespace Petsc {

// REVIEW ME: using "Vec" as the namespace causes ambiguity errors when referring to "Vec" the
// type later on
namespace vec {

namespace cupm {

namespace impl {

namespace {

template <bool>
struct UseComplexTag { };

} // anonymous namespace

template <device::cupm::DeviceType T>
struct VecSeq_CUPM : Vec_CUPMBase<T, VecSeq_CUPM<T>> {
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type, T, VecSeq_CUPM<T>);

private:
  PETSC_CXX_COMPAT_DECL(auto VecIMPLCast_(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_Seq *>(v->data));

  PETSC_CXX_COMPAT_DECL(constexpr auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECSEQCUPM());

  // common core for min and max
  template <typename TupleFuncT, typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(TupleFuncT &&, UnaryFuncT &&, PetscReal, Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext));
  // common core for pointwise binary and pointwise unary thrust functions
  template <typename BinaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisebinary_async_(BinaryFuncT &&, PetscDeviceContext, Vec, Vec, Vec));
  template <typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwiseunary_async_(UnaryFuncT &&, PetscDeviceContext, Vec, Vec /*out*/ = nullptr));
  // mdot dispatchers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<true>, Vec, PetscInt, const Vec[], PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<false>, Vec, PetscInt, const Vec[], PetscManagedScalar, PetscDeviceContext));
  // dispatcher for the actual kernels for mdot when NOT configured for complex, called by
  // mdot_async_(use_complex_tag<false>,...)
  template <std::size_t... Idx>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext, cupmStream_t, const PetscScalar *, const Vec[], PetscInt, PetscScalar *, util::index_sequence<Idx...>));
  template <int>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext, cupmStream_t, const PetscScalar *, const Vec[], PetscInt, PetscScalar *, PetscInt &));
  template <std::size_t... Idx>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_kernel_dispatch_(PetscDeviceContext, cupmStream_t, PetscScalar *, const PetscScalar *, const Vec *, PetscInt, util::index_sequence<Idx...>));
  template <int>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_kernel_dispatch_(PetscDeviceContext, cupmStream_t, PetscScalar *, const PetscScalar *, const Vec *, PetscInt, PetscInt &));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async_(Vec, PetscDeviceContext, PetscScalar * /*host_ptr*/ = nullptr, PetscScalar * /*device_ptr*/ = nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm, PetscInt, PetscInt, PetscDeviceContext, Vec *, PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupmwithbotharrays_async(MPI_Comm, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], PetscDeviceContext, Vec *));

  // callable indirectly via function pointers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec, Vec *, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec, PetscManagedScalar, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec, PetscManagedScalar, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec, Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec, Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec, PetscManagedScalar, Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec, PetscManagedInt, PetscManagedScalar, Vec *, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec, Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_async(Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode scale_async(Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec, Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_async(Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode swap_async(Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpby_async(Vec, PetscManagedScalar, PetscManagedScalar, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpbypcz_async(Vec, PetscManagedScalar, PetscManagedScalar, PetscManagedScalar, Vec, Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec, NormType, PetscManagedReal, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec, Vec, PetscManagedScalar, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode conjugate_async(Vec, PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec, Vec, PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec, Vec, PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec, PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec, const PetscScalar *, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec, PetscManagedScalar, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec, PetscRandom, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec, PetscBool, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setpreallocationcoo_async(Vec, PetscCount, const PetscInt[], PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setvaluescoo_async(Vec, const PetscScalar[], InsertMode, PetscDeviceContext));
};

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async_(Vec v, PetscDeviceContext dctx, PetscScalar *host_array, PetscScalar *device_array)) {
  NVTX_RANGE;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)), &size));
  PetscCheck(size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must create VecSeq on communicator of size 1, have size %d", size);
  PetscCall(VecCreate_Seq_Private(v, host_array, dctx));
  PetscCall(Initialize_CUPMBase(v, PETSC_FALSE, host_array, device_array, dctx));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                             constructors/destructors                               //

// v->ops->create
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(createseqcupm_async_(v, dctx));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPM()
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscDeviceContext dctx, Vec *v, PetscBool call_set_type)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm, bs, n, n, dctx, v, call_set_type));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPMWithArrays()
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupmwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar host_array[], const PetscScalar device_array[], PetscDeviceContext dctx, Vec *v)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // createseqcupm_async_() is called!
  PetscCall(createseqcupm_async(comm, bs, n, dctx, v, PETSC_FALSE));
  PetscCall(createseqcupm_async_(*v, dctx, PetscRemoveConstCast(host_array), PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(Duplicate_CUPMBase(v, y, dctx));
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(Destroy_CUPMBase(v, dctx, VecDestroy_Seq));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  utility methods                                   //
//                                                                                    //
// ================================================================================== //

// v->ops->bindtocpu
template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(BindToCPU_CUPMBase(v, usehost, dctx));

  // REVIEW ME: this absolutely should be some sort of bulk mempcy rather than this mess
  VecSetOp_CUPM(dot, VecDot_Seq, dot_async);
  VecSetOp_CUPM(norm, VecNorm_Seq, norm_async);
  VecSetOp_CUPM(tdot, VecTDot_Seq, tdot_async);
  VecSetOp_CUPM(mdot, VecMDot_Seq, mdot_async);
  VecSetOp_CUPM(resetarray, VecResetArray_Seq, resetarray_async<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(placearray, VecPlaceArray_Seq, placearray_async<PETSC_MEMTYPE_HOST>);
  v->ops->mtdot = v->ops->mtdot_local = VecMTDot_Seq;
  VecSetOp_CUPM(conjugate, VecConjugate_Seq, conjugate_async);
  VecSetOp_CUPM(max, VecMax_Seq, max_async);
  VecSetOp_CUPM(min, VecMin_Seq, min_async);
  VecSetOp_CUPM(setpreallocationcoo, VecSetPreallocationCOO_Seq, setpreallocationcoo_async);
  VecSetOp_CUPM(setvaluescoo, VecSetValuesCOO_Seq, setvaluescoo_async);
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT &&binary, PetscDeviceContext dctx, Vec xin, Vec yin, Vec zout)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(device::cupm::impl::ThrustApplyPointwise<T>(dctx, std::forward<BinaryFuncT>(binary), zout->map->n, DeviceArrayRead(dctx, xin).data(), DeviceArrayRead(dctx, yin).data(), DeviceArrayWrite(dctx, zout).data()));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT &&unary, PetscDeviceContext dctx, Vec xinout, Vec yin)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (!yin || (xinout == yin)) {
    // in-place
    PetscCall(device::cupm::impl::ThrustApplyPointwise<T>(dctx, std::forward<UnaryFuncT>(unary), xinout->map->n, DeviceArrayReadWrite(dctx, xinout).data()));
  } else {
    PetscCall(device::cupm::impl::ThrustApplyPointwise<T>(dctx, std::forward<UnaryFuncT>(unary), xinout->map->n, DeviceArrayRead(dctx, xinout).data(), DeviceArrayWrite(dctx, yin).data()));
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->resetarray or VecCUPMResetArray()
template <device::cupm::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(base_type::template ResetArray_CUPMBase<mtype>(v, VecResetArray_Seq, dctx));
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <device::cupm::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(base_type::template PlaceArray_CUPMBase<mtype>(v, a, VecPlaceArray_Seq, dctx));
  PetscFunctionReturn(0);
}

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w), VECSEQCUPM(), &wisseqcupm));
  if (wisseqcupm) {
    if (const auto wseq = VecIMPLCast(w)) {
      if (auto &alloced = wseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w);

        PetscCall(PetscFree(alloced));
        w->pinned_memory = PETSC_FALSE;
      }
      wseq->array = wseq->unplacedarray = nullptr;
    }
    if (const auto wcu = VecCUPMCast(w)) {
      if (auto &device_array = wcu->array_d) {
        cupmStream_t stream;

        PetscCall(GetHandles_(dctx, &stream));
        PetscCallCUPM(cupmFreeAsync(device_array, stream));
      }
      PetscCall(PetscFree(w->spptr /* wcu */));
    }
  }
  if (v->petscnative && wisseqcupm) {
    PetscCall(PetscFree(w->data));
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(w)));
  } else {
    const auto arrayptr = &VecIMPLCast(w)->array;
    if (access == PETSC_MEMORY_ACCESS_READ) {
      PetscCall(VecGetArrayReadAsync(v, const_cast<const PetscScalar **>(arrayptr), dctx));
    } else {
      PetscCall(VecGetArrayAsync(v, arrayptr, dctx));
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) PetscCall(DeviceAllocateCheck_(dctx, w));
  }
  PetscFunctionReturn(0);
}

// v->ops->restorelocalvector or v->ops->restorelocalvectorread
template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w), VECSEQCUPM(), &wisseqcupm));
  if (v->petscnative && wisseqcupm) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr; // these assignments are __critical__, as w may persist
    w->spptr         = nullptr; // after this call returns and shouldn't share data with v!
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  } else {
    auto array = &VecIMPLCast(w)->array;
    if (access == PETSC_MEMORY_ACCESS_READ) {
      PetscCall(VecRestoreArrayReadAsync(v, const_cast<const PetscScalar **>(array), dctx));
    } else {
      PetscCall(VecRestoreArrayAsync(v, array, dctx));
    }
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;

      PetscCall(GetHandles_(dctx, &stream));
      PetscCallCUPM(cupmFreeAsync(VecCUPMCast(w)->array_d, stream));
      PetscCall(PetscFree(w->spptr));
    }
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

namespace detail {

// thrust::complex constructors are not constexpr
// (https://github.com/NVIDIA/thrust/issues/1677), and so you cannot constant-initialize them
// otherwise nvcc throws a
// error: dynamic initialization is not supported for a __constant__ variable
// so make a dummy "complex" type with a constexpr constructor
struct complex_one {
  const PetscReal r;
  const PetscReal i;

  constexpr complex_one() noexcept : r(1.0), i(0.0) { }
};
// REVIEW ME:
// 1. find a way to not have this be a global variable. problem is that you cant attach device
//    memory spaces to class member variables (static or otherwise) or inside host functions...
// 2. allocating and de-allocating this one demand seems dumb too...
static const PETSC_CONSTMEM_DECL PETSC_DEVICE_DECL complex_one GlobalDeviceOne{};

} // namespace detail

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto n = static_cast<cupmBlasInt_t>(yin->map->n);

  PetscFunctionBegin;
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(PetscCUPMMemcpyAsync(DeviceArrayWrite(dctx, yin).data(), DeviceArrayRead(dctx, xin).data(), n, cupmMemcpyDeviceToDevice, stream));
    PetscCall(PetscLogGpuTimeEnd());
  } else {
    const auto       alphaIsOne = PetscManagedScalarKnownAndEqual(alpha, 1.0);
    cupmBlasHandle_t cupmBlasHandle;
    PetscScalar     *aptr;

    PetscCall(PetscManagedScalarGetArray(dctx, alpha, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    {
      const auto cmode  = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};
      const auto calpha = cupmScalarCast(aptr);
      const auto yptr   = DeviceArrayReadWrite(dctx, yin);
      const auto xptr   = DeviceArrayRead(dctx, xin);

      PetscCall(PetscLogGpuTimeBegin());
      if (alphaIsOne) {
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, calpha, xptr.cupmdata(), 1, yptr.cupmdata(), 1));
      } else {
        static cupmScalar_t *d_one = nullptr;

        if (!d_one) { PetscCallCUPM(cupmGetSymbolAddress(reinterpret_cast<void **>(&d_one), detail::GlobalDeviceOne)); }
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, calpha, yptr.cupmdata(), 1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, d_one, xptr.cupmdata(), 1, yptr.cupmdata(), 1));
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops((alphaIsOne ? 1 : 2) * n));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscBool xiscupm;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(xin), &xiscupm, VECSEQCUPM(), VECMPICUPM(), ""));
  if (xiscupm) {
    const auto       n = static_cast<cupmBlasInt_t>(yin->map->n);
    PetscScalar     *aptr;
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(PetscManagedScalarGetArray(dctx, alpha, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, cupmScalarCast(aptr), DeviceArrayRead(dctx, xin), 1, DeviceArrayReadWrite(dctx, yin), 1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(2 * n));
  } else {
    PetscCall(VecAXPY_Seq(yin, alpha, xin, dctx));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseDivide_Seq(win, xin, yin, dctx));
  } else {
    // note order of arguments! xin and yin are read, win is written!
    PetscCall(pointwisebinary_async_(thrust::divides<PetscScalar>{}, dctx, xin, yin, win));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseMult_Seq(win, xin, yin, dctx));
  } else {
    // note order of arguments! xin and yin are read, win is written!
    PetscCall(pointwisebinary_async_(thrust::multiplies<PetscScalar>{}, dctx, xin, yin, win));
  }
  PetscFunctionReturn(0);
}

namespace detail {

struct reciprocal {
  PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(PetscScalar s) const {
    // yes all of this verbosity is needed because sometimes PetscScalar is a thrust::complex
    // and then it matter whether we do s ? true : false vs s == 0, as well as whether we wrap
    // everything in PetscScalar...
    return s == PetscScalar{0.0} ? s : PetscScalar{1.0} / s;
  }
};

} // namespace detail

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(pointwiseunary_async_(detail::reciprocal{}, dctx, xin));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscManagedScalar alpha, Vec xin, Vec yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    PetscCall(copy_async(yin, win, dctx));
  } else {
    const auto       n    = win->map->n;
    const auto       wptr = DeviceArrayWrite(dctx, win);
    cupmBlasHandle_t cupmBlasHandle;
    cupmStream_t     stream;
    PetscScalar     *aptr;

    PetscCall(PetscManagedScalarGetArray(dctx, alpha, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
    PetscCall(GetHandles_(dctx, &cupmBlasHandle, &stream));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCall(PetscCUPMMemcpyAsync(wptr.data(), DeviceArrayRead(dctx, yin).data(), n, cupmMemcpyDeviceToDevice, stream, true));
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, cupmScalarCast(aptr), DeviceArrayRead(dctx, xin), 1, wptr.cupmdata(), 1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(2 * n));
  }
  PetscFunctionReturn(0);
}

namespace kernels {

template <typename... Args>
PETSC_KERNEL_DECL static void maxpy_kernel(const PetscInt size, PetscScalar *PETSC_RESTRICT xptr, const PetscScalar *PETSC_RESTRICT aptr, Args... yptr) {
  constexpr int                    N = sizeof...(Args);
  PETSC_SHAREDMEM_DECL PetscScalar aptr_shmem[N];
  const auto                       tx       = threadIdx.x;
  const PetscScalar               *yptr_p[] = {yptr...};

  // load a to shared memory
  if (tx < N) aptr_shmem[tx] = aptr[tx];
  __syncthreads();

  ::Petsc::device::cupm::kernels::util::grid_stride_1D(size, [&](PetscInt i) {
  // these may look the same but give different results!
#if 0
    PetscScalar sum = 0.0;

#pragma unroll
    for (auto j = 0; j < N; ++j) sum += aptr_shmem[j]*yptr_p[j][i];
    xptr[i] += sum;
#else
    auto sum = xptr[i];

#pragma unroll
    for (auto j = 0; j < N; ++j) sum += aptr_shmem[j]*yptr_p[j][i];
    xptr[i] = sum;
#endif
  });
  return;
}

} // namespace kernels

namespace detail {

// a helper-struct to gobble the size_t input, it is used with template parameter pack
// expansion such that
// typename repeat_type<MyType, IdxParamPack>...
// expands to
// MyType, MyType, MyType, ... [repeated sizeof...(IdxParamPack) times]
template <typename T, std::size_t>
struct repeat_type {
  using type = T;
};

template <typename T, std::size_t i>
using repeat_type_t = typename repeat_type<T, i>::type;

} // namespace detail

template <device::cupm::DeviceType T>
template <std::size_t... Idx>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, PetscScalar *xptr, const PetscScalar *aptr, const Vec *yin, PetscInt size, util::index_sequence<Idx...>)) {
  PetscFunctionBegin;
  PetscCall(PetscCUPMLaunchKernel1D(size, 0, stream, kernels::maxpy_kernel<detail::repeat_type_t<const PetscScalar *, Idx>...>, size, xptr, aptr, DeviceArrayRead(dctx, yin[Idx]).data()...));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
template <int N>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, PetscScalar *xptr, const PetscScalar *aptr, const Vec *yin, PetscInt size, PetscInt &yidx)) {
  PetscFunctionBegin;
  PetscCall(maxpy_kernel_dispatch_(dctx, stream, xptr, aptr + yidx, yin + yidx, size, util::make_index_sequence<N>{}));
  yidx += N;
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscManagedInt nv, PetscManagedScalar alpha, Vec *yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto   n    = xin->map->n;
  const auto   xptr = DeviceArrayReadWrite(dctx, xin);
  PetscInt     yidx = 0;
  PetscInt     nvval;
  PetscInt    *nvptr;
  PetscScalar *aptr;
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &stream));
  PetscCall(PetscManagedScalarGetArray(dctx, alpha, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
  // possible (but highly unlikely) implicit sync
  PetscCall(PetscManagedIntGetArray(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  nvval = *nvptr;
  PetscCall(PetscLogGpuTimeBegin());
  do {
    switch (nvval - yidx) {
    case 7: PetscCall(maxpy_kernel_dispatch_<7>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 6: PetscCall(maxpy_kernel_dispatch_<6>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 5: PetscCall(maxpy_kernel_dispatch_<5>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 4: PetscCall(maxpy_kernel_dispatch_<4>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 3: PetscCall(maxpy_kernel_dispatch_<3>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 2: PetscCall(maxpy_kernel_dispatch_<2>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    case 1: PetscCall(maxpy_kernel_dispatch_<1>(dctx, stream, xptr.data(), aptr, yin, n, yidx)); break;
    default: // 8 or more
      PetscCall(maxpy_kernel_dispatch_<8>(dctx, stream, xptr.data(), aptr, yin, n, yidx));
      break;
    }
  } while (yidx < nvval);
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(nvval * 2 * n));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto       n = xin->map->n;
  PetscScalar     *zptr;
  cupmBlasHandle_t cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &cupmBlasHandle));
  PetscCall(PetscManagedScalarGetArray(dctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr));
  {
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};

    // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
    // second
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXdot(cupmBlasHandle, n, DeviceArrayRead(dctx, yin), 1, DeviceArrayRead(dctx, xin), 1, cupmScalarCast(zptr)));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(PetscMax(2 * n - 1, 0)));
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels {

PETSC_HOSTDEVICE_INLINE_DECL static PetscInt EntriesPerGroup(const PetscInt size) {
  const auto group_entries = (size - 1) / gridDim.x + 1;
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <typename... ConstPetscScalarPointer>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscInt size, PetscScalar *PETSC_RESTRICT results, ConstPetscScalarPointer... y) {
  constexpr int                    N = sizeof...(ConstPetscScalarPointer);
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N * MDOT_WORKGROUP_SIZE];
  const PetscScalar               *ylocal[] = {y...};
  PetscScalar                      sumlocal[N];
  // HIP -- for whatever reason -- has threadIdx, blockIdx, blockDim, and gridDim as separate
  // types, so each of these go on separate lines...
  const auto                       tx       = threadIdx.x;
  const auto                       bx       = blockIdx.x;
  const auto                       bdx      = blockDim.x;
  const auto                       gdx      = gridDim.x;
  const auto                       worksize = EntriesPerGroup(size);
  const auto                       begin    = tx + bx * worksize;
  const auto                       end      = min((bx + 1) * worksize, size);

#pragma unroll
  for (auto i = 0; i < N; ++i) sumlocal[i] = 0;

#pragma unroll
  for (auto i = begin; i < end; i += bdx) {
    const auto xi = x[i]; // load only once from global memory!

#pragma unroll
    for (auto j = 0; j < N; ++j) sumlocal[j] += ylocal[j][i] * xi;
  }

#pragma unroll
  for (auto i = 0; i < N; ++i) shmem[tx + i * MDOT_WORKGROUP_SIZE] = sumlocal[i];

    // parallel reduction
#pragma unroll
  for (auto stride = bdx / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
#pragma unroll
      for (auto i = 0; i < N; ++i) shmem[tx + i * MDOT_WORKGROUP_SIZE] += shmem[tx + stride + i * MDOT_WORKGROUP_SIZE];
    }
  }
  __syncthreads();
  // bottom N threads per block write to global memory
  // REVIEW ME: I am ~pretty~ sure we don't need another __syncthreads() here since each thread
  // writes to the same sections in the above loop that it is about to read from below
  if (tx < N) results[bx + tx * gdx] = shmem[tx * MDOT_WORKGROUP_SIZE];
  return;
}

PETSC_KERNEL_DECL static void sum_kernel(const PetscInt size, PetscScalar *PETSC_RESTRICT z, const PetscScalar *PETSC_RESTRICT results) {
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(size, [=](PetscInt i) {
    PetscScalar z_sum = 0;

    for (auto j = i * MDOT_WORKGROUP_NUM; j < (i + 1) * MDOT_WORKGROUP_NUM; ++j) z_sum += results[j];
    z[i] = z_sum;
  });
  return;
}

} // namespace kernels

template <device::cupm::DeviceType T>
template <std::size_t... Idx>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar *results, util::index_sequence<Idx...>)) {
  PetscFunctionBegin;
  // REVIEW ME: convert this kernel launch to PetscCUPMLaunchKernel1D(), it currently launches
  // 128 blocks of 128 threads every time which may be wasteful
  PetscCallCUPM(cupmLaunchKernel(kernels::mdot_kernel<detail::repeat_type_t<const PetscScalar *, Idx>...>, MDOT_WORKGROUP_NUM, MDOT_WORKGROUP_SIZE, 0, stream, xarr, size, results, DeviceArrayRead(dctx, yin[Idx]).data()...));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
template <int N>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar *results, PetscInt &yidx)) {
  PetscFunctionBegin;
  PetscCall(mdot_kernel_dispatch_(dctx, stream, xarr, yin + yidx, size, results + yidx * MDOT_WORKGROUP_NUM, util::make_index_sequence<N>{}));
  yidx += N;
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx)) {
  // the largest possible size of a batch
  constexpr auto batchsize       = 8;
  // how many sub streams to create, if nv <= batchsize we can do this without looping, so we
  // do not create substreams. Note we don't create more than 8 streams, in practice we could
  // not get more parallelism with higher numbers.
  const auto     num_sub_streams = nv > batchsize ? std::min((nv + batchsize) / batchsize, batchsize) : 0;
  const auto     n               = xin->map->n;
  // number of vectors that we handle via the batches. note any singletons are handled by
  // cublas, hence the nv-1.
  const auto     nvbatch         = ((nv % batchsize) == 1) ? nv - 1 : nv;
  const auto     nwork           = nvbatch * MDOT_WORKGROUP_NUM;
  PetscScalar   *d_results, *zptr = nullptr;
  cupmStream_t   stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &stream));
  // allocate scratchpad memory for the results of individual work groups
  PetscCall(PetscCUPMMallocAsync(&d_results, nwork, stream));
  {
    const auto          xptr       = DeviceArrayRead(dctx, xin);
    PetscInt            yidx       = 0;
    auto                subidx     = 0;
    auto                cur_stream = stream;
    auto                cur_ctx    = dctx;
    PetscDeviceContext *sub        = nullptr;
    PetscStreamType     stype;

    // REVIEW ME: maybe PetscDeviceContextFork() should insert dctx into the first entry of
    // sub. Ideally the parent context should also join in on the fork, but it is extremely
    // fiddly to do so presently
    PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
    if (stype == PETSC_STREAM_GLOBAL_BLOCKING) stype = PETSC_STREAM_DEFAULT_BLOCKING;
    // If we have a globally blocking stream create nonblocking streams instead (as we can
    // locally exploit the parallelism). Otherwise use the prescribed stream type.
    PetscCall(PetscDeviceContextForkWithStreamType(dctx, stype, num_sub_streams, &sub));
    PetscCall(PetscLogGpuTimeBegin());
    do {
      if (num_sub_streams) {
        cur_ctx = sub[subidx++ % num_sub_streams];
        PetscCall(GetHandles_(cur_ctx, &cur_stream));
      }
      // REVIEW ME: Should probably try and load-balance these. Consider the case where nv = 9;
      // it is very likely better to do 4+5 rather than 8+1
      switch (nv - yidx) {
      case 7: PetscCall(mdot_kernel_dispatch_<7>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 6: PetscCall(mdot_kernel_dispatch_<6>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 5: PetscCall(mdot_kernel_dispatch_<5>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 4: PetscCall(mdot_kernel_dispatch_<4>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 3: PetscCall(mdot_kernel_dispatch_<3>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 2: PetscCall(mdot_kernel_dispatch_<2>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx)); break;
      case 1: {
        cupmBlasHandle_t cupmBlasHandle;

        PetscCall(GetHandles_(cur_ctx, &cupmBlasHandle));
        PetscCall(PetscManagedScalarGetArray(cur_ctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr));
        {
          const auto with = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};
          PetscCallCUPMBLAS(cupmBlasXdot(cupmBlasHandle, static_cast<cupmBlasInt_t>(n), DeviceArrayRead(cur_ctx, yin[yidx]), 1, xptr.cupmdata(), 1, cupmScalarCast(zptr + yidx)));
        }
        ++yidx;
      } break;
      default: // 8 or more
        PetscCall(mdot_kernel_dispatch_<8>(cur_ctx, cur_stream, xptr.data(), yin, n, d_results, yidx));
        break;
      }
    } while (yidx < nv);
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscDeviceContextJoin(dctx, num_sub_streams, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  }

  if (!zptr) { PetscCall(PetscManagedScalarGetArray(dctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr)); }
  PetscCall(PetscCUPMLaunchKernel1D(nvbatch, 0, stream, kernels::sum_kernel, nvbatch, zptr, d_results));
  // do these now while final reduction is in flight
  PetscCall(PetscLogFlops(nwork));
  // free these here, we will already have synchronized
  PetscCallCUPM(cupmFreeAsync(d_results, stream));
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx)) {
  // probably not worth it to run more than 8 of these at a time?
  const auto          n_sub = PetscMin(nv, 8);
  const auto          n     = static_cast<cupmBlasInt_t>(xin->map->n);
  const auto          xptr  = DeviceArrayRead(dctx, xin);
  PetscScalar        *zptr;
  PetscDeviceContext *subctx;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(PetscManagedScalarGetArray(dctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr));
  PetscCall(PetscDeviceContextFork(dctx, n_sub, &subctx));
  for (PetscInt i = 0; i < nv; ++i) {
    const auto       sub = subctx[i % n_sub];
    cupmBlasHandle_t handle;

    PetscCall(GetHandles_(sub, &handle));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{handle, CUPMBLAS_POINTER_MODE_DEVICE};
      PetscCallCUPMBLAS(cupmBlasXdot(handle, n, DeviceArrayRead(sub, yin[i]), 1, xptr.cupmdata(), 1, cupmScalarCast(zptr + i)));
    }
  }
  PetscCall(PetscDeviceContextJoin(dctx, n_sub, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &subctx));
  PetscCall(PetscLogGpuTimeEnd());
  // REVIEW ME: flops?????
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  NVTX_RANGE;
  if (PetscUnlikely(PetscManagedIntKnownAndEqual(nv, 1))) {
    PetscCall(dot_async(xin, PetscRemoveConstCast(yin[0]), z, dctx));
  } else if (const auto n = PetscLikely(xin->map->n)) {
    PetscInt *nvptr;

    // implicity sync
    PetscCall(PetscManagedIntGetArray(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
    PetscCheck(*nvptr > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Number of vectors provided to %s %" PetscInt_FMT " not positive", PETSC_FUNCTION_NAME, *nvptr);

    PetscCall(mdot_async_(UseComplexTag<PetscDefined(USE_COMPLEX)>{}, xin, *nvptr, yin, z, dctx));
    // REVIEW ME: double count of flops??
    PetscCall(PetscLogGpuFlops(PetscMax((*nvptr) * (2 * n - 1), 0)));
  } else {
    const auto zero = PetscScalar{0};

    PetscCall(PetscManagedScalarApplyOperator(dctx, z, PETSC_OPERATOR_EQUAL, PETSC_MEMTYPE_HOST, &zero, nullptr));
  }
  PetscFunctionReturn(0);
}

namespace kernels {

PETSC_KERNEL_DECL static void set_kernel(const PetscInt n, PetscScalar *PETSC_RESTRICT array, const PetscScalar *PETSC_RESTRICT v) {
  PETSC_SHAREDMEM_DECL PetscScalar value_shm;

  // thread leader of each block loads from global to shared memory
  if (!threadIdx.x) value_shm = *v;
  __syncthreads();
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(n, [=](PetscInt i) { array[i] = value_shm; });
  return;
}

} // namespace kernels

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto   n    = xin->map->n;
  const auto   xptr = DeviceArrayWrite(dctx, xin);
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &stream));
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    PetscCall(PetscCUPMMemsetAsync(xptr.data(), 0, n, stream));
  } else {
    PetscMemType mtype;
    PetscScalar *ptr;

    PetscCall(PetscManagedScalarGetPointerAndMemType(dctx, alpha, PETSC_MEMORY_ACCESS_READ, &ptr, &mtype));
    if (PetscMemTypeHost(mtype)) {
      PetscCall(device::cupm::impl::ThrustSet<T>(stream, n, xptr.data(), ptr));
    } else {
      PetscCall(PetscCUPMLaunchKernel1D(n, 0, stream, kernels::set_kernel, n, xptr.data(), ptr));
    }
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  NVTX_RANGE;
  if (PetscManagedScalarKnownAndEqual(alpha, 1.0)) PetscFunctionReturn(0);
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    PetscCall(set_async(xin, alpha, dctx));
  } else {
    const auto       n      = static_cast<cupmBlasInt_t>(xin->map->n);
    const auto       amtype = PETSC_MEMTYPE_DEVICE;
    PetscScalar     *aptr;
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    PetscCall(PetscManagedScalarGetArray(dctx, alpha, amtype, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, amtype};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, cupmScalarCast(aptr), DeviceArrayReadWrite(dctx, xin), 1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(n));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto       n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscScalar     *zptr;
  cupmBlasHandle_t cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &cupmBlasHandle));
  PetscCall(PetscManagedScalarGetArray(dctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr));
  {
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};

    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXdotu(cupmBlasHandle, n, DeviceArrayRead(dctx, xin), 1, DeviceArrayRead(dctx, yin), 1, cupmScalarCast(zptr)));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(PetscMax(2 * n - 1, 0)));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  NVTX_RANGE;
  if (xin != yin) {
    const auto   n     = xin->map->n;
    const auto   xmask = xin->offloadmask;
    // silence buggy gcc warning: ‘mode’ may be used uninitialized in this function
    auto         mode  = cupmMemcpyDeviceToDevice;
    cupmStream_t stream;

    // translate from PetscOffloadMask to cupmMemcpyKind
    switch (const auto ymask = yin->offloadmask) {
    case PETSC_OFFLOAD_UNALLOCATED: {
      PetscBool yiscupm;

      PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(yin), &yiscupm, VECSEQCUPM(), VECMPICUPM(), ""));
      if (yiscupm) {
        mode = PetscOffloadDevice(xmask) ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToHost;
        break;
      }
    } // fall-through if unallocated and not cupm
    case PETSC_OFFLOAD_CPU: mode = PetscOffloadHost(xmask) ? cupmMemcpyHostToHost : cupmMemcpyDeviceToHost; break;
    case PETSC_OFFLOAD_BOTH:
    case PETSC_OFFLOAD_GPU: mode = PetscOffloadDevice(xmask) ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice; break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incompatible offload mask %s", PetscOffloadMasks(ymask));
    }

    PetscCall(GetHandles_(dctx, &stream));
    switch (mode) {
    case cupmMemcpyDeviceToDevice: // the best case
    case cupmMemcpyHostToDevice: { // not terrible
      const auto yptr = DeviceArrayWrite(dctx, yin);
      const auto xptr = mode == cupmMemcpyDeviceToDevice ? DeviceArrayRead(dctx, xin).data() : HostArrayRead(dctx, xin).data();

      PetscCall(PetscLogGpuTimeBegin());
      PetscCall(PetscCUPMMemcpyAsync(yptr.data(), xptr, n, mode, stream));
      PetscCall(PetscLogGpuTimeEnd());
    } break;
    case cupmMemcpyDeviceToHost: // not great
    case cupmMemcpyHostToHost: { // worst case
      const auto   xptr = mode == cupmMemcpyDeviceToHost ? DeviceArrayRead(dctx, xin).data() : HostArrayRead(dctx, xin).data();
      PetscScalar *yptr;

      PetscCall(VecGetArrayWriteAsync(yin, &yptr, dctx));
      if (mode == cupmMemcpyDeviceToHost) PetscCall(PetscLogGpuTimeBegin());
      PetscCall(PetscCUPMMemcpyAsync(yptr, xptr, n, mode, stream, /* force async */ true));
      if (mode == cupmMemcpyDeviceToHost) PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecRestoreArrayWriteAsync(yin, &yptr, dctx));
    } break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "Unknown cupmMemcpyKind %d", static_cast<int>(mode));
    }
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (xin != yin) {
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXswap(cupmBlasHandle, static_cast<cupmBlasInt_t>(xin->map->n), DeviceArrayReadWrite(dctx, xin), 1, DeviceArrayReadWrite(dctx, yin), 1));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscManagedScalar alpha, PetscManagedScalar beta, Vec xin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    PetscCall(scale_async(yin, beta, dctx));
  } else if (PetscManagedScalarKnownAndEqual(beta, 1.0)) {
    PetscCall(axpy_async(yin, alpha, xin, dctx));
  }
  if (PetscManagedScalarKnownAndEqual(alpha, 1.0)) {
    PetscCall(aypx_async(yin, beta, xin, dctx));
  } else {
    const auto       bzero = PetscManagedScalarKnownAndEqual(beta, 0.0);
    const auto       n     = static_cast<cupmBlasInt_t>(yin->map->n);
    const auto       xptr  = DeviceArrayRead(dctx, xin);
    cupmBlasHandle_t cupmBlasHandle;
    PetscScalar     *aptr;

    PetscCall(PetscManagedScalarGetArray(dctx, alpha, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &aptr));
    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    {
      const auto calpha = cupmScalarCast(aptr);
      const auto cmode  = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};

      if (bzero /* beta = 0 */) {
        // here we can get away with purely write-only as we memcpy into it first
        const auto   yptr = DeviceArrayWrite(dctx, yin);
        cupmStream_t stream;

        PetscCall(GetHandles_(dctx, &stream));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCall(PetscCUPMMemcpyAsync(yptr.data(), xptr.data(), n, cupmMemcpyDeviceToDevice, stream));
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, calpha, yptr.cupmdata(), 1));
      } else {
        const auto   yptr = DeviceArrayReadWrite(dctx, yin);
        PetscScalar *bptr;

        PetscCall(PetscManagedScalarGetArray(dctx, beta, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &bptr));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle, n, cupmScalarCast(bptr), yptr.cupmdata(), 1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle, n, calpha, xptr.cupmdata(), 1, yptr.cupmdata(), 1));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops((bzero ? 1 : 3) * n));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec xin, Vec yin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  // note this fires even if gamma may secretely be equal to 1 since known may return false
  if (!PetscManagedScalarKnownAndEqual(gamma, 1.0)) PetscCall(scale_async(zin, gamma, dctx));
  PetscCall(axpy_async(zin, alpha, xin, dctx));
  PetscCall(axpy_async(zin, beta, yin, dctx));
  PetscFunctionReturn(0);
}

namespace kernels {

PETSC_KERNEL_DECL static void norm_infinity(const PetscScalar *PETSC_RESTRICT x, PetscReal *PETSC_RESTRICT value, const PetscInt *PETSC_RESTRICT i) {
  *value = PetscAbsScalar(x[(*i) - 1]);
  return;
}

} // namespace kernels

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  NVTX_RANGE;
  if (const auto n = static_cast<cupmBlasInt_t>(xin->map->n)) {
    const auto       xptr      = DeviceArrayRead(dctx, xin);
    PetscInt         flopCount = 0;
    cupmBlasHandle_t cupmBlasHandle;
    PetscReal       *zptr;

    PetscCall(GetHandles_(dctx, &cupmBlasHandle));
    PetscCall(PetscManagedRealGetArray(dctx, z, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &zptr));
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle, CUPMBLAS_POINTER_MODE_DEVICE};
    // one-shot cublas kernels are often much faster if run on the host
    PetscCall(PetscLogGpuTimeBegin());
    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      PetscCallCUPMBLAS(cupmBlasXasum(cupmBlasHandle, n, xptr.cupmdata(), 1, cupmRealCast(zptr)));
      flopCount = PetscMax(n - 1, 0);
      if (type == NORM_1) break;
      ++zptr; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      PetscCallCUPMBLAS(cupmBlasXnrm2(cupmBlasHandle, n, xptr.cupmdata(), 1, cupmRealCast(zptr)));
      flopCount += PetscMax(2 * n - 1, 0); // += in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmStream_t    stream;
      PetscManagedInt iscal;
      PetscInt       *i_ptr;

      PetscCall(GetHandles_(dctx, &stream));
      PetscCall(PetscManagedIntCreateDefault(dctx, 1, &iscal));
      PetscCall(PetscManagedIntGetArray(dctx, iscal, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &i_ptr));
      PetscCallCUPMBLAS(cupmBlasXamax(cupmBlasHandle, n, xptr.cupmdata(), 1, i_ptr));
      PetscCall(PetscCUPMLaunchKernel1D(1, 0, stream, kernels::norm_infinity, xptr.data(), zptr, i_ptr));
      PetscCall(PetscManagedIntDestroy(dctx, &iscal));
      // REVIEW ME: flopCount = ???
    } break;
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(flopCount));
  } else {
    constexpr PetscReal zero[] = {0.0, 0.0};

    PetscCall(PetscManagedRealSetValues(dctx, z, PETSC_MEMTYPE_HOST, zero, 1 + (type == NORM_1_AND_2)));
  }
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscDeviceContext *sub;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextFork(dctx, 1, &sub));
  PetscCall(dot_async(s, t, dp, dctx));
  PetscCall(dot_async(t, t, nm, *sub));
  PetscCall(PetscDeviceContextJoin(dctx, 1, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &sub));
  PetscFunctionReturn(0);
}

namespace detail {

struct conjugate {
  PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec xin, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) PetscCall(pointwiseunary_async_(detail::conjugate{}, dctx, xin));
  PetscFunctionReturn(0);
}

namespace detail {

struct real_part {
  PETSC_HOSTDEVICE_INLINE_DECL
  thrust::tuple<PetscReal, PetscInt> operator()(const thrust::tuple<PetscScalar, PetscInt> &x) const { return thrust::make_tuple(PetscRealPart(x.get<0>()), x.get<1>()); }

  PETSC_HOSTDEVICE_INLINE_DECL PetscReal operator()(PetscScalar x) const { return PetscRealPart(x); }
};

template <typename Operator>
struct tuple_compare {
  using tuple_type = thrust::tuple<PetscReal, PetscInt>;

  PETSC_HOSTDEVICE_INLINE_DECL tuple_type operator()(const tuple_type &x, const tuple_type &y) const {
    if (Operator{}(y.get<0>(), x.get<0>())) {
      // if y is strictly greater/less than x, return y
      return y;
    } else if (y.get<0>() == x.get<0>()) {
      // if equal, prefer lower index
      return y.get<1>() < x.get<1>() ? y : x;
    } else {
      // otherwise return x
      return x;
    }
  }
};

} // namespace detail

template <device::cupm::DeviceType T>
template <typename TupleFuncT, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT &&tuple_ftr, UnaryFuncT &&unary_ftr, PetscReal mval, Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (const auto n = v->map->n) {
    const auto   vptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, v).data());
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx, &stream));
    // needed to:
    // 1. switch between transform_reduce and reduce
    // 2. strip the real_part functor from the arguments
#if PetscDefined(USE_COMPLEX)
#define THRUST_MINMAX_REDUCE(...) THRUST_CALL(thrust::transform_reduce, __VA_ARGS__)
#else
#define THRUST_MINMAX_REDUCE(s, b, e, real_part__, ...) THRUST_CALL(thrust::reduce, s, b, e, __VA_ARGS__)
#endif
    if (p) {
      auto       pval = PetscInt{-1};
      const auto zip  = thrust::make_zip_iterator(thrust::make_tuple(std::move(vptr), thrust::make_counting_iterator(PetscInt{0})));
      // need to use preprocessor conditionals since otherwise thrust complains about not being
      // able to convert a thrust::device_reference<PetscScalar> to a PetscReal on complex
      // builds...
      PetscCallThrust(thrust::tie(mval, pval) = THRUST_MINMAX_REDUCE(stream, zip, zip + n, detail::real_part{}, thrust::make_tuple(mval, pval), std::forward<TupleFuncT>(tuple_ftr)););
      PetscCall(PetscManagedIntSetValues(dctx, p, PETSC_MEMTYPE_HOST, &pval, 1));
    } else {
      PetscCallThrust(mval = THRUST_MINMAX_REDUCE(stream, vptr, vptr + n, detail::real_part{}, mval, std::forward<UnaryFuncT>(unary_ftr)););
    }
#undef THRUST_MINMAX_REDUCE
  }
  PetscCall(PetscManagedRealSetValues(dctx, m, PETSC_MEMTYPE_HOST, &mval, 1));
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx)) {
  using value_type    = util::remove_pointer_t<decltype(m->device)>;
  using tuple_functor = detail::tuple_compare<thrust::greater<value_type>>;
  using unary_functor = thrust::maximum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{}, unary_functor{}, PETSC_MIN_REAL, v, p, m, dctx));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx)) {
  using value_type    = util::remove_pointer_t<decltype(m->device)>;
  using tuple_functor = detail::tuple_compare<thrust::less<value_type>>;
  using unary_functor = thrust::minimum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{}, unary_functor{}, PETSC_MAX_REAL, v, p, m, dctx));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscManagedScalar sum, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto   n    = v->map->n;
  const auto   dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx, v).data());
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &stream));
  // REVIEW ME: why not cupmBlasXasum()?
  PetscCallThrust(const auto ret = THRUST_CALL(thrust::reduce, stream, dptr, dptr + n, PetscScalar{0.0});
                  // need this inside since we have to capture the output
                  PetscCall(PetscManagedScalarSetValues(dctx, sum, PETSC_MEMTYPE_HOST, &ret, 1)););
  // REVIEW ME: must be at least n additions
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscManagedScalar shift, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscScalar *ptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetArray(dctx, shift, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &ptr));
  PetscCall(pointwiseunary_async_(device::cupm::impl::make_shift_operator(ptr, thrust::plus<PetscScalar>{}), dctx, v));
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand, PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto n = v->map->n;
  PetscBool  iscurand;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(rand), PETSCCURAND, &iscurand));
  if (iscurand) PetscCall(PetscRandomGetValues(rand, n, DeviceArrayWrite(dctx, v)));
  else PetscCall(PetscRandomGetValues(rand, n, HostArrayWrite(dctx, v)));
  // REVIEW ME: flops????
  // REVIEW ME: Timing???
  PetscFunctionReturn(0);
}

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setpreallocationcoo_async(Vec v, PetscCount ncoo, const PetscInt coo_i[], PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecSetPreallocationCOO_Seq(v, ncoo, coo_i, dctx));
  PetscCall(SetPreallocationCOO_CUPMBase(v, ncoo, coo_i, dctx));
  PetscFunctionReturn(0);
}

namespace kernels {

template <typename F>
PETSC_DEVICE_INLINE_DECL void add_coo_values_impl(const PetscScalar *PETSC_RESTRICT vv, PetscCount n, const PetscCount *PETSC_RESTRICT jmap, const PetscCount *PETSC_RESTRICT perm, InsertMode imode, PetscScalar *PETSC_RESTRICT xv, F &&xvindex) {
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(n, [=](PetscCount i) {
    const auto  end = jmap[i + 1];
    const auto  idx = xvindex(i);
    PetscScalar sum = 0.0;

    for (auto k = jmap[i]; k < end; ++k) sum += vv[perm[k]];

    if (imode == INSERT_VALUES) {
      xv[idx] = sum;
    } else {
      xv[idx] += sum;
    }
  });
  return;
}

PETSC_KERNEL_DECL static void add_coo_values(const PetscScalar *PETSC_RESTRICT v, PetscCount n, const PetscCount *PETSC_RESTRICT jmap1, const PetscCount *PETSC_RESTRICT perm1, InsertMode imode, PetscScalar *PETSC_RESTRICT xv) {
  add_coo_values_impl(v, n, jmap1, perm1, imode, xv, [](PetscCount i) { return i; });
  return;
}

} // namespace kernels

template <device::cupm::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setvaluescoo_async(Vec x, const PetscScalar v[], InsertMode imode, PetscDeviceContext dctx)) {
  auto         vv = const_cast<PetscScalar *>(v);
  PetscMemType memtype;
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx, &stream));
  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) {
    const auto size = VecIMPLCast(x)->coo_n;

    // If user gave v[] in host, we might need to copy it to device if any
    PetscCall(PetscCUPMMallocAsync(&vv, size, stream));
    PetscCall(PetscCUPMMemcpyAsync(vv, v, size, cupmMemcpyHostToDevice, stream));
  }

  if (const auto n = x->map->n) {
    const auto vcu = VecCUPMCast(x);

    PetscCall(PetscCUPMLaunchKernel1D(n, 0, stream, kernels::add_coo_values, vv, n, vcu->jmap1_d, vcu->perm1_d, imode, imode == INSERT_VALUES ? DeviceArrayWrite(dctx, x).data() : DeviceArrayReadWrite(dctx, x).data()));
  }

  if (PetscMemTypeHost(memtype)) PetscCallCUPM(cupmFreeAsync(vv, stream));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                 implementations                                    //

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCreateSeqCUPMAsync(T &&VecSeq_CUPM_Impls, MPI_Comm comm, PetscInt n, PetscDeviceContext dctx, Vec *v)) {
  PetscFunctionBegin;
  PetscValidPointer(v, 4);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.createseqcupm_async(comm, 0, n, dctx, v, PETSC_TRUE));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCreateSeqCUPMWithArraysAsync(T &&VecSeq_CUPM_Impls, MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], PetscDeviceContext dctx, Vec *v)) {
  PetscFunctionBegin;
  if (n && cpuarray) PetscValidScalarPointer(cpuarray, 5);
  PetscValidPointer(v, 7);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.createseqcupmwithbotharrays_async(comm, bs, n, cpuarray, gpuarray, dctx, v));
  PetscFunctionReturn(0);
}

template <PetscMemoryAccessMode mode, typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayAsync_Private(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscValidPointer(a, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template getarray_async<PETSC_MEMTYPE_DEVICE, mode>(v, a, dctx));
  PetscFunctionReturn(0);
}

template <PetscMemoryAccessMode mode, typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayAsync_Private(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template restorearray_async<PETSC_MEMTYPE_DEVICE, mode>(v, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayAsync(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE>(std::forward<T>(VecSeq_CUPM_Impls), v, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayAsync(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE>(std::forward<T>(VecSeq_CUPM_Impls), v, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayReadAsync(T &&VecSeq_CUPM_Impls, Vec v, const PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ>(std::forward<T>(VecSeq_CUPM_Impls), v, const_cast<PetscScalar **>(a), dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayReadAsync(T &&VecSeq_CUPM_Impls, Vec v, const PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ>(std::forward<T>(VecSeq_CUPM_Impls), v, const_cast<PetscScalar **>(a), dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayWriteAsync(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE>(std::forward<T>(VecSeq_CUPM_Impls), v, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayWriteAsync(T &&VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE>(std::forward<T>(VecSeq_CUPM_Impls), v, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMPlaceArrayAsync(T &&VecSeq_CUPM_Impls, Vec vin, const PetscScalar a[], PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template placearray_async<PETSC_MEMTYPE_DEVICE>(vin, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMReplaceArrayAsync(T &&VecSeq_CUPM_Impls, Vec vin, const PetscScalar a[], PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template replacearray_async<PETSC_MEMTYPE_DEVICE>(vin, a, dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMResetArrayAsync(T &&VecSeq_CUPM_Impls, Vec vin, PetscDeviceContext dctx)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin, VEC_CLASSID, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template resetarray_async<PETSC_MEMTYPE_DEVICE>(vin, dctx));
  PetscFunctionReturn(0);
}

// REVIEW ME: TODO figure out a way around gcc 9.3.1 linker bug to get these to
// work. Everything compiles but get linker error in optimized build for mdot_async_(). I think
// this is because it is overloaded function and since we __always__ only call one overload,
// compiler doesn't generate the object code for the other overload...
// /usr/bin/ld: arch-linux-c-opt/lib/libpetsc.so: undefined reference to
// Petsc::Vector::CUPM::Impl::VecSeq_CUPM<(Petsc::device::cupm::DeviceType)0>::mdot_async_(Petsc::Vector::CUPM::Impl::(anonymousnamespace)::UseComplexTag<false>,
// _p_Vec*, int, _p_Vec* const*, double*)'

// declare the extern templates, each is explicitly instantiated in the respective
// implementation directories
#if PetscDefined(HAVE_CUDA)
//extern template struct VecSeq_CUPM<device::cupm::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
//extern template struct VecSeq_CUPM<Device::CUPM::DeviceType::HIP>;
#endif

} // namespace impl

} // namespace cupm

} // namespace vec

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
