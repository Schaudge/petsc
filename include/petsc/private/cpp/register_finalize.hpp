#ifndef PETSC_CPP_REGISTER_FINALIZE_HPP
#define PETSC_CPP_REGISTER_FINALIZE_HPP

#include <petscsys.h>

#if defined(__cplusplus)
template <typename T>
PETSC_NODISCARD static inline PetscErrorCode PetscCxxObjectRegisterFinalize(T *obj, MPI_Comm comm = PETSC_COMM_SELF) noexcept {
  const auto finalizer = [](void *ptr) {
    PetscFunctionBegin;
    PetscCall(static_cast<T *>(ptr)->finalize());
    PetscFunctionReturn(0);
  };
  PetscContainer contain;

  PetscFunctionBegin;
  PetscCall(PetscContainerCreate(comm, &contain));
  PetscCall(PetscContainerSetPointer(contain, obj));
  PetscCall(PetscContainerSetUserDestroy(contain, std::move(finalizer)));
  PetscCall(PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain)));
  PetscFunctionReturn(0);
}

namespace Petsc {

template <typename Derived>
struct RegisterFinalizeable {
  using derived_type = Derived;

  PETSC_NODISCARD PetscErrorCode finalize() noexcept;
  PETSC_NODISCARD PetscErrorCode register_finalize(MPI_Comm comm = PETSC_COMM_SELF) noexcept;

private:
  RegisterFinalizeable() = default;
  friend derived_type;

  // default implementations if the derived class does not want to implement them
  PETSC_NODISCARD static PetscErrorCode finalize_() noexcept { return 0; }
  PETSC_NODISCARD static PetscErrorCode register_finalize_() noexcept { return 0; }

  bool registered_ = false;
};

template <typename D>
inline PetscErrorCode RegisterFinalizeable<D>::finalize() noexcept {
  PetscFunctionBegin;
  PetscCall(static_cast<derived_type *>(this)->finalize_());
  registered_ = false;
  PetscFunctionReturn(0);
}

template <typename D>
inline PetscErrorCode RegisterFinalizeable<D>::register_finalize(MPI_Comm comm) noexcept {
  PetscFunctionBegin;
  if (PetscLikely(registered_)) PetscFunctionReturn(0);
  registered_ = true;
  PetscCall(static_cast<derived_type *>(this)->register_finalize_());
  PetscCall(PetscCxxObjectRegisterFinalize(this, comm));
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_REGISTER_FINALIZE_HPP
