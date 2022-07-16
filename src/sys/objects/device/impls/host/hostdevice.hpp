#ifndef HOSTDEVICE_HPP
#define HOSTDEVICE_HPP

#if defined(__cplusplus)
#include "../impldevicebase.hpp" /*I "petscdevice.h" I*/

namespace Petsc {

namespace device {

namespace host {

struct Device : ::Petsc::device::impl::DeviceBase<Device> {
private:
  PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_type, Device);

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType GetPetscDeviceType_()) { return PETSC_DEVICE_HOST; }

public:
  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm, PetscInt *, PetscDeviceInitType *) noexcept;
  PETSC_NODISCARD PetscErrorCode        getDevice(PetscDevice, PetscInt) const noexcept;
  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;
  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice, PetscViewer) noexcept;
};

} // namespace host

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // HOSTDEVICE_HPP
