!
!  Description: Demonstrate copying a global vector on a DMDA3d grid to global
!               vectors on DMDA3d grids created on a bunch of subcommunicators,
!               with preserving natural ordering of the grid points.
!
!  Contributed-by: Randall Mackie <rlmackie862@gmail.com>
!
!  Note: This example was used to show an MPI failure at large scale caused by
!        overwhelming traffic during VecScatter setup.
!
!        Seealso https://lists.mcs.anl.gov/pipermail/petsc-users/2020-April/040788.html
! ---------------------------------------------------------------------------------------

Program main

#include "petsc/finclude/petscdmda.h"
  use petscdmda

  implicit none

  DM             :: daGlobal, daSub
  Vec            :: vtmpParent, transVec, vtmp
  AO             :: aoParent, aoSub
  IS             :: from, to
  VecScatter     :: parentToSub
  PetscInt, allocatable  :: ind1(:), ind2(:)
  PetscErrorCode :: ierr
  PetscScalar, pointer :: ptr_v(:,:,:,:)
  PetscScalar, pointer :: ptr_x1(:)
  PetscReal      :: rval
  PetscInt       :: nx, ny, nz, tmp_nsubs
  PetscInt       :: i, j, k, nis, iComp
  PetscInt       :: M, mstart, mend, mlocal
  PetscInt       :: xs, ys, zs, xm, ym, zm, xe, ye, ze
  PetscInt       :: xs_s, ys_s, zs_s, xm_s, ym_s, zm_s, xe_s, ye_s, ze_s
  PetscBool      :: flg
  PetscMPIInt    :: size, rank, subrank, subsize, nproc_per_sub, sub, nsubs
  character(len=200) :: cstring
  MPI_Comm :: subcomm, comm_parent
  PetscMPIInt, parameter :: i0=0
  PetscInt, parameter    :: i1=1
  PetscInt, parameter    :: i2=2

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr);CHKERRA(ierr)
  ! defaults
  nx = 8
  ny = 8
  nz = 8
  tmp_nsubs = 1

  ! input some parameters
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nx',nx,flg,ierr);CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-ny',ny,flg,ierr);CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nz',nz,flg,ierr);CHKERRA(ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nsubs',tmp_nsubs,flg,ierr);CHKERRA(ierr)
  nsubs = int(tmp_nsubs,kind(nsubs))

  !=========================================================================================
  ! create subcomms
  call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr);CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)

  if (mod(size, nsubs) /= 0) then
    cstring='The total number of MPI processes is not a multiple of nsubcomm\n\n'
    call PetscPrintf(PETSC_COMM_WORLD,TRIM(cstring),ierr);CHKERRA(ierr)
    goto 999
  end if

  ! Initiate sub-communicators
  nsubs = min(size,nsubs)
  nproc_per_sub = size / nsubs
  sub = rank / nproc_per_sub

  ! Create subcommunicator and get team ranks
  call MPI_Comm_split(PETSC_COMM_WORLD, sub, i0, SUBCOMM, ierr);CHKERRA(ierr)
  call MPI_Comm_rank(SUBCOMM,subrank,ierr);CHKERRA(ierr)
  call MPI_Comm_size(SUBCOMM,subsize,ierr);CHKERRA(ierr)
  !=========================================================================================

  ! set up DM on entire comm
  call DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, &
    DMDA_STENCIL_BOX,nx,ny,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,i2,i2,PETSC_NULL_INTEGER, &
    PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, daGlobal,ierr);CHKERRA(ierr)
  call DMSetUp(daGlobal,ierr);CHKERRA(ierr)

  ! Get the Global DMDA grid info for each processor
  call DMDAGetCorners(daGlobal,xs,ys,zs,xm,ym,zm,ierr);CHKERRA(ierr)
  xs=xs+1
  ys=ys+1
  zs=zs+1
  xe=xs+xm-1
  ye=ys+ym-1
  ze=zs+zm-1

  ! Set AO type here
  call DMDASetAOType(daGlobal,AOMEMORYSCALABLE,ierr);CHKERRA(ierr)

  ! set up DM on subcomm
  call DMDACreate3d(subcomm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, &
    DMDA_STENCIL_BOX,nx,ny,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,i2,i2,PETSC_NULL_INTEGER, &
    PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,daSub,ierr);CHKERRA(ierr)
  call DMSetUp(daSub,ierr);CHKERRA(ierr)

  ! Get the SUB DMDA grid info for each processor
  call DMDAGetCorners(daSub,xs_s,ys_s,zs_s,xm_s,ym_s,zm_s,ierr);CHKERRA(ierr)
  xs_s=xs_s+1
  ys_s=ys_s+1
  zs_s=zs_s+1
  xe_s=xs_s+xm_s-1
  ye_s=ys_s+ym_s-1
  ze_s=zs_s+zm_s-1

  ! Set AO type here
  call DMDASetAOType(daSub,AOMEMORYSCALABLE,ierr);CHKERRA(ierr)

  !=========================================================================================
  ! Create Vector Scatter from parent to subcomm
  !=========================================================================================

  ! Get communicator for parent DA
  call PetscObjectGetComm(daGlobal, comm_parent, ierr);CHKERRA(ierr)

  ! Get application orderings
  call DMDAGetAO(daGlobal,aoParent,ierr);CHKERRA(ierr)
  call DMDAGetAO(daSub,aoSub,ierr);CHKERRA(ierr)

  ! Get vector on global comm and its size
  call DMGetGlobalVector(daGlobal,vtmpParent,ierr);CHKERRA(ierr)
  call VecGetSize(vtmpParent,M,ierr);CHKERRA(ierr)
  call VecGetOwnershipRange(vtmpParent,mstart,mend,ierr);CHKERRA(ierr)
  mlocal=mend-mstart

  ! Create index sets for scatter
  nis=mlocal*nsubs
  allocate (ind1(0:nis-1), ind2(0:nis-1), STAT=ierr);CHKERRA(ierr)
  ind1(0:nis-1)=0 ! To avoid gcc warning: 'xx.dim[0].ubound' may be used uninitialized in this function [-Wmaybe-uninitialized]
  ind2(0:nis-1)=0
  j=0
  do k=0,nsubs-1
    do i=mstart,mend-1
      ind1(j)=i
      j=j+1
    end do
  end do
  ind2(0:nis-1)=ind1(0:nis-1)

  call AOApplicationToPetsc(aoParent,nis,ind1,ierr);CHKERRA(ierr)
  call AOApplicationToPetsc(aoSub,nis,ind2,ierr);CHKERRA(ierr)

  j=0
  do k=0,nsubs-1
    do i=mstart,mend-1
      ind2(j)=ind2(j)+M*k
      j=j+1
    end do
  end do

  call ISCreateGeneral(comm_parent,nis,ind1,PETSC_COPY_VALUES,from,ierr);CHKERRA(ierr)
  call ISCreateGeneral(comm_parent,nis,ind2,PETSC_COPY_VALUES,to,ierr);CHKERRA(ierr)
  deallocate (ind1,ind2)

  ! Create empty global vector to receive vector scatter on all sub communicators
  call DMGetGlobalVector(daSub,vtmp,ierr);CHKERRA(ierr)
  call VecGetLocalSize(vtmp,mlocal,ierr);CHKERRA(ierr)
  call VecCreateMPIWithArray(comm_parent,i1,mlocal,PETSC_DECIDE,PETSC_NULL_SCALAR,transVec,ierr);CHKERRA(ierr)

  ! Create vector scatter
  call VecScatterCreate(vtmpParent,from,transVec,to,parentToSub,ierr);CHKERRA(ierr)
  !cstring='Successful VecScatter create\n'
  !call PetscPrintf(PETSC_COMM_WORLD,TRIM(cstring),ierr);CHKERRA(ierr)

  ! Release memory
  call ISDestroy(from,ierr);CHKERRA(ierr)
  call ISDestroy(to,ierr);CHKERRA(ierr)
  call VecDestroy(transVec, ierr);CHKERRA(ierr)
  call DMRestoreGlobalVector(daSub,vtmp,ierr);CHKERRA(ierr)

  !=========================================================================================
  ! Now do the vector Scatter from parent to subcomm
  !=========================================================================================

  ! Set some elements in the global vector of the Global DMDA
  call DMDAVecGetArrayF90(daGlobal,vtmpParent,ptr_v,ierr);CHKERRA(ierr)
  do k=zs,ze
    do j=ys,ye
      do i=xs,xe
        do iComp=1,i2
          ptr_v(iComp-1,i-1,j-1,k-1)=1000.0*real(iComp)+100.0*real(i) + 10.0*real(j) + real(k)
        end do
      end do
    end do
  end do
  call DMDAVecRestoreArrayF90(daGlobal,vtmpParent,ptr_v,ierr);CHKERRA(ierr)

  call DMGetGlobalVector(daSub,vtmp,ierr);CHKERRA(ierr)
  call VecGetLocalSize(vtmp,mlocal,ierr);CHKERRA(ierr)
  call VecGetArrayF90(vtmp,ptr_x1,ierr);CHKERRA(ierr)
  call VecCreateMPIWithArray(comm_parent,i1,mlocal,PETSC_DECIDE,ptr_x1,transVec,ierr);CHKERRA(ierr)

  call VecScatterBegin(parentToSub,vtmpParent,transVec,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
  call VecScatterEnd(parentTosub,vtmpParent,transVec,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)

  call VecDestroy(transVec, ierr);CHKERRA(ierr)
  call VecRestoreArrayF90(vtmp,ptr_x1,ierr);CHKERRA(ierr)

  ! Check that our scatter was correct
  call DMDAVecGetArrayF90(daSub,vtmp,ptr_v,ierr);CHKERRA(ierr)
  do k=zs_s,ze_s
    do j=ys_s,ye_s
      do i=xs_s,xe_s
        do iComp=1,i2
          rval=1000.0*real(iComp)+100.0*real(i) + 10.0*real(j) + real(k)
          if (rval /= ptr_v(iComp-1,i-1,j-1,k-1)) print*, 'problem ',rank,i,j,k,rval,ptr_v(iComp-1,i-1,j-1,k-1)
        end do
      end do
    end do
  end do
  call DMDAVecRestoreArrayF90(daSub,vtmp,ptr_v,ierr);CHKERRA(ierr)

  ! Release memory
  call VecScatterDestroy(parentToSub,ierr);CHKERRA(ierr)
  call DMRestoreGlobalVector(daGlobal,vtmpParent,ierr);CHKERRA(ierr)
  call DMRestoreGlobalVector(daSub,vtmp,ierr);CHKERRA(ierr)
  call DMDestroy(daGlobal,ierr);CHKERRA(ierr)
  call DMDestroy(dasub,ierr);CHKERRA(ierr)
  call MPI_Comm_free(SUBCOMM,ierr);CHKERRA(ierr)

  999 continue
  call PetscFinalize(ierr);

end program main

!/*TEST
!
!   testset:
!     nsize: 8
!     args: -nsubs 2 -nx 4 -ny 4 -nz 4
!     output_file: output/ex1f.out
!
!     test:
!
!     test:
!       suffix: 2
!       args: -max_pending_isends 1
!
!TEST*/
