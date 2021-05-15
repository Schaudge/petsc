#ifndef __SEQAIJKOKKOSIMPL_HPP
#define __SEQAIJKOKKOSIMPL_HPP

#include <petscaijdevice.h>
#include <petsc/private/vecimpl_kokkos.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spiluk.hpp>

/*
   Kokkos::View<struct _n_SplitCSRMat,DefaultMemorySpace> is not handled correctly so we define SplitCSRMat
   for the singular purpose of working around this.
*/
typedef struct _n_SplitCSRMat SplitCSRMat;

using MatRowOffsetType    = PetscInt;
using MatColumnIndexType  = PetscInt;
using MatScalarType       = PetscScalar;

template<class MemorySpace> using KokkosCsrMatrixType   = typename KokkosSparse::CrsMatrix<MatScalarType,MatColumnIndexType,MemorySpace,void/* MemoryTraits */,MatRowOffsetType>;
template<class MemorySpace> using KokkosCsrGraphType    = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph                      = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrGraphHost                  = KokkosCsrGraphType<Kokkos::HostSpace>;

using KokkosCsrMatrix                     = KokkosCsrMatrixType<DefaultMemorySpace>;
using KokkosCsrMatrixHost                 = KokkosCsrMatrixType<Kokkos::HostSpace>;

using MatRowOffsetKokkosView              = KokkosCsrGraph::row_map_type::non_const_type;
using MatColumnIndexKokkosView            = KokkosCsrGraph::entries_type::non_const_type;
using MatScalarKokkosView                 = KokkosCsrMatrix::values_type::non_const_type;

using MatRowOffsetKokkosViewHost          = KokkosCsrGraphHost::row_map_type::non_const_type;
using MatColumnIndexKokkosViewHost        = KokkosCsrGraphHost::entries_type::non_const_type;
using MatScalarKokkosViewHost             = KokkosCsrMatrixHost::values_type::non_const_type;

using ConstMatRowOffsetKokkosView         = KokkosCsrGraph::row_map_type::const_type;
using ConstMatColumnIndexKokkosView       = KokkosCsrGraph::entries_type::const_type;
using ConstMatScalarKokkosView            = KokkosCsrMatrix::values_type::const_type;

using ConstMatRowOffsetKokkosViewHost     = KokkosCsrGraphHost::row_map_type::const_type;
using ConstMatColumnIndexKokkosViewHost   = KokkosCsrGraphHost::entries_type::const_type;
using ConstMatScalarKokkosViewHost        = KokkosCsrMatrixHost::values_type::const_type;

using MatScalarKokkosDualView             = Kokkos::DualView<MatScalarType*>;

using KernelHandle                        = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowOffsetType,MatColumnIndexType,MatScalarType,DefaultExecutionSpace,DefaultMemorySpace,DefaultMemorySpace>;

/* For mat->spptr of a factorized matrix */
struct Mat_SeqAIJKokkosTriFactors {
  MatRowOffsetKokkosView         iL_d,iU_d,iLt_d,iUt_d; /* rowmap for L, U, L^t, U^t of A=LU */
  MatColumnIndexKokkosView       jL_d,jU_d,jLt_d,jUt_d; /* column ids */
  MatScalarKokkosView            aL_d,aU_d,aLt_d,aUt_d; /* matrix values */
  KernelHandle                   kh,khL,khU,khLt,khUt;  /* Kernel handles for A, L, U, L^t, U^t */
  PetscBool                      transpose_updated;     /* Are L^T, U^T updated wrt L, U*/
  PetscBool                      sptrsv_symbolic_completed; /* Have we completed the symbolic solve for L and U */
  PetscScalarKokkosView          workVector;

  Mat_SeqAIJKokkosTriFactors(PetscInt n)
    : transpose_updated(PETSC_FALSE),sptrsv_symbolic_completed(PETSC_FALSE),workVector("workVector",n) {}

  ~Mat_SeqAIJKokkosTriFactors() {Destroy();}

  void Destroy() {
    kh.destroy_spiluk_handle();
    khL.destroy_sptrsv_handle();
    khU.destroy_sptrsv_handle();
    khLt.destroy_sptrsv_handle();
    khUt.destroy_sptrsv_handle();
    transpose_updated = sptrsv_symbolic_completed = PETSC_FALSE;
  }
};

/* For mat->spptr of a regular matrix */
struct Mat_SeqAIJKokkos {
  ConstMatRowOffsetKokkosViewHost i_h;
  ConstMatRowOffsetKokkosView     i_d;

  MatColumnIndexKokkosViewHost   j_h;
  MatColumnIndexKokkosView       j_d;

  MatScalarKokkosViewHost        a_h;
  MatScalarKokkosView            a_d;
  MatScalarKokkosDualView        a_dual;

  KokkosCsrGraph                 csrgraph;

  KokkosCsrMatrix                csrmat; /* The CSR matrix */
  PetscObjectState               nonzerostate; /* State of the nonzero pattern (graph) on device */

  Mat                            At,Ah; /* Transpose and Hermitian of the matrix in MATAIJKOKKOS type (built on demand) */
  PetscBool                      transpose_updated,hermitian_updated; /* Are At, Ah updated wrt the matrix? */

  Kokkos::View<PetscInt*>        *i_uncompressed_d;
  Kokkos::View<PetscInt*>        *colmap_d; // ugh, this is a parallel construct
  Kokkos::View<SplitCSRMat,DefaultMemorySpace> device_mat_d;
  Kokkos::View<PetscInt*>        *diag_d; // factorizations

   /* Construct Mat_SeqAIJKokkos for a nrows by ncols matrix with nnz nonzeros from the given (i,j,a), which are on host */
  Mat_SeqAIJKokkos(MatColumnIndexType nrows,MatColumnIndexType ncols,MatRowOffsetType nnz,const MatRowOffsetType *i,MatColumnIndexType *j,MatScalarType *a)
   : i_h(i,nrows+1),j_h(j,nnz),a_h(a,nnz)
  {
     i_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),i_h);
     j_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),j_h);
     a_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),a_h);
     csrgraph   = KokkosCsrGraph(j_d,i_d);
     a_dual     = MatScalarKokkosDualView(a_d,a_h);
     csrmat     = KokkosCsrMatrix("csrmat",ncols,a_d,csrgraph);

     Init();
  }

  /* Construct Mat_SeqAIJKokkos with a KokkosCsrMatrix, which is on device */
  Mat_SeqAIJKokkos(const KokkosCsrMatrix& csr)
    : i_d(csr.graph.row_map),j_d(csr.graph.entries),a_d(csr.values),csrgraph(csr.graph),csrmat(csr)
  {
    i_h        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),i_d);
    j_h        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),j_d);
    a_h        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),a_d);
    a_dual     = MatScalarKokkosDualView(a_d,a_h);
    Init();
  }

  ~Mat_SeqAIJKokkos()
  {
    DestroyMatTranspose();
  }

  /* Shared init stuff */
  void Init(void)
  {
    At = Ah = NULL;
    transpose_updated = hermitian_updated = PETSC_FALSE;
    i_uncompressed_d = colmap_d = diag_d = NULL;
  }

  PetscErrorCode DestroyMatTranspose(void)
  {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = MatDestroy(&At);CHKERRQ(ierr);
    ierr = MatDestroy(&Ah);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
};

struct MatProductData_SeqAIJKokkos {
  KernelHandle kh;
  bool transA,transB;
  MatProductData_SeqAIJKokkos(bool transA_,bool transB_) : transA(transA_),transB(transB_){}
};

#endif
