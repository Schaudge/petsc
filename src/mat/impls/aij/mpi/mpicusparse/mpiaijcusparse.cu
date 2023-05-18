#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>
#include <thrust/advance.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <petscsf.h>

static PetscErrorCode MatSetOps_MPIAIJCUSPARSE(Mat);

struct VecCUDAEquals {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

static PetscErrorCode MatResetPreallocationCOO_MPIAIJCUSPARSE(Mat mat)
{
  Mat_MPIAIJ         *a    = (Mat_MPIAIJ *)mat->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  // refcnt = 1 means 'mat' is the last owner of the coo data, therefore we free it.
  if (cusp && a->coo_refcnt && (*a->coo_refcnt == 1)) {
    if (cusp->use_extended_coo) {
      PetscCallCUDA(cudaFree(cusp->Ajmap1_d));
      PetscCallCUDA(cudaFree(cusp->Aperm1_d));
      PetscCallCUDA(cudaFree(cusp->Bjmap1_d));
      PetscCallCUDA(cudaFree(cusp->Bperm1_d));
      PetscCallCUDA(cudaFree(cusp->Aimap2_d));
      PetscCallCUDA(cudaFree(cusp->Ajmap2_d));
      PetscCallCUDA(cudaFree(cusp->Aperm2_d));
      PetscCallCUDA(cudaFree(cusp->Bimap2_d));
      PetscCallCUDA(cudaFree(cusp->Bjmap2_d));
      PetscCallCUDA(cudaFree(cusp->Bperm2_d));
      PetscCallCUDA(cudaFree(cusp->Cperm1_d));
      PetscCallCUDA(cudaFree(cusp->sendbuf_d));
      PetscCallCUDA(cudaFree(cusp->recvbuf_d));
    }
    cusp->use_extended_coo = PETSC_FALSE;
    delete cusp->coo_p;
    delete cusp->coo_pw;
    cusp->coo_p  = NULL;
    cusp->coo_pw = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE_Basic(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *a    = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE *)a->spptr;
  PetscInt            n    = cusp->coo_nd + cusp->coo_no;

  PetscFunctionBegin;
  if (cusp->coo_p && v) {
    thrust::device_ptr<const PetscScalar> d_v;
    THRUSTARRAY                          *w = NULL;

    if (isCudaMem(v)) {
      d_v = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      w->assign(v, v + n);
      PetscCall(PetscLogCpuToGpu(n * sizeof(PetscScalar)));
      d_v = w->data();
    }

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->coo_p->begin()), cusp->coo_pw->begin()));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v, cusp->coo_p->end()), cusp->coo_pw->end()));
    PetscCall(PetscLogGpuTimeBegin());
    thrust::for_each(zibit, zieit, VecCUDAEquals());
    PetscCall(PetscLogGpuTimeEnd());
    delete w;
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A, cusp->coo_pw->data().get(), imode));
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B, cusp->coo_pw->data().get() + cusp->coo_nd, imode));
  } else {
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A, v, imode));
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B, v ? v + cusp->coo_nd : NULL, imode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Tuple>
struct IsNotOffDiagT {
  PetscInt _cstart, _cend;

  IsNotOffDiagT(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) { }
  __host__ __device__ inline bool operator()(Tuple t) { return !(thrust::get<1>(t) < _cstart || thrust::get<1>(t) >= _cend); }
};

struct IsOffDiag {
  PetscInt _cstart, _cend;

  IsOffDiag(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) { }
  __host__ __device__ inline bool operator()(const PetscInt &c) { return c < _cstart || c >= _cend; }
};

struct GlobToLoc {
  PetscInt _start;

  GlobToLoc(PetscInt start) : _start(start) { }
  __host__ __device__ inline PetscInt operator()(const PetscInt &c) { return c - _start; }
};

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(Mat B, PetscCount n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_MPIAIJ            *b    = (Mat_MPIAIJ *)B->data;
  Mat_MPIAIJCUSPARSE    *cusp = (Mat_MPIAIJCUSPARSE *)b->spptr;
  PetscInt               N, *jj;
  size_t                 noff = 0;
  THRUSTINTARRAY         d_i(n); /* on device, storing partitioned coo_i with diagonal first, and off-diag next */
  THRUSTINTARRAY         d_j(n);
  ISLocalToGlobalMapping l2g;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&b->A));
  PetscCall(MatDestroy(&b->B));

  PetscCall(PetscLogCpuToGpu(2. * n * sizeof(PetscInt)));
  d_i.assign(coo_i, coo_i + n);
  d_j.assign(coo_j, coo_j + n);
  PetscCall(PetscLogGpuTimeBegin());
  auto firstoffd = thrust::find_if(thrust::device, d_j.begin(), d_j.end(), IsOffDiag(B->cmap->rstart, B->cmap->rend));
  auto firstdiag = thrust::find_if_not(thrust::device, firstoffd, d_j.end(), IsOffDiag(B->cmap->rstart, B->cmap->rend));
  if (firstoffd != d_j.end() && firstdiag != d_j.end()) {
    cusp->coo_p  = new THRUSTINTARRAY(n);
    cusp->coo_pw = new THRUSTARRAY(n);
    thrust::sequence(thrust::device, cusp->coo_p->begin(), cusp->coo_p->end(), 0);
    auto fzipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(), d_j.begin(), cusp->coo_p->begin()));
    auto ezipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(), d_j.end(), cusp->coo_p->end()));
    auto mzipp = thrust::partition(thrust::device, fzipp, ezipp, IsNotOffDiagT<thrust::tuple<PetscInt, PetscInt, PetscInt>>(B->cmap->rstart, B->cmap->rend));
    firstoffd  = mzipp.get_iterator_tuple().get<1>();
  }
  cusp->coo_nd = thrust::distance(d_j.begin(), firstoffd);
  cusp->coo_no = thrust::distance(firstoffd, d_j.end());

  /* from global to local */
  thrust::transform(thrust::device, d_i.begin(), d_i.end(), d_i.begin(), GlobToLoc(B->rmap->rstart));
  thrust::transform(thrust::device, d_j.begin(), firstoffd, d_j.begin(), GlobToLoc(B->cmap->rstart));
  PetscCall(PetscLogGpuTimeEnd());

  /* copy offdiag column indices to map on the CPU */
  PetscCall(PetscMalloc1(cusp->coo_no, &jj)); /* jj[] will store compacted col ids of the offdiag part */
  PetscCallCUDA(cudaMemcpy(jj, d_j.data().get() + cusp->coo_nd, cusp->coo_no * sizeof(PetscInt), cudaMemcpyDeviceToHost));
  auto o_j = d_j.begin();
  PetscCall(PetscLogGpuTimeBegin());
  thrust::advance(o_j, cusp->coo_nd); /* sort and unique offdiag col ids */
  thrust::sort(thrust::device, o_j, d_j.end());
  auto wit = thrust::unique(thrust::device, o_j, d_j.end()); /* return end iter of the unique range */
  PetscCall(PetscLogGpuTimeEnd());
  noff = thrust::distance(o_j, wit);
  /* allocate the garray, the columns of B will not be mapped in the multiply setup */
  PetscCall(PetscMalloc1(noff, &b->garray));
  PetscCallCUDA(cudaMemcpy(b->garray, d_j.data().get() + cusp->coo_nd, noff * sizeof(PetscInt), cudaMemcpyDeviceToHost));
  PetscCall(PetscLogGpuToCpu((noff + cusp->coo_no) * sizeof(PetscInt)));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, noff, b->garray, PETSC_COPY_VALUES, &l2g));
  PetscCall(ISLocalToGlobalMappingSetType(l2g, ISLOCALTOGLOBALMAPPINGHASH));
  PetscCall(ISGlobalToLocalMappingApply(l2g, IS_GTOLM_DROP, cusp->coo_no, jj, &N, jj));
  PetscCheck(N == cusp->coo_no, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected is size %" PetscInt_FMT " != %" PetscInt_FMT " coo size", N, cusp->coo_no);
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));

  PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
  PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
  PetscCall(MatSetType(b->A, MATSEQAIJCUSPARSE));
  PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
  PetscCall(MatSetSizes(b->B, B->rmap->n, noff, B->rmap->n, noff));
  PetscCall(MatSetType(b->B, MATSEQAIJCUSPARSE));

  /* GPU memory, cusparse specific call handles it internally */
  PetscCall(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->A, cusp->coo_nd, d_i.data().get(), d_j.data().get()));
  PetscCall(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->B, cusp->coo_no, d_i.data().get() + cusp->coo_nd, jj));
  PetscCall(PetscFree(jj));
  // TODO: remove the whole MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic() in favor of the extended COO.
  // UGLY: since we build COO data in *_Basic(), we need to set the reference count
  PetscCall(PetscMalloc1(1, &b->coo_refcnt));
  *b->coo_refcnt = 1;
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusp->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusp->offdiagGPUMatFormat));

  PetscCall(MatBindToCPU(b->A, B->boundtocpu));
  PetscCall(MatBindToCPU(b->B, B->boundtocpu));
  PetscCall(MatSetUpMultiply_MPIAIJ(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_MPIAIJ         *mpiaij = (Mat_MPIAIJ *)mat->data;
  Mat_MPIAIJCUSPARSE *mpidev;
  PetscBool           coo_basic = PETSC_TRUE;
  PetscMemType        mtype     = PETSC_MEMTYPE_DEVICE;
  PetscInt            rstart, rend;

  PetscFunctionBegin;
  PetscCall(PetscFree(mpiaij->garray));
  PetscCall(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&mpiaij->colmap));
#else
  PetscCall(PetscFree(mpiaij->colmap));
#endif
  PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
  mat->assembled     = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;
  // The two MatResetPreallocationCOO_* must be done in order. The former relies on values that might be destroyed by the latter
  PetscCall(MatResetPreallocationCOO_MPIAIJCUSPARSE(mat));
  PetscCall(MatResetPreallocationCOO_MPIAIJ(mat));
  if (coo_i) {
    PetscCall(PetscLayoutGetRange(mat->rmap, &rstart, &rend));
    PetscCall(PetscGetMemType(coo_i, &mtype));
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k = 0; k < coo_n; k++) { /* Are there negative indices or remote entries? */
        if (coo_i[k] < 0 || coo_i[k] < rstart || coo_i[k] >= rend || coo_j[k] < 0) {
          coo_basic = PETSC_FALSE;
          break;
        }
      }
    }
  }
  /* All ranks must agree on the value of coo_basic */
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &coo_basic, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)mat)));
  if (coo_basic) {
    PetscCall(MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(mat, coo_n, coo_i, coo_j));
  } else {
    PetscCall(MatSetPreallocationCOO_MPIAIJ(mat, coo_n, coo_i, coo_j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* creates the GPU memory */
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->A));
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->B));
    mpidev                   = static_cast<Mat_MPIAIJCUSPARSE *>(mpiaij->spptr);
    mpidev->use_extended_coo = PETSC_TRUE;

    PetscCallCUDA(cudaMalloc((void **)&mpidev->Ajmap1_d, (mpiaij->Annz + 1) * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Aperm1_d, mpiaij->Atot1 * sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void **)&mpidev->Bjmap1_d, (mpiaij->Bnnz + 1) * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Bperm1_d, mpiaij->Btot1 * sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void **)&mpidev->Aimap2_d, mpiaij->Annz2 * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Ajmap2_d, (mpiaij->Annz2 + 1) * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Aperm2_d, mpiaij->Atot2 * sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void **)&mpidev->Bimap2_d, mpiaij->Bnnz2 * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Bjmap2_d, (mpiaij->Bnnz2 + 1) * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->Bperm2_d, mpiaij->Btot2 * sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void **)&mpidev->Cperm1_d, mpiaij->sendlen * sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->sendbuf_d, mpiaij->sendlen * sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void **)&mpidev->recvbuf_d, mpiaij->recvlen * sizeof(PetscScalar)));

    PetscCallCUDA(cudaMemcpy(mpidev->Ajmap1_d, mpiaij->Ajmap1, (mpiaij->Annz + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Aperm1_d, mpiaij->Aperm1, mpiaij->Atot1 * sizeof(PetscCount), cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Bjmap1_d, mpiaij->Bjmap1, (mpiaij->Bnnz + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bperm1_d, mpiaij->Bperm1, mpiaij->Btot1 * sizeof(PetscCount), cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Aimap2_d, mpiaij->Aimap2, mpiaij->Annz2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Ajmap2_d, mpiaij->Ajmap2, (mpiaij->Annz2 + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Aperm2_d, mpiaij->Aperm2, mpiaij->Atot2 * sizeof(PetscCount), cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Bimap2_d, mpiaij->Bimap2, mpiaij->Bnnz2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bjmap2_d, mpiaij->Bjmap2, (mpiaij->Bnnz2 + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bperm2_d, mpiaij->Bperm2, mpiaij->Btot2 * sizeof(PetscCount), cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Cperm1_d, mpiaij->Cperm1, mpiaij->sendlen * sizeof(PetscCount), cudaMemcpyHostToDevice));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

__global__ static void MatPackCOOValues(const PetscScalar kv[], PetscCount nnz, const PetscCount perm[], PetscScalar buf[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < nnz; i += grid_size) buf[i] = kv[perm[i]];
}

__global__ static void MatAddLocalCOOValues(const PetscScalar kv[], InsertMode imode, PetscCount Annz, const PetscCount Ajmap1[], const PetscCount Aperm1[], PetscScalar Aa[], PetscCount Bnnz, const PetscCount Bjmap1[], const PetscCount Bperm1[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz + Bnnz; i += grid_size) {
    PetscScalar sum = 0.0;
    if (i < Annz) {
      for (PetscCount k = Ajmap1[i]; k < Ajmap1[i + 1]; k++) sum += kv[Aperm1[k]];
      Aa[i] = (imode == INSERT_VALUES ? 0.0 : Aa[i]) + sum;
    } else {
      i -= Annz;
      for (PetscCount k = Bjmap1[i]; k < Bjmap1[i + 1]; k++) sum += kv[Bperm1[k]];
      Ba[i] = (imode == INSERT_VALUES ? 0.0 : Ba[i]) + sum;
    }
  }
}

__global__ static void MatAddRemoteCOOValues(const PetscScalar kv[], PetscCount Annz2, const PetscCount Aimap2[], const PetscCount Ajmap2[], const PetscCount Aperm2[], PetscScalar Aa[], PetscCount Bnnz2, const PetscCount Bimap2[], const PetscCount Bjmap2[], const PetscCount Bperm2[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz2 + Bnnz2; i += grid_size) {
    if (i < Annz2) {
      for (PetscCount k = Ajmap2[i]; k < Ajmap2[i + 1]; k++) Aa[Aimap2[i]] += kv[Aperm2[k]];
    } else {
      i -= Annz2;
      for (PetscCount k = Bjmap2[i]; k < Bjmap2[i + 1]; k++) Ba[Bimap2[i]] += kv[Bperm2[k]];
    }
  }
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
  Mat_MPIAIJCUSPARSE *mpidev = static_cast<Mat_MPIAIJCUSPARSE *>(mpiaij->spptr);
  Mat                 A = mpiaij->A, B = mpiaij->B;
  PetscCount          Annz = mpiaij->Annz, Annz2 = mpiaij->Annz2, Bnnz = mpiaij->Bnnz, Bnnz2 = mpiaij->Bnnz2;
  PetscScalar        *Aa, *Ba = NULL;
  PetscScalar        *vsend = mpidev->sendbuf_d, *v2 = mpidev->recvbuf_d;
  const PetscScalar  *v1     = v;
  const PetscCount   *Ajmap1 = mpidev->Ajmap1_d, *Ajmap2 = mpidev->Ajmap2_d, *Aimap2 = mpidev->Aimap2_d;
  const PetscCount   *Bjmap1 = mpidev->Bjmap1_d, *Bjmap2 = mpidev->Bjmap2_d, *Bimap2 = mpidev->Bimap2_d;
  const PetscCount   *Aperm1 = mpidev->Aperm1_d, *Aperm2 = mpidev->Aperm2_d, *Bperm1 = mpidev->Bperm1_d, *Bperm2 = mpidev->Bperm2_d;
  const PetscCount   *Cperm1 = mpidev->Cperm1_d;
  PetscMemType        memtype;

  PetscFunctionBegin;
  if (mpidev->use_extended_coo) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));
    PetscCall(PetscGetMemType(v, &memtype));
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we need to copy it to device */
      PetscCallCUDA(cudaMalloc((void **)&v1, mpiaij->coo_n * sizeof(PetscScalar)));
      PetscCallCUDA(cudaMemcpy((void *)v1, v, mpiaij->coo_n * sizeof(PetscScalar), cudaMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) {
      PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(A, &Aa)); /* write matrix values */
      PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(B, &Ba));
    } else {
      PetscCall(MatSeqAIJCUSPARSEGetArray(A, &Aa)); /* read & write matrix values */
      PetscCall(MatSeqAIJCUSPARSEGetArray(B, &Ba));
    }

    /* Pack entries to be sent to remote */
    if (mpiaij->sendlen) {
      MatPackCOOValues<<<(mpiaij->sendlen + 255) / 256, 256>>>(v1, mpiaij->sendlen, Cperm1, vsend);
      PetscCallCUDA(cudaPeekAtLastError());
    }

    /* Send remote entries to their owner and overlap the communication with local computation */
    PetscCall(PetscSFReduceWithMemTypeBegin(mpiaij->coo_sf, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, vsend, PETSC_MEMTYPE_CUDA, v2, MPI_REPLACE));
    /* Add local entries to A and B */
    if (Annz + Bnnz > 0) {
      MatAddLocalCOOValues<<<(Annz + Bnnz + 255) / 256, 256>>>(v1, imode, Annz, Ajmap1, Aperm1, Aa, Bnnz, Bjmap1, Bperm1, Ba);
      PetscCallCUDA(cudaPeekAtLastError());
    }
    PetscCall(PetscSFReduceEnd(mpiaij->coo_sf, MPIU_SCALAR, vsend, v2, MPI_REPLACE));

    /* Add received remote entries to A and B */
    if (Annz2 + Bnnz2 > 0) {
      MatAddRemoteCOOValues<<<(Annz2 + Bnnz2 + 255) / 256, 256>>>(v2, Annz2, Aimap2, Ajmap2, Aperm2, Aa, Bnnz2, Bimap2, Bjmap2, Bperm2, Ba);
      PetscCallCUDA(cudaPeekAtLastError());
    }

    if (imode == INSERT_VALUES) {
      PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(A, &Aa));
      PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(B, &Ba));
    } else {
      PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &Aa));
      PetscCall(MatSeqAIJCUSPARSERestoreArray(B, &Ba));
    }
    if (PetscMemTypeHost(memtype)) PetscCallCUDA(cudaFree((void *)v1));
  } else {
    PetscCall(MatSetValuesCOO_MPIAIJCUSPARSE_Basic(mat, v, imode));
  }
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A, MatReuse scall, IS *glob, Mat *A_loc)
{
  Mat             Ad, Ao;
  const PetscInt *cmap;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &cmap));
  PetscCall(MatSeqAIJCUSPARSEMergeMats(Ad, Ao, scall, A_loc));
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    PetscCall(MatGetLocalSize(Ad, NULL, &dn));
    PetscCall(MatGetLocalSize(Ao, NULL, &on));
    PetscCall(MatGetOwnershipRangeColumn(A, &cst, NULL));
    PetscCall(PetscMalloc1(dn + on, &gidx));
    for (i = 0; i < dn; i++) gidx[i] = cst + i;
    for (i = 0; i < on; i++) gidx[i + dn] = cmap[i];
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad), dn + on, gidx, PETSC_OWN_POINTER, glob));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ *)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)b->spptr;
  PetscInt            i;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (PetscDefined(USE_DEBUG) && d_nnz) {
    for (i = 0; i < B->rmap->n; i++) PetscCheck(d_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, d_nnz[i]);
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i = 0; i < B->rmap->n; i++) PetscCheck(o_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, o_nnz[i]);
  }
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  PetscCall(MatDestroy(&b->B));
  if (!b->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
    PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
  }
  if (!b->B) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
    PetscCall(MatSetSizes(b->B, B->rmap->n, size > 1 ? B->cmap->N : 0, B->rmap->n, size > 1 ? B->cmap->N : 0));
  }
  PetscCall(MatSetType(b->A, MATSEQAIJCUSPARSE));
  PetscCall(MatSetType(b->B, MATSEQAIJCUSPARSE));
  PetscCall(MatBindToCPU(b->A, B->boundtocpu));
  PetscCall(MatBindToCPU(b->B, B->boundtocpu));
  PetscCall(MatSeqAIJSetPreallocation(b->A, d_nz, d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B, o_nz, o_nnz));
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusparseStruct->offdiagGPUMatFormat));
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->mult)(a->A, xx, yy));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, yy, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscCall(MatZeroEntries(l->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->multadd)(a->A, xx, yy, zz));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, zz, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall((*a->B->ops->multtranspose)(a->B, xx, a->lvec));
  PetscCall((*a->A->ops->multtranspose)(a->A, xx, yy));
  PetscCall(VecScatterBegin(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A, MatCUSPARSEFormatOperation op, MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT_DIAG:
    cusparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_MULT_OFFDIAG:
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseStruct->diagGPUMatFormat    = format;
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(Mat A, PetscOptionItems *PetscOptionsObject)
{
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_MPIAIJ              *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE      *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPIAIJCUSPARSE options");
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format", "sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_DIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format", "sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->offdiagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_OFFDIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_storage_format", "sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_ALL, format));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A, MatAssemblyType mode)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A, mode));
  if (mpiaij->lvec) PetscCall(VecSetType(mpiaij->lvec, VECSEQCUDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)aij->spptr;

  PetscFunctionBegin;
  PetscCheck(cusparseStruct, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing spptr");
  /* Free COO */
  PetscCall(MatResetPreallocationCOO_MPIAIJCUSPARSE(A));
  PetscCallCXX(delete cusparseStruct);
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUSPARSESetFormat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijcusparse_hypre_C", NULL));
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* defines MatSetValues_MPICUSPARSE_Hash() */
#define TYPE AIJ
#define TYPE_AIJ
#define SUB_TYPE_CUSPARSE
#include "../src/mat/impls/aij/mpi/mpihashmat.h"
#undef TYPE
#undef TYPE_AIJ
#undef SUB_TYPE_CUSPARSE

static PetscErrorCode MatSetUp_MPI_HASH_CUSPARSE(Mat A)
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)b->spptr;

  PetscFunctionBegin;
  PetscCall(MatSetUp_MPI_Hash(A));
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusparseStruct->offdiagGPUMatFormat));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_MPIAIJCUSPARSE(Mat A, MatDuplicateOption dupOption, Mat *B)
{
  Mat_MPIAIJ         *Adata = static_cast<Mat_MPIAIJ *>(A->data), *Bdata;
  Mat_MPIAIJCUSPARSE *Adev  = static_cast<Mat_MPIAIJCUSPARSE *>(Adata->spptr);
  Mat                 mat;

  PetscFunctionBegin;
  PetscCall(MatDuplicate_MPIAIJ(A, dupOption, B));
  mat   = *B;
  Bdata = static_cast<Mat_MPIAIJ *>(mat->data);
  PetscCallCXX(Bdata->spptr = new Mat_MPIAIJCUSPARSE(*Adev)); // use the shallow copy ctor to copy A's coo info on device
  // matrix defaultvectype was handled by MatDuplicate()
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, MATMPIAIJCUSPARSE));
  PetscCall(MatSetOps_MPIAIJCUSPARSE(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOps_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;
  A->ops->setup                 = MatSetUp_MPI_HASH_CUSPARSE;
  A->ops->duplicate             = MatDuplicate_MPIAIJCUSPARSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUSPARSESetFormat_C", MatCUSPARSESetFormat_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijcusparse_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType, MatReuse reuse, Mat *newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B, *newmat, SAME_NONZERO_PATTERN));
  A             = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &A->defaultvectype));

  a = (Mat_MPIAIJ *)A->data;
  if (a->A) PetscCall(MatSetType(a->A, MATSEQAIJCUSPARSE));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQAIJCUSPARSE));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQCUDA));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) PetscCallCXX(a->spptr = new Mat_MPIAIJCUSPARSE);
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJCUSPARSE));
  PetscCall(MatSetOps_MPIAIJCUSPARSE(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJCUSPARSE(A, MATMPIAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCreateAIJCUSPARSE - Creates a sparse matrix in `MATAIJCUSPARSE` (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to NVIDIA GPUs and use the CuSPARSE library for calculations.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to `PETSC_COMM_SELF`
.  m - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if `N` is given) For square matrices `n` is almost always `m`.
.  M - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
.  N - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or `NULL`, if `d_nz` is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e `m`.
           For matrices you plan to factor you must leave room for the diagonal entry and
           put in the entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or `NULL`, if `o_nz` is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e `m`.

   Output Parameter:
.  A - the matrix

   Level: intermediate

   Notes:
   It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

   The AIJ format, also called the
   compressed row storage), is fully compatible with standard Fortran
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.

.seealso: [](chapter_matrices), `Mat`, `MATAIJCUSPARSE`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`, `MATMPIAIJCUSPARSE`, `MATAIJCUSPARSE`
@*/
PetscErrorCode MatCreateAIJCUSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPIAIJCUSPARSE));
    PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQAIJCUSPARSE));
    PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATAIJCUSPARSE - A matrix type to be used for sparse matrices; it is as same as `MATMPIAIJCUSPARSE`.

   A matrix type type whose data resides on NVIDIA GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on NVIDIA GPUs using the CuSPARSE library.

   This matrix type is identical to `MATSEQAIJCUSPARSE` when constructed with a single process communicator,
   and `MATMPIAIJCUSPARSE` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijcusparse - sets the matrix type to `MATMPIAIJCUSPARSE`
.  -mat_cusparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices. Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MatCreateAIJCUSPARSE()`, `MATSEQAIJCUSPARSE`, `MATMPIAIJCUSPARSE`, `MatCreateSeqAIJCUSPARSE()`, `MatCUSPARSESetFormat()`, `MatCUSPARSEStorageFormat`, `MatCUSPARSEFormatOperation`
M*/

/*MC
   MATMPIAIJCUSPARSE - A matrix type to be used for sparse matrices; it is as same as `MATAIJCUSPARSE`.

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATAIJCUSPARSE`, `MATSEQAIJCUSPARSE`
M*/
