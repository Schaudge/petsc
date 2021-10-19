#if !defined(_PETSCGINKGO_H)
#define _PETSCGINKGO_H

#include <ginkgo.hpp>
#include <petsc/private/matimpl.h>
#include <petscmatginkgo.h>

using ValueType = PetscGinkgoScalar;
using IndexType = PetscGinkgoInt;

using vec = gko::matrix::Dense<ValueType>;
using parr = gko::Array<IndexType>;
using Csr = gko::matrix::Csr<ValueType, IndexType>;
using Dense = gko::matrix::Dense<ValueType>;


typedef struct {
  /* std::shared_ptr<gko::Executor> exec; */
  /* std::shared_ptr<gko::LinOp> mat; */
  std::shared_ptr<Csr> A_csr;
  std::shared_ptr<vec> b;
  std::shared_ptr<vec> x;
} Mat_GinkgoCSR;

typedef struct {
  std::shared_ptr<gko::Executor> exec;
  std::shared_ptr<Dense> A_dense;
  std::shared_ptr<vec> b;
  std::shared_ptr<vec> x;
} Mat_GinkgoDense;

#endif
