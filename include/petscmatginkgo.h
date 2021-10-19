#if !defined(PETSCMATGINKGO_H)
#define PETSCMATGINKGO_H

#include <petscmat.h>

#if defined(PETSC_HAVE_GINKGO) && defined(__cplusplus)
#include <ginkgo.hpp>
#if defined(PETSC_USE_COMPLEX)
typedef std::complex<double> PetscGinkgoScalar;

#else
typedef double PetscGinkgoScalar;
#endif
#endif
/*TODO Need to understand ginkgo's long int options */
typedef int PetscGinkgoInt;

#endif /* PETSCMATGINKGO_H */
