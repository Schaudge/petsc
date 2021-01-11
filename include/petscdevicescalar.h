
#include <petscmath.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
     Defines a PetscScalar indirectly as a pointer to an address.

     Regular arithematic and & can be used on the PetscScalar as if it were a regular variable.

     By changing the methods below and the malloc() the pointer can be an address on a GPU.

*/
class PetscDeviceScalar {
public:
  double *value;

  PetscDeviceScalar() {
    this->value = (PetscScalar*) malloc(sizeof(PetscScalar));
    printf("creating %p\n",this->value);
  }

  PetscDeviceScalar(double a) {
    this->value = (double*) malloc(sizeof(double));
    printf("creating with double value %p\n",this->value);
    *(this->value) = a;
  }

  PetscDeviceScalar(const PetscDeviceScalar & a) {
    this->value = (double*) malloc(sizeof(double));
    printf("creating with PetscDeviceScalar value %p\n",this->value);
    *(this->value) = *a.value;
  }

  ~PetscDeviceScalar() {
    printf("freeing %p\n",this->value);
    if (this->value) free(this->value);
  }

  PetscDeviceScalar & operator=(const double & other) {
    *(this->value) = other;
    return *this;
  }

  PetscDeviceScalar & operator=(double & other) {
    *(this->value) = other;
    return *this;
  }

  PetscDeviceScalar & operator=(const PetscDeviceScalar & other) {
    printf("direct assignment from other %p\n",other.value);
    *(this->value) = *(other.value);
    return *this;
  }

  // This may not make sense in the context of GPUs
  operator PetscScalar() {
    return *(this->value);
  }

  PetscDeviceScalar & operator+=(const PetscDeviceScalar & a) {
    *(this->value) += *(a.value);
    return *this;
  }

  PetscDeviceScalar & operator-=(const PetscDeviceScalar & a) {
    *(this->value) -= *(a.value);
    return *this;
  }

  PetscDeviceScalar & operator*=(const PetscDeviceScalar & a) {
    *(this->value) *= *(a.value);
    return *this;
  }

  PetscDeviceScalar & operator/=(const PetscDeviceScalar & a) {
    *(this->value) /= *(a.value);
    return *this;
  }

  bool operator==(const PetscDeviceScalar & a) {
    return (*(this->value) == *(a.value));
  }

  bool operator==(const PetscScalar & a) {
    return (*(this->value) == a);
  }

  bool operator==(const int & a) {
    return (*(this->value) == a);
  }

#if !defined(PETSC_USE_COMPLEX)
  bool operator<(const PetscDeviceScalar & a) {
    return (*(this->value) < *(a.value));
  }

  bool operator<(const PetscScalar & a) {
    return (*(this->value) < a);
  }
#endif

  PetscScalar * operator&() {
    return this->value;
  }

  friend PetscDeviceScalar operator+(const PetscDeviceScalar &a,const PetscDeviceScalar &b)
  {
    PetscDeviceScalar c = a;
    c += b;
    return c;
  }

  friend PetscDeviceScalar operator-(PetscDeviceScalar &a,const PetscDeviceScalar &b)
  {
    PetscDeviceScalar c = a;
    c -= b;
    return c;
  }

  friend PetscDeviceScalar operator*(PetscDeviceScalar &a,const PetscDeviceScalar &b)
  {
    PetscDeviceScalar c = a;
    c *= b;
    return c;
  }

  friend PetscDeviceScalar operator/(PetscDeviceScalar &a,const PetscDeviceScalar &b)
  {
    PetscDeviceScalar c = a;
    c /= b;
    return c;
  }
};

