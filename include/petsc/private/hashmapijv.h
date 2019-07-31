#if !defined(_PETSC_HASHMAPIJV_H)
#define _PETSC_HASHMAPIJV_H

#include <petsc/private/hashmap.h>

#if !defined(PETSC_HASHIJKEY)
#define PETSC_HASHIJKEY
typedef struct _PetscHashIJKey { PetscInt i, j; } PetscHashIJKey;
#define PetscHashIJKeyHash(key) PetscHashCombine(PetscHashInt((key).i),PetscHashInt((key).j))
#define PetscHashIJKeyEqual(k1,k2) (((k1).i == (k2).i) ? ((k1).j == (k2).j) : 0)
#endif

PETSC_HASH_MAP(HMapIJV, PetscHashIJKey, PetscScalar, PetscHashIJKeyHash, PetscHashIJKeyEqual, -1)


/*MC
  PetscHMapIVAddValue - Add value to the value of a given key if the key exists,
  otherwise, insert a new (key,value) entry in the hash table

  Synopsis:
  #include <petsc/private/hashmapiv.h>
  PetscErrorCode PetscHMapIVAddValue(PetscHMapT ht,KeyType key,ValType val)

  Input Parameters:
+ ht  - The hash table
. key - The key
- val - The value

  Level: developer

.seealso: PetscHMapTGet(), PetscHMapTIterSet(), PetscHMapIVSet()
M*/
PETSC_STATIC_INLINE
PetscErrorCode PetscHMapIJVAddValue(PetscHMapIJV ht,PetscHashIJKey key,PetscScalar val)
{
  int      ret;
  khiter_t iter;
  PetscFunctionBeginHot;
  PetscValidPointer(ht,1);
  iter = kh_put(HMapIJV,ht,key,&ret);
  PetscHashAssert(ret>=0);
  if (ret) kh_val(ht,iter) = val;
  else  kh_val(ht,iter) += val;
  PetscFunctionReturn(0);
}

#endif /* PETSC_HASHMAPIJV_H */
