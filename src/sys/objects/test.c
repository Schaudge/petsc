
#include <petsc/private/petscimpl.h>        /*I    "petscsys.h"   I*/
#include <petsc/private/snesimpl.h>

PetscErrorCode PetscPointerGetPetscHeaderType_Private(const void *ptr,size_t bytes,PetscHeaderType *type)
{
  PetscObject hdr = NULL;
  PetscFunctionBegin;
  *type = PETSC_HEADER_VALID;
  if (!ptr) { /* "Null Object: Parameter # %d" */
    if (type) *type = PETSC_HEADER_NULL_PTR;
    PetscFunctionReturn(0);
  }
  if (!PetscCheckPointer(ptr,PETSC_OBJECT)) { /* "Invalid Pointer to Object: Parameter # %d" */
    if (type) *type = PETSC_HEADER_INVALID_PTR;
    PetscFunctionReturn(0);
  }

  if (bytes < sizeof(_p_PetscObject)) {
    if (type) *type = PETSC_HEADER_INVALID_PTR;
    PetscFunctionReturn(0);
  }

  /* check if it is safe to cast to PetscObject and dereference to access classid */
  {
    _p_PetscObject rawhdr;
    size_t         byte_offset = 0;

    /* we do not have to assume classid is the first member of the struct */
    byte_offset = (char*)&rawhdr.classid - (char*)&rawhdr;
    if (!PetscCheckPointer((void*)((char*)ptr + byte_offset),PETSC_INT)) {
      if (type) *type = PETSC_HEADER_INVALID_PTR;
      PetscFunctionReturn(0);
    }
  }
  hdr = (PetscObject)ptr;
  if (!PetscCheckPointer((void*)hdr->class_name,PETSC_STRING)) {
    if (type) *type = PETSC_HEADER_INVALID_PTR;
    PetscFunctionReturn(0);
  }
  /* check for garbage in first and last member */
  if (!PetscCheckPointer((void*)&hdr->donotPetscObjectPrintClassNamePrefixType,PETSC_BOOL)) {
    if (type) *type = PETSC_HEADER_INVALID_PTR;
    PetscFunctionReturn(0);
  }
  if ((hdr->donotPetscObjectPrintClassNamePrefixType != PETSC_TRUE) && (hdr->donotPetscObjectPrintClassNamePrefixType != PETSC_FALSE)) {
    if (type) *type = PETSC_HEADER_INVALID_PTR;
    PetscFunctionReturn(0);
  }
  if (((PetscObject)(ptr))->classid == PETSCFREEDHEADER){ /* "Object already free: Parameter # %d" */
    if (type) *type = PETSC_HEADER_FREED;
    PetscFunctionReturn(0);
  }
  /* check class_name, classid and comm are defined */
  if (!hdr->class_name) {
    if (type) *type = PETSC_HEADER_INVALID;
  }
  if (hdr->classid < PETSC_SMALLEST_CLASSID
             || hdr->classid > PETSC_LARGEST_CLASSID) { /* "Invalid type of object: Parameter # %d" */
    if (type) *type = PETSC_HEADER_INVALID;
  }
  if (!hdr->class_name) {
    if (type) *type = PETSC_HEADER_INVALID;
  }
  if (!((int)(hdr->comm))) { /* this should be valid for both typedef int MPI_Comm (MPICH) and typedef struct comm_info* MPI_Comm (OpenMPI) */
    if (type) *type = PETSC_HEADER_INVALID;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscPointerTestPetscObject - Test if a pointer references a PetscObject

   Not collective

   Input Parameter:
.  ptr - A C pointer

   Output Parameter:
.  flg - boolean indicating if the pointer references a PetscObject (or a derived object)

   Concepts: PetscObject query

.seealso: PetscPointerGetPetscHeaderType()
@*/
PetscErrorCode PetscPointerTestPetscObject(const void *ptr,PetscBool *flg)
{
  PetscHeaderType   type;
  PetscBool         is_petsc_object = PETSC_FALSE;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  ierr = PetscPointerGetPetscHeaderType(ptr,&type);CHKERRQ(ierr);
  if (type == PETSC_HEADER_VALID) {
    is_petsc_object = PETSC_TRUE;
  }
  if (flg) *flg = is_petsc_object;
  PetscFunctionReturn(0);
}
