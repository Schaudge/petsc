#ifndef lint
static char vcid[] = "$Id: stride.c,v 1.31 1995/10/06 22:23:12 bsmith Exp curfman $";
#endif
/*
       Index sets of evenly space integers, defined by a 
    start, stride and length.
*/
#include "isimpl.h"             /*I   "is.h"   I*/
#include "pinclude/pviewer.h"

typedef struct {
  int n,first,step;
} IS_Stride;

/*@
   ISStrideGetInfo - Returns the first index in a stride index set and 
   the stride width.

   Input Parameter:
.  is - the index set

   Output Parameters:
.  first - the first index
.  step - the stride width

   Notes:
   Returns info on stride index set. This is a pseudo-public function that
   should not be needed by most users.

.keywords: IS, index set, stride, get, information

.seealso: ISCreateStrideSeq()
@*/
int ISStrideGetInfo(IS is,int *first,int *step)
{
  IS_Stride *sub = (IS_Stride *) is->data;
  if (is->type != IS_STRIDE_SEQ) return 0;
  *first = sub->first; *step = sub->step;
  return 1;
}

static int ISDestroy_Stride(PetscObject obj)
{
  IS is = (IS) obj;
  PETSCFREE(is->data); 
  PLogObjectDestroy(is);
  PETSCHEADERDESTROY(is); return 0;
}

static int ISGetIndices_Stride(IS in,int **idx)
{
  IS_Stride *sub = (IS_Stride *) in->data;
  int       i;

  if (sub->n) {
    *idx = (int *) PETSCMALLOC(sub->n*sizeof(int)); CHKPTRQ(idx);
    (*idx)[0] = sub->first;
    for ( i=1; i<sub->n; i++ ) (*idx)[i] = (*idx)[i-1] + sub->step;
  }
  else *idx = 0;
  return 0;
}

static int ISRestoreIndices_Stride(IS in,int **idx)
{
  if (*idx) PETSCFREE(*idx);
  return 0;
}

static int ISGetSize_Stride(IS is,int *size)
{
  IS_Stride *sub = (IS_Stride *)is->data;
  *size = sub->n; return 0;
}

static int ISView_Stride(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_Stride   *sub = (IS_Stride *)is->data;
  int         i,n = sub->n,ierr;
  PetscObject vobj = (PetscObject) viewer;
  FILE        *fd;

  if (!viewer) {
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == VIEWER_COOKIE) {
    if ((vobj->type == ASCII_FILE_VIEWER) || (vobj->type == ASCII_FILES_VIEWER)){
      ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
      if (is->isperm) {
        fprintf(fd,"Index set is permutation\n");
      }
      fprintf(fd,"Number of indices in set %d\n",n);
      for ( i=0; i<n; i++ ) {
        fprintf(fd,"%d %d\n",i,sub->first + i*sub->step);
      }
    }
  }
  return 0;
}
  
static struct _ISOps myops = { ISGetSize_Stride,
                               ISGetSize_Stride,
                               ISGetIndices_Stride,
                               ISRestoreIndices_Stride,0};
/*@C
   ISCreateStrideSeq - Creates a data structure for an index set 
   containing a list of evenly spaced integers.

   Input Parameters:
.  comm - the MPI communicator
.  n - the length of the index set
.  first - the first element of the index set
.  step - the change to the next index

   Output Parameter:
.  is - the location to stash the index set

.keywords: IS, index set, create, stride, sequential

.seealso: ISCreateSeq()
@*/
int ISCreateStrideSeq(MPI_Comm comm,int n,int first,int step,IS *is)
{
  int       min, max;
  IS        Nindex;
  IS_Stride *sub;

  *is = 0;
   if (n < 0) SETERRQ(1,"ISCreateStrideSeq:Number of indices < 0");
  if (step == 0) SETERRQ(1,"ISCreateStrideSeq:Step must be nonzero");

  PETSCHEADERCREATE(Nindex, _IS,IS_COOKIE,IS_STRIDE_SEQ,comm); 
  PLogObjectCreate(Nindex);
  PLogObjectMemory(Nindex,sizeof(IS_Stride) + sizeof(struct _IS));
  sub            = (IS_Stride *) PETSCMALLOC(sizeof(IS_Stride)); CHKPTRQ(sub);
  sub->n         = n;
  sub->first     = first;
  sub->step      = step;
  if (step > 0) {min = first; max = first + step*(n-1);}
  else          {max = first; min = first + step*(n-1);}

  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(&Nindex->ops,&myops,sizeof(myops));
  Nindex->destroy = ISDestroy_Stride;
  Nindex->view    = ISView_Stride;
  Nindex->isperm  = 0;
  *is = Nindex; 
  return 0;
}

