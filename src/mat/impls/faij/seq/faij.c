
/*
    Defines the basic matrix operations for the FAIJ (compressed row)
  matrix storage format.
*/
#include <../src/mat/impls/faij/seq/faij.h>  /*I   "petscmat.h"  I*/


PetscErrorCode MatMarkDiagonal_SeqFAIJ(Mat A)
{
  Mat_SeqFAIJ    *a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = a->mbs;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr         = PetscMalloc1(m,&a->diag);CHKERRQ(ierr);
    ierr         = PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt));CHKERRQ(ierr);
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<m; i++) {
    a->diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqFAIJ(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_SeqFAIJ    *a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *diag,*ii = a->i,i;

  PetscFunctionBegin;
  ierr     = MatMarkDiagonal_SeqFAIJ(A);CHKERRQ(ierr);
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !ii) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    ierr = PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n");CHKERRQ(ierr);
  } else {
    diag = a->diag;
    for (i=0; i<a->mbs; i++) {
      if (diag[i] >= ii[i+1]) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        ierr = PetscInfo1(A,"Matrix is missing block diagonal number %D\n",i);CHKERRQ(ierr);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqFAIJ(Mat A)
{
  Mat_SeqFAIJ    *a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D",A->rmap->N,A->cmap->n,a->nz);
#endif
  ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
  ierr = ISDestroy(&a->row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr);
  if (a->free_diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  ierr = PetscFree(a->idiag);CHKERRQ(ierr);
  if (a->free_imax_ilen) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = PetscFree(a->mult_work);CHKERRQ(ierr);
  ierr = PetscFree(a->sor_workt);CHKERRQ(ierr);
  ierr = PetscFree(a->sor_work);CHKERRQ(ierr);
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);
  ierr = PetscFree2(a->compressedrow.i,a->compressedrow.rindex);CHKERRQ(ierr);

  ierr = MatDestroy(&a->parent);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqFAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqFAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqFAIJ    *a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused = (flg ? -1 : 0);
    break;
  case MAT_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_SUBMAT_SINGLEIS:
  case MAT_STRUCTURE_ONLY:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqFAIJ_ASCII_structonly(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat_SeqFAIJ    *a = (Mat_SeqFAIJ*)A->data;
  PetscInt       i,bs = A->rmap->bs,k;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0; i<a->mbs; i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"row %D-%D:",i*bs,i*bs+bs-1);CHKERRQ(ierr);
    for (k=a->i[i]; k<a->i[i+1]; k++) {
      ierr = PetscViewerASCIIPrintf(viewer," (%D-%D) ",bs*a->j[k],bs*a->j[k]+bs-1);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqFAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqFAIJ       *a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,bs = A->rmap->bs,k;
  PetscViewerFormat format;
  PetscScalar       *value = a->a;

  PetscFunctionBegin;
  if (A->structure_only) {
    ierr = MatView_SeqFAIJ_ASCII_structonly(A,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size is %D\n",bs);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    const char *matname;
    Mat        aij;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)A,&matname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)aij,matname);CHKERRQ(ierr);
    ierr = MatView(aij,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&aij);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          ierr = PetscViewerASCIIPrintf(viewer," %D %g ",bs*a->j[k]+j,value[bs*k+j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqFAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_SeqFAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Note that this is different than for the BAIJ matrix since the block is the diagonal only of the block */
PetscErrorCode MatSetValuesBlocked_SeqFAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqFAIJ       *a = (Mat_SeqFAIJ*)A->data;
  PetscInt          *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscErrorCode    ierr;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs=A->rmap->bs;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap=NULL,*aa = a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (row >= a->mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block row index too large %D max %D",row,a->mbs-1);
#endif
    rp   = aj + ai[row];
    if (!A->structure_only) ap = aa + bs*ai[row];
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (in[l] >= a->nbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block column index too large %D max %D",in[l],a->nbs-1);
#endif
      col = in[l];
      if (!A->structure_only) {
        if (roworiented) {
          value = v + bs*(k*n + l);
        } else {
          value = v + bs*(l*m + k);
        }
      }
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (A->structure_only) goto noinsert2;
          bap = ap +  bs*i;
          if (is == ADD_VALUES) {
            for (ii=0; ii<bs; ii++) {
              bap[ii] += value[ii];
            }
          } else {
            for (ii=0; ii<bs; ii++) {
              bap[ii]  = value[ii];
            }
          }
          goto noinsert2;
        }
      }
      if (nonew == 1) goto noinsert2;
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new blocked index new nonzero block (%D, %D) in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A,a->mbs,bs,nrow,row,col,rmax,ai,aj,rp,imax,nonew,MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      }
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        if (!A->structure_only) {
          ierr = PetscMemcpy(ap+bs*(ii+1),ap+bs*(ii),bs*sizeof(MatScalar));CHKERRQ(ierr);
        }
      }
      if (N >= i && !A->structure_only) {
        ierr = PetscMemzero(ap+bs*i,bs*sizeof(MatScalar));CHKERRQ(ierr);
      }

      rp[i] = col;
      if (!A->structure_only) {
        bap   = ap +  bs*i;
        for (ii=0; ii<bs; ii++) {
          bap[ii] = *value++;
        }
      }
noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqFAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqFAIJ    *a     = (Mat_SeqFAIJ*)A->data;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m      = A->rmap->N,*ip,N,*ailen = a->ilen;
  PetscErrorCode ierr;
  PetscInt       mbs  = a->mbs,bs2 = a->bs2,rmax = 0;
  MatScalar      *aa  = a->a,*ap;
  PetscReal      ratio=0.6;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i]; ap = aa + bs2*ai[i];
      N  = ailen[i];
      for (j=0; j<N; j++) {
        ip[j-fshift] = ip[j];
        if (!A->structure_only) {
          ierr = PetscMemcpy(ap+(j-fshift)*bs2,ap+j*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
        }
      }
    }
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
    ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }

  /* reset ilen and imax for each row */
  a->nonzerorowcnt = 0;
  if (A->structure_only) {
    ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);
  } else { /* !A->structure_only */
    for (i=0; i<mbs; i++) {
      ailen[i] = imax[i] = ai[i+1] - ai[i];
      a->nonzerorowcnt += ((ai[i+1] - ai[i]) > 0);
    }
  }
  a->nz = ai[mbs];

  /* diagonals may have moved, so kill the diagonal pointers */
  a->idiagvalid = PETSC_FALSE;
  if (fshift && a->diag) {
    ierr    = PetscFree(a->diag);CHKERRQ(ierr);
    ierr    = PetscLogObjectMemory((PetscObject)A,-(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    a->diag = 0;
  }
  if (fshift && a->nounused == -1) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Unused space detected in matrix: %D X %D block size %D, %D unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);
  ierr = PetscInfo5(A,"Matrix size: %D X %D, block size %D; storage space: %D unneeded, %D used\n",m,A->cmap->n,A->rmap->bs,fshift*bs2,a->nz*bs2);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Most nonzeros blocks in any row is %D\n",rmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (PetscReal)fshift*bs2;
  a->rmax             = rmax;

  if (!A->structure_only) {
    ierr = MatCheckCompressedRow(A,a->nonzerorowcnt,&a->compressedrow,a->i,mbs,ratio);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   This function returns an array of flags which indicate the locations of contiguous
   blocks that should be zeroed. for eg: if bs = 3  and is = [0,1,2,3,5,6,7,8,9]
   then the resulting sizes = [3,1,1,3,1] correspondig to sets [(0,1,2),(3),(5),(6,7,8),(9)]
   Assume: sizes should be long enough to hold all the values.
*/
static PetscErrorCode MatZeroRows_SeqFAIJ_Check_Blocks(PetscInt idx[],PetscInt n,PetscInt bs,PetscInt sizes[], PetscInt *bs_max)
{
  PetscInt  i,j,k,row;
  PetscBool flg;

  PetscFunctionBegin;
  for (i=0,j=0; i<n; j++) {
    row = idx[i];
    if (row%bs!=0) { /* Not the begining of a block */
      sizes[j] = 1;
      i++;
    } else if (i+bs > n) { /* complete block doesn't exist (at idx end) */
      sizes[j] = 1;         /* Also makes sure atleast 'bs' values exist for next else */
      i++;
    } else { /* Begining of the block, so check if the complete block exists */
      flg = PETSC_TRUE;
      for (k=1; k<bs; k++) {
        if (row+k != idx[i+k]) { /* break in the block */
          flg = PETSC_FALSE;
          break;
        }
      }
      if (flg) { /* No break in the bs */
        sizes[j] = bs;
        i       += bs;
      } else {
        sizes[j] = 1;
        i++;
      }
    }
  }
  *bs_max = j;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRows_SeqFAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqFAIJ       *faij=(Mat_SeqFAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,count,*rows;
  PetscInt          bs=A->rmap->bs,bs2=faij->bs2,*sizes,row,bs_max;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<is_n; i++) {
      bb[is_idx[i]] = diag*xx[is_idx[i]];
    }
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* Make a copy of the IS and  sort it */
  /* allocate memory for rows,sizes */
  ierr = PetscMalloc2(is_n,&rows,2*is_n,&sizes);CHKERRQ(ierr);

  /* copy IS values to rows, and sort them */
  for (i=0; i<is_n; i++) rows[i] = is_idx[i];
  ierr = PetscSortInt(is_n,rows);CHKERRQ(ierr);

  if (faij->keepnonzeropattern) {
    for (i=0; i<is_n; i++) sizes[i] = 1;
    bs_max          = is_n;
  } else {
    ierr = MatZeroRows_SeqFAIJ_Check_Blocks(rows,is_n,bs,sizes,&bs_max);CHKERRQ(ierr);
    A->nonzerostate++;
  }

  for (i=0,j=0; i<bs_max; j+=sizes[i],i++) {
    row = rows[j];
    if (row < 0 || row > A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range",row);
    count = (faij->i[row/bs +1] - faij->i[row/bs])*bs;
    aa    = ((MatScalar*)(faij->a)) + faij->i[row/bs]*bs2 + (row%bs);
    if (sizes[i] == bs && !faij->keepnonzeropattern) {
      if (diag != (PetscScalar)0.0) {
        if (faij->ilen[row/bs] > 0) {
          faij->ilen[row/bs]       = 1;
          faij->j[faij->i[row/bs]] = row/bs;

          ierr = PetscMemzero(aa,count*bs*sizeof(MatScalar));CHKERRQ(ierr);
        }
        /* Now insert all the diagonal values for this bs */
        for (k=0; k<bs; k++) {
          ierr = (*A->ops->setvalues)(A,1,rows+j+k,1,rows+j+k,&diag,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else { /* (diag == 0.0) */
        faij->ilen[row/bs] = 0;
      } /* end (diag == 0.0) */
    } else { /* (sizes[i] != bs) */
#if defined(PETSC_USE_DEBUG)
      if (sizes[i] != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal Error. Value should be 1");
#endif
      for (k=0; k<count; k++) {
        aa[0] =  zero;
        aa   += bs;
      }
      if (diag != (PetscScalar)0.0) {
        ierr = (*A->ops->setvalues)(A,1,rows+j,1,rows+j,&diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree2(rows,sizes);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqFAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqFAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqFAIJ       *faij=(Mat_SeqFAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,count;
  PetscInt          bs   =A->rmap->bs,bs2=faij->bs2,row,col;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;
  PetscBool         *zeroed,vecs = PETSC_FALSE;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    vecs = PETSC_TRUE;
  }

  /* zero the columns */
  ierr = PetscCalloc1(A->rmap->n,&zeroed);CHKERRQ(ierr);
  for (i=0; i<is_n; i++) {
    if (is_idx[i] < 0 || is_idx[i] >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range",is_idx[i]);
    zeroed[is_idx[i]] = PETSC_TRUE;
  }
  for (i=0; i<A->rmap->N; i++) {
    if (!zeroed[i]) {
      row = i/bs;
      for (j=faij->i[row]; j<faij->i[row+1]; j++) {
        for (k=0; k<bs; k++) {
          col = bs*faij->j[j] + k;
          if (zeroed[col]) {
            aa = ((MatScalar*)(faij->a)) + j*bs2 + (i%bs) + bs*k;
            if (vecs) bb[i] -= aa[0]*xx[col];
            aa[0] = 0.0;
          }
        }
      }
    } else if (vecs) bb[i] = diag*xx[i];
  }
  ierr = PetscFree(zeroed);CHKERRQ(ierr);
  if (vecs) {
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* zero the rows */
  for (i=0; i<is_n; i++) {
    row   = is_idx[i];
    count = (faij->i[row/bs +1] - faij->i[row/bs])*bs;
    aa    = ((MatScalar*)(faij->a)) + faij->i[row/bs]*bs2 + (row%bs);
    for (k=0; k<count; k++) {
      aa[0] =  zero;
      aa   += bs;
    }
    if (diag != (PetscScalar)0.0) {
      ierr = (*A->ops->setvalues)(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd_SeqFAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatSetUp_SeqFAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqFAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqFAIJGetArray_SeqFAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqFAIJ *a = (Mat_SeqFAIJ*)A->data;

  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqFAIJRestoreArray_SeqFAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_SeqFAIJ(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqFAIJ        *aij = (Mat_SeqFAIJ*)A->data;

  PetscFunctionBegin;
  ierr = PetscMemzero(aij->a,(A->rmap->bs*aij->i[A->rmap->n])*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>
PetscErrorCode MatScale_SeqFAIJ(Mat A,PetscScalar a)
{
  PetscErrorCode     ierr;
  Mat_SeqFAIJ        *aij = (Mat_SeqFAIJ*)A->data;
  PetscBLASInt       bnz;
  const PetscScalar  oa = a;
  const PetscBLASInt one = 1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->bs*aij->nz,&bnz);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASscal",BLASscal_(&bnz,&oa,aij->a,&one));
  ierr = PetscLogFlops(aij->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqFAIJ(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqFAIJ     *aij = (Mat_SeqFAIJ*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !aij->nz) {
    ierr = MatSeqFAIJSetPreallocation(Y,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqFAIJ_N(Mat A,Vec xx,Vec zz)
{
  Mat_SeqFAIJ       *a = (Mat_SeqFAIJ*)A->data;
  PetscScalar       *z,*zarray;
  const PetscScalar *x,*xarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,bs=A->rmap->bs,j,l;
  const PetscInt    *idx,*ii;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  mbs = a->mbs;
  ii  = a->i;
  z   = zarray;

  for (i=0; i<mbs; i++) {
    for (j=ii[i]; j<ii[i+1]; j++) {  /* could copy all the needed x[] for a row into contiqous workspace and fuse the j and i loops into one */
      x = xarray + bs*(*idx++);
      for (l=0; l<bs; l++) {
        z[l] += v[l]*x[l];
      }
      v += bs;
    }
    z += bs;
  }
  ierr = VecRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(3.0*ii[mbs]*bs - mbs*bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {0,
                                       0,
                                       0,
                                       MatMult_SeqFAIJ_N,
                               /* 4*/  0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 10*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                              /*  25*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 20*/ 0,
                                       MatAssemblyEnd_SeqFAIJ,
                                       MatSetOption_SeqFAIJ,
                                       MatZeroEntries_SeqFAIJ,
                               /* 24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 29*/ MatSetUp_SeqFAIJ,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 34*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 39*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 44*/ 0,
                                       MatScale_SeqFAIJ,
                                       MatShift_SeqFAIJ,
                                       0,
                                       0,
                               /* 49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 54*/ 0,
                                       0,
                                       0,
                                       0,
                                       MatSetValuesBlocked_SeqFAIJ,
                               /* 59*/ 0,
                                       MatDestroy_SeqFAIJ,
                                       MatView_SeqFAIJ,
                                       0,
                                       0,
                               /* 64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 69*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 74*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 79*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 84*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 89*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 94*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 99*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*104*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*109*/ 0,
                                       0,
                                       0,
                                       0,
                                       MatMissingDiagonal_SeqFAIJ,
                               /*114*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*119*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*124*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ MatSetBlockSizes_Default,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*144*/0,
                                       0
};



PetscErrorCode  MatSeqFAIJSetPreallocation_SeqFAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqFAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       i,mbs,nbs,bs2;
  PetscBool      skipallocation = PETSC_FALSE,realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  ierr = MatSetBlockSize(B,PetscAbs(bs));CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;

  mbs = B->rmap->n/bs;
  nbs = B->cmap->n/bs;
  bs2 = bs*bs;

  if (mbs*bs!=B->rmap->n || nbs*bs!=B->cmap->n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows %D, cols %D must be divisible by blocksize %D",B->rmap->N,B->cmap->n,bs);

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %D",nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %D value %D",i,nnz[i]);
      if (nnz[i] > nbs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %D value %D rowlength %D",i,nnz[i],nbs);
    }
  }

  b    = (Mat_SeqFAIJ*)B->data;

  b->mbs = mbs;
  b->nbs = nbs;
  if (!skipallocation) {
    if (!b->imax) {
      ierr = PetscMalloc2(mbs,&b->imax,mbs,&b->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);

      b->free_imax_ilen = PETSC_TRUE;
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) b->ilen[i] = 0;
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz < 0) nz = 1;
      nz = PetscMin(nz,nbs);
      for (i=0; i<mbs; i++) b->imax[i] = nz;
      nz = nz*mbs;
    } else {
      nz = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
    }

    /* allocate the matrix space */
    ierr = MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i);CHKERRQ(ierr);
    if (B->structure_only) {
      ierr = PetscMalloc1(nz,&b->j);CHKERRQ(ierr);
      ierr = PetscMalloc1(B->rmap->N+1,&b->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc3(bs2*nz,&b->a,nz,&b->j,B->rmap->N+1,&b->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
      ierr = PetscMemzero(b->a,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
    ierr = PetscMemzero(b->j,nz*sizeof(PetscInt));CHKERRQ(ierr);

    if (B->structure_only) {
      b->singlemalloc = PETSC_FALSE;
      b->free_a       = PETSC_FALSE;
    } else {
      b->singlemalloc = PETSC_TRUE;
      b->free_a       = PETSC_TRUE;
    }
    b->free_ij = PETSC_TRUE;

    b->i[0] = 0;
    for (i=1; i<mbs+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }

  } else {
    b->free_a  = PETSC_FALSE;
    b->free_ij = PETSC_FALSE;
  }

  b->bs2              = bs2;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz;
  B->info.nz_unneeded = (PetscReal)b->maxnz*bs2;
  B->was_assembled    = PETSC_FALSE;
  B->assembled        = PETSC_FALSE;
  if (realalloc) {ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


/*MC
   MATSEQFAIJ - MATSEQFAIJ = "seqfaij" - A matrix type to be used for multiple matrices with the same nonzero structure

   Options Database Keys:
. -mat_type seqfaij - sets the matrix type to "seqfaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqFAIJ()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_SeqFAIJ(Mat B)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_SeqFAIJ    *b;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  b->row          = 0;
  b->col          = 0;
  b->icol         = 0;
  b->reallocs     = 0;
  b->saved_values = 0;

  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = 0;
  B->spptr              = 0;
  B->info.nz_unneeded   = (PetscReal)b->maxnz*b->bs2;
  b->keepnonzeropattern = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqFAIJSetPreallocation_C",MatSeqFAIJSetPreallocation_SeqFAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQFAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicateNoCreate_SeqFAIJ(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqFAIJ    *c = (Mat_SeqFAIJ*)C->data,*a = (Mat_SeqFAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 = a->bs2;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt matrix");

  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax           = a->imax;
    c->ilen           = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    ierr = PetscMalloc2(mbs,&c->imax,mbs,&c->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i];
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (mallocmatspace) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      ierr = PetscCalloc1(bs2*nz,&c->a);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,a->i[mbs]*bs2*sizeof(PetscScalar));CHKERRQ(ierr);

      c->i            = a->i;
      c->j            = a->j;
      c->singlemalloc = PETSC_FALSE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_FALSE;
      c->parent       = A;
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;

      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc3(bs2*nz,&c->a,nz,&c->j,mbs+1,&c->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,a->i[mbs]*(bs2*sizeof(PetscScalar)+sizeof(PetscInt))+(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);

      c->singlemalloc = PETSC_TRUE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_TRUE;

      ierr = PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      if (mbs > 0) {
        ierr = PetscMemcpy(c->j,a->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
        if (cpvalues == MAT_COPY_VALUES) {
          ierr = PetscMemcpy(c->a,a->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
        } else {
          ierr = PetscMemzero(c->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
        }
      }
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;
    }
  }

  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  ierr = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);

  c->bs2         = a->bs2;
  c->mbs         = a->mbs;
  c->nbs         = a->nbs;

  if (a->diag) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      c->diag      = a->diag;
      c->free_diag = PETSC_FALSE;
    } else {
      ierr = PetscMalloc1(mbs+1,&c->diag);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0; i<mbs; i++) c->diag[i] = a->diag[i];
      c->free_diag = PETSC_TRUE;
    }
  } else c->diag = 0;

  c->nz         = a->nz;
  c->maxnz      = a->nz;         /* Since we allocate exactly the right amount */
  c->solve_work = NULL;
  c->mult_work  = NULL;
  c->sor_workt  = NULL;
  c->sor_work   = NULL;

  c->compressedrow.use   = a->compressedrow.use;
  c->compressedrow.nrows = a->compressedrow.nrows;
  if (a->compressedrow.use) {
    i    = a->compressedrow.nrows;
    ierr = PetscMalloc2(i+1,&c->compressedrow.i,i+1,&c->compressedrow.rindex);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,(2*i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(c->compressedrow.i,a->compressedrow.i,(i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(c->compressedrow.rindex,a->compressedrow.rindex,i*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
  }
  C->nonzerostate = A->nonzerostate;

  ierr = PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
   MatCreateSeqFAIJ - Creates a sparse matrix in block FAIJ compressed row format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block, the blocks are ALWAYS square.
.  m - number of rows
.  n - number of columns
.  nz - number of nonzero blocks  per block row (same for all rows)
-  nnz - array containing the number of nonzero blocks in the various block rows
         (possibly different for each block row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Level: intermediate

   Notes:
   The number of rows and columns must be divisible by blocksize.

   If the nnz parameter is given then the nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateFAIJ()
@*/
PetscErrorCode  MatCreateSeqFAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQFAIJ);CHKERRQ(ierr);
  ierr = MatSeqFAIJSetPreallocation(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqFAIJSetPreallocation - Sets the block size and expected nonzeros
   per row in the matrix. For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix
.  bs - size of block, the blocks are ALWAYS square. 
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the various block rows
         (possibly different for each block row) or NULL


   Level: intermediate

   Notes:
   If the nnz parameter is given then the nz parameter is ignored

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string
   malloc in them to see if additional memory allocation was needed.

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateFAIJ(), MatGetInfo()
@*/
PetscErrorCode  MatSeqFAIJSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatSeqFAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

