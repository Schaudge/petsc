#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

static const char *const TaoTermMasks[] = {"none", "objective", "gradient", "hessian", "TaoTermMask", "TAOTERM_MASK_", NULL};

typedef struct _n_TaoTerm_Sum TaoTerm_Sum;

typedef struct _n_TaoTermSumHessCache {
  PetscObjectId    x_id;
  PetscObjectId    p_id;
  PetscObjectState x_state;
  PetscObjectState p_state;
  PetscInt         n_terms;
  Mat             *hessians;
  Vec             *Axs;
} TaoTermSumHessCache;

struct _n_TaoTerm_Sum {
  PetscInt            n_terms;
  TaoMappedTerm      *terms;
  TaoTermSumHessCache hessian_cache;
};

static PetscErrorCode TaoTermSumVecNestGetSubVecsRead(Vec params, PetscInt *n, Vec **subparams, PetscBool **is_dummy)
{
  PetscContainer is_dummy_container = NULL;

  PetscFunctionBegin;
  *is_dummy = NULL;
  PetscCall(VecNestGetSubVecsRead(params, n, subparams));
  PetscCall(PetscObjectQuery((PetscObject)params, "__TaoTermSumParametersPack", (PetscObject *)&is_dummy_container));
  if (is_dummy_container) { PetscCall(PetscContainerGetPointer(is_dummy_container, (void **)is_dummy)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumVecNestRestoreSubVecsRead(Vec params, PetscInt *n, Vec **subparams, PetscBool **is_dummy)
{
  PetscFunctionBegin;
  PetscCall(VecNestRestoreSubVecsRead(params, n, subparams));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumHessCacheReset(TaoTermSumHessCache *cache)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < cache->n_terms; i++) PetscCall(MatDestroy(&cache->hessians[i]));
  PetscCall(PetscFree(cache->hessians));
  for (PetscInt i = 0; i < cache->n_terms; i++) PetscCall(VecDestroy(&cache->Axs[i]));
  PetscCall(PetscFree(cache->Axs));
  cache->n_terms = 0;
  cache->x_id    = 0;
  cache->p_id    = 0;
  cache->x_state = 0;
  cache->p_state = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TaoTermSumGetSubVec(params, sub_params, is_dummy, i) (!(params) || ((is_dummy) && ((is_dummy)[(i)]))) ? NULL : (sub_params)[(i)]

static PetscErrorCode TaoTermSumHessCacheGetHessians(TaoTerm term, Vec x, Vec params, TaoTermSumHessCache *cache, Mat **hessians, Vec **Axs)
{
  TaoTerm_Sum     *sum = (TaoTerm_Sum *)term->data;
  PetscObjectId    x_id, p_id       = 0;
  PetscObjectState x_state, p_state = 0;

  PetscFunctionBegin;
  if (sum->n_terms != cache->n_terms) PetscCall(TaoTermSumHessCacheReset(cache));
  if (!cache->n_terms) {
    cache->n_terms = sum->n_terms;
    PetscCall(PetscCalloc1(sum->n_terms, &cache->hessians));
    PetscCall(PetscCalloc1(sum->n_terms, &cache->Axs));
    for (PetscInt i = 0; i < sum->n_terms; i++) {
      TaoMappedTerm *summand = &sum->terms[i];
      if (summand->_unmapped_H) {
        PetscCall(PetscObjectReference((PetscObject)summand->_unmapped_H));
        cache->hessians[i] = summand->_unmapped_H;
      } else PetscCall(TaoTermCreateHessianMatrices(summand->term, &cache->hessians[i], NULL));
      if (summand->map) PetscCall(MatCreateVecs(summand->map, NULL, &cache->Axs[i]));
    }
  }
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (params) {
    PetscCall(PetscObjectGetId((PetscObject)params, &p_id));
    PetscCall(PetscObjectStateGet((PetscObject)params, &p_state));
  }
  if (x_id != cache->x_id || x_state != cache->x_state || p_id != cache->p_id || p_state != cache->p_state) {
    Vec       *sub_params = NULL;
    PetscBool *is_dummy   = NULL;

    cache->x_id    = x_id;
    cache->x_state = x_state;
    cache->p_id    = p_id;
    cache->p_state = p_state;
    if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
    for (PetscInt i = 0; i < sum->n_terms; i++) {
      TaoMappedTerm *summand   = &sum->terms[i];
      Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);
      Vec            Ax        = x;

      if (summand->map) {
        PetscCall(MatMult(summand->map, x, cache->Axs[i]));
        Ax = cache->Axs[i];
      }
      PetscCall(TaoTermHessian(summand->term, Ax, sub_param, cache->hessians[i], NULL));
    }
    if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  }
  *hessians = cache->hessians;
  *Axs      = cache->Axs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumIsDummyDestroy(void *ctx)
{
  return PetscFree(ctx);
}

/*@
  TaoTermSumParametersPack - Concatenate the parameters for subterms into a `VECNEST` parameter vector for a `TAOTERMSUM`

  Collective

  Input Parameters:
+ term      - a `TaoTerm` of type `TAOTERMSUM`
- subparams - an array of parameters `Vec`s, one for each subterm in the sum.  An entry can be NULL for a subterm that doesn't take parameters.

  Output Parameter:
. params - a `Vec` of type `VECNEST` that concatenates all of the parameters

  Level: intermediate

  Note:
  This is a wrapper around `VecCreateNest()`, but that function does not allow `NULL` for any of the `Vec`s in the array.  A 0-length
  vector will be created for each NULL `Vec` that wil be internally ignored by `TAOTERMSUM`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumParametersUnpack()`,
          `VECNEST`,
          `VecNestGetTaoTermSumSubParameters()`,
          `VecCreateNest()`
@*/
PetscErrorCode TaoTermSumParametersPack(TaoTerm term, Vec subparams[], Vec *params)
{
  PetscInt       n_terms;
  Vec           *p;
  PetscBool     *is_dummy;
  PetscContainer is_dummy_container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms));
  PetscCall(PetscMalloc1(n_terms, &p));
  PetscCall(PetscMalloc1(n_terms, &is_dummy));
  for (PetscInt i = 0; i < n_terms; i++) {
    if (subparams[i]) {
      PetscValidHeaderSpecific(subparams[i], VEC_CLASSID, 2);
      p[i]        = subparams[i];
      is_dummy[i] = PETSC_FALSE;
    } else {
      TaoTerm               subterm;
      Vec                   dummy_vec;
      TaoTermParametersMode mode;
      VecType               vec_type = VECSTANDARD;
      PetscLayout           layout   = NULL;

      PetscCall(TaoTermSumGetSubterm(term, i, NULL, NULL, &subterm, NULL));
      PetscCall(TaoTermGetParametersMode(subterm, &mode));
      if (mode != TAOTERM_PARAMETERS_NONE) {
        PetscCall(TaoTermGetVecTypes(subterm, NULL, &vec_type));
        PetscCall(TaoTermGetLayouts(subterm, NULL, &layout));
        layout->refcnt++;
      } else {
        PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)term), &layout));
        PetscCall(PetscLayoutSetLocalSize(layout, 0));
        PetscCall(PetscLayoutSetSize(layout, 0));
      }
      PetscCall(VecCreate(PetscObjectComm((PetscObject)term), &dummy_vec));
      PetscCall(VecSetLayout(dummy_vec, layout));
      PetscCall(PetscLayoutDestroy(&layout));
      PetscCall(VecSetType(dummy_vec, vec_type));
      is_dummy[i] = PETSC_TRUE;
      p[i]        = dummy_vec;
    }
  }
  PetscCall(VecCreateNest(PetscObjectComm((PetscObject)term), n_terms, NULL, p, params));
  for (PetscInt i = 0; i < n_terms; i++) {
    if (!subparams[i]) PetscCall(VecDestroy(&p[i]));
  }
  PetscCall(PetscFree(p));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &is_dummy_container));
  PetscCall(PetscContainerSetPointer(is_dummy_container, (void *)is_dummy));
  PetscCall(PetscContainerSetUserDestroy(is_dummy_container, TaoTermSumIsDummyDestroy));
  PetscCall(PetscObjectCompose((PetscObject)*params, "__TaoTermSumParametersPack", (PetscObject)is_dummy_container));
  PetscCall(PetscContainerDestroy(&is_dummy_container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumParametersUnpack - Unpack the concatenated parameters created by `TaoTermSumParametersPack()` and destroy the `VECNEST`

  Collective

  Input Parameters:
+ term   - a `TaoTerm` of type `TAOTERMSUM`
- params - a `Vec` created by `TaoTermSumParametersPack()`

  Output Parameter:
. subparams - an array of parameters `Vec`s, one for each subterm in the sum.  An entry will be NULL if NULL was passed in the same position of `TaoTermSumParametersPack()`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermParametersPack()`,
          `VecNestGetTaoTermSumSubParameters()`,
@*/
PetscErrorCode TaoTermSumParametersUnpack(TaoTerm term, Vec *params, Vec subparams[])
{
  PetscInt       n_terms;
  PetscInt      *is_dummy           = NULL;
  PetscContainer is_dummy_container = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(*params, VEC_CLASSID, 2);
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms));
  PetscCall(PetscObjectQuery((PetscObject)*params, "__TaoTermSumParametersPack", (PetscObject *)&is_dummy_container));
  if (is_dummy_container) PetscCall(PetscContainerGetPointer(is_dummy_container, (void **)&is_dummy));
  for (PetscInt i = 0; i < n_terms; i++) {
    Vec subparam;

    PetscCall(VecNestGetSubVec(*params, i, &subparam));
    if (is_dummy && is_dummy[i]) {
      subparams[i] = NULL;
    } else {
      PetscCall(PetscObjectReference((PetscObject)subparam));
      subparams[i] = subparam;
    }
  }
  PetscCall(VecDestroy(params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecNestGetTaoTermSumSubParameters - A wrappper around `VecNestGetSubVec()` for `TAOTERMSUM`.

  Not collective

  Input Parameters:
+ params - a `VECNEST` that has one nested vector for each term of a `TAOTERMSUM`
- index  - the index of a subterm

  Output Parameter:
. subparams - the parameters of the subterm (may be NULL)

  Level: intermediate

  Note:
  `VecNestGetSubVec()` cannot return NULL for the subvec.  If `params` was
  created by `TaoTermParametersPack()`, then any NULL subvecs that were passed
  to that function will be returned NULL by this function.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermParametersPack()`,
          `TaoTermParametersUnpack()`,
          `VECNEST`,
          `VecNestGetSubVec()`,
@*/
PetscErrorCode VecNestGetTaoTermSumSubParameters(Vec params, PetscInt index, Vec *subparams)
{
  PetscInt      *is_dummy           = NULL;
  PetscContainer is_dummy_container = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(params, VEC_CLASSID, 1);
  PetscCall(PetscObjectQuery((PetscObject)params, "__TaoTermSumParametersPack", (PetscObject *)&is_dummy_container));
  if (is_dummy_container) PetscCall(PetscContainerGetPointer(is_dummy_container, (void **)&is_dummy));
  if (is_dummy && is_dummy[index]) {
    *subparams = NULL;
  } else {
    PetscCall(VecNestGetSubVec(params, index, subparams));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermDestroy_Sum(TaoTerm term)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;

  PetscFunctionBegin;
  PetscCall(TaoTermSumHessCacheReset(&sum->hessian_cache));
  for (PetscInt i = 0; i < sum->n_terms; i++) PetscCall(TaoMappedTermReset(&sum->terms[i]));
  for (PetscInt i = 0; i < sum->hessian_cache.n_terms; i++) PetscCall(MatDestroy(&sum->hessian_cache.hessians[i]));
  PetscCall(PetscFree(sum->hessian_cache.hessians));
  PetscCall(PetscFree(sum->terms));
  PetscCall(PetscFree(sum));
  term->data = NULL;
  PetscCall(PetscObjectReference((PetscObject)term->parameters_factory_orig));
  PetscCall(MatDestroy(&term->parameters_factory));
  term->parameters_factory = term->parameters_factory_orig;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetNumSubterms_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetNumSubterms_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubterm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubterm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumAddSubterm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubtermHessianMatrices_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubtermMask_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubtermMask_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Sum(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;
  PetscBool    iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Sum of %" PetscInt_FMT " terms:\n", sum->n_terms));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (PetscInt i = 0; i < sum->n_terms; i++) {
      PetscReal   scale;
      const char *subprefix;
      TaoTerm     subterm;
      Mat         map;
      TaoTermMask mask;

      PetscCall(TaoTermSumGetSubterm(term, i, &subprefix, &scale, &subterm, &map));

      PetscCall(PetscViewerASCIIPrintf(viewer, "Summand %" PetscInt_FMT ":\n", i));
      PetscCall(PetscViewerASCIIPushTab(viewer));

      PetscCall(PetscViewerASCIIPrintf(viewer, "Scale (taoterm_sum_%sscale): %g\n", subprefix, (double)scale));
      if (map == NULL) PetscCall(PetscViewerASCIIPrintf(viewer, "Map: unmapped\n"));
      else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Map:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        {
          PetscViewerFormat format;
          PetscBool         pop = PETSC_FALSE;

          PetscCall(PetscViewerGetFormat(viewer, &format));
          if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
            PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
            pop = PETSC_TRUE;
          }
          PetscCall(MatView(map, viewer));
          if (pop) PetscCall(PetscViewerPopFormat(viewer));
        }
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "Term:\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(TaoTermView(subterm, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(TaoTermSumGetSubtermMask(term, i, &mask));
      if (mask != TAOTERM_MASK_NONE) {
        PetscBool preceding = PETSC_FALSE;

        PetscCall(PetscViewerASCIIPrintf(viewer, "Mask (taoterm_sum_%smask): ", subprefix));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        if (TaoTermObjectiveMasked(mask)) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "objective"));
          preceding = PETSC_TRUE;
        }
        if (TaoTermGradientMasked(mask)) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "%sgradient", preceding ? ", " : ""));
          preceding = PETSC_TRUE;
        }
        if (TaoTermHessianMasked(mask)) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "%shessian", preceding ? ", " : ""));
          preceding = PETSC_TRUE;
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      }

      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetNumSubterms - Set the number of terms in the sum

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
- n_terms - the number of terms that will be in the sum

  Level: intermediate

  Note:
  If `n_terms` is smaller than the current number of terms, the trailing terms will be dropped.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumGetNumSubterms()`
@*/
PetscErrorCode TaoTermSumSetNumSubterms(TaoTerm term, PetscInt n_terms)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(term, n_terms, 2);
  PetscTryMethod(term, "TaoTermSumSetNumSubterms_C", (TaoTerm, PetscInt), (term, n_terms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetNumSubterms_Sum(TaoTerm term, PetscInt n_terms)
{
  TaoTerm_Sum   *sum         = (TaoTerm_Sum *)term->data;
  PetscInt       n_terms_old = sum->n_terms;
  TaoMappedTerm *new_summands;

  PetscFunctionBegin;
  if (n_terms == n_terms_old) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = n_terms; i < n_terms_old; i++) PetscCall(TaoMappedTermReset(&sum->terms[i]));
  PetscCall(PetscMalloc1(n_terms, &new_summands));
  PetscCall(PetscArraycpy(new_summands, sum->terms, PetscMin(n_terms, n_terms_old)));
  PetscCall(PetscArrayzero(&new_summands[n_terms_old], PetscMax(0, n_terms - n_terms_old)));
  PetscCall(PetscFree(sum->terms));
  sum->terms   = new_summands;
  sum->n_terms = n_terms;
  for (PetscInt i = n_terms_old; i < n_terms; i++) PetscCall(TaoTermSumSetSubterm(term, i, NULL, 1.0, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumGetNumSubterms - Get the number of terms in the sum

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMSUM`

  Output Parameter:
. n_terms - the number of terms that will be in the sum

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumSetNumSubterms()`
@*/
PetscErrorCode TaoTermSumGetNumSubterms(TaoTerm term, PetscInt *n_terms)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(n_terms, 2);
  PetscUseMethod(term, "TaoTermSumGetNumSubterms_C", (TaoTerm, PetscInt *), (term, n_terms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetNumSubterms_Sum(TaoTerm term, PetscInt *n_terms)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;

  PetscFunctionBegin;
  *n_terms = sum->n_terms;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumGetSubterm - Get the data for a subterm in a sum of terms

  Not collective

  Input Parameter:
+ term  - a `TaoTerm` of type `TAOTERMSUM`
- index - a number $0 \leq i < n$, where $n$ is the number of terms in `TaoTermSumGetNumSubterms()`

  Output Parameters:
+ prefix  - the prefix used for configuring the subterm
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumSetSubterm()`,
          `TaoTermSumAddSubterm()`,
@*/
PetscErrorCode TaoTermSumGetSubterm(TaoTerm term, PetscInt index, const char **prefix, PetscReal *scale, TaoTerm *subterm, Mat *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (prefix) PetscAssertPointer(prefix, 3);
  if (subterm) PetscAssertPointer(subterm, 5);
  if (scale) PetscAssertPointer(scale, 4);
  if (map) PetscAssertPointer(map, 6);
  PetscUseMethod(term, "TaoTermSumGetSubterm_C", (TaoTerm, PetscInt, const char **, PetscReal *, TaoTerm *, Mat *), (term, index, prefix, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetSubterm_Sum(TaoTerm term, PetscInt index, const char **prefix, PetscReal *scale, TaoTerm *subterm, Mat *map)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];
  PetscCall(TaoMappedTermGetData(summand, prefix, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetSubterm - Set a subterm in a sum of terms

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
. index   - a number $0 \leq i < n$, where $n$ is the number of terms in `TaoTermSumGetNumSubterms()`
. prefix  - the prefix used for configuring the term (if NULL, the `subterm_x_` will be the prefix, e.g. "subterm_0_", "subterm_1_", etc.)
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumGetSubterm()`,
          `TaoTermSumAddSubterm()`,
@*/
PetscErrorCode TaoTermSumSetSubterm(TaoTerm term, PetscInt index, const char prefix[], PetscReal scale, TaoTerm subterm, Mat map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(term, index, 2);
  if (prefix) PetscAssertPointer(prefix, 3);
  if (subterm) PetscValidHeaderSpecific(subterm, TAOTERM_CLASSID, 5);
  PetscValidLogicalCollectiveReal(term, scale, 4);
  if (map) PetscValidHeaderSpecific(map, MAT_CLASSID, 6);
  PetscTryMethod(term, "TaoTermSumSetSubterm_C", (TaoTerm, PetscInt, const char[], PetscReal, TaoTerm, Mat), (term, index, prefix, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetSubterm_Sum(TaoTerm term, PetscInt index, const char prefix[], PetscReal scale, TaoTerm subterm, Mat map)
{
  char           subterm_x_[256];
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];
  if (prefix == NULL) {
    PetscCall(PetscSNPrintf(subterm_x_, 256, "subterm_%" PetscInt_FMT "_", index));
    prefix = subterm_x_;
  }
  PetscCall(TaoMappedTermSetData(summand, prefix, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetSubtermHessianMatrices - Set Hessian matrices that can be used internally by a `TAOTERMSUM`

  Logically collective

  Input Parameters:
+ term          - a `TaoTerm` of type `TAOTERMSUM`
. index         - the index for the subterm from `TaoTermSumSetSubterm() or `TaoTermSumAddSubterm()`
. unmapped_H    - (optional) mapped Hessian matrix
. unmapped_Hpre - (optional) mapped Hessian matrix for preconditioning
. mapped_H      - (optional) Hessian matrix
- mapped_Hpre   - (optional) Hessian matrix for preconditioning

  Level: advanced

  Notes:
  If the subterm has the form $g(x) = \alpha f(Ax; p)$, the "mapped" Hessians should be able to hold the Hessian
  $\nabla^2 g$ and the unmapped Hessians should be able to hold the Hessian $\nabla_x^2 f$.  If the subterm is not mapped,
  just pass the unmapped Hessians (e.g. `TaoTermSumSetSubtermHessianMatrices(term, 0, H, Hpre, NULL, NULL)`).

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermSumSetSubtermHessianMatrices(TaoTerm term, PetscInt index, Mat unmapped_H, Mat unmapped_Hpre, Mat mapped_H, Mat mapped_Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermSumSetSubtermHessianMatrices_C", (TaoTerm, PetscInt, Mat, Mat, Mat, Mat), (term, index, unmapped_H, unmapped_Hpre, mapped_H, mapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetSubtermHessianMatrices_Sum(TaoTerm term, PetscInt index, Mat unmapped_H, Mat unmapped_Hpre, Mat mapped_H, Mat mapped_Hpre)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];

  if (!summand->map) {
    // accept inputs in either position
    unmapped_H = unmapped_H ? unmapped_H : mapped_H;
    mapped_H   = NULL;

    unmapped_Hpre = unmapped_Hpre ? unmapped_Hpre : mapped_Hpre;
    mapped_Hpre   = NULL;
  }

  PetscCall(PetscObjectReference((PetscObject)unmapped_H));
  PetscCall(MatDestroy(&summand->_unmapped_H));
  summand->_unmapped_H = unmapped_H;

  PetscCall(PetscObjectReference((PetscObject)unmapped_Hpre));
  PetscCall(MatDestroy(&summand->_unmapped_Hpre));
  summand->_unmapped_Hpre = unmapped_Hpre;

  PetscCall(PetscObjectReference((PetscObject)mapped_H));
  PetscCall(MatDestroy(&summand->_mapped_H));
  summand->_mapped_H = mapped_H;

  PetscCall(PetscObjectReference((PetscObject)mapped_Hpre));
  PetscCall(MatDestroy(&summand->_mapped_Hpre));
  summand->_mapped_Hpre = mapped_Hpre;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumGetSubtermMask - Get the `TaoTermMask` of a term in the sum

  Logically collective

  Input Parameters:
+ term  - a `TaoTerm` of type `TAOTERMSUM`
- index - the index for the subterm from `TaoTermSumSetSubterm() or `TaoTermSumAddSubterm()`

  Output Parameter:
. mask - a bitmask of `TaoTermMask` evaluation methods to mask (e.g. just `TAOTERM_MASK_OBJECTIVE` or a bitwise-or like `TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT`)

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumSetSubtermMask()`
@*/
PetscErrorCode TaoTermSumGetSubtermMask(TaoTerm term, PetscInt index, TaoTermMask *mask)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscUseMethod(term, "TaoTermSumGetSubtermMask_C", (TaoTerm, PetscInt, TaoTermMask *), (term, index, mask));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetSubtermMask_Sum(TaoTerm term, PetscInt index, TaoTermMask *mask)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];
  *mask   = summand->mask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetSubtermMask - Set a `TaoTermMask` on a term in the sum

  Logically collective

  Input Parameters:
+ term  - a `TaoTerm` of type `TAOTERMSUM`
. index - the index for the subterm from `TaoTermSumSetSubterm() or `TaoTermSumAddSubterm()`
- mask  - a bitmask of `TaoTermMask` evaluation methods to mask (e.g. just `TAOTERM_MASK_OBJECTIVE` or a bitwise-or like `TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT`)

  Options Database Keys:
. -taoterm_sum_<prefix_>mask - a list containing any of `none`, `objective`, `gradient`, and `hessian` to indicate which evaluations to mask for a term with a given prefix (see `TaoTermSumSetSubterm()`)

  Level: advanced

  Note:
  Some optimization methods want to add a damping term to the Hessian of an
  objective function without affecting the objective or gradient.  If, e.g.,
  the regularizer has index `1`, then this can be accomplished with
  `TaoTermSumSetSubtermMask(term, 1, TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT)`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSUM`,
          `TaoTermSumGetSubtermMask()`
@*/
PetscErrorCode TaoTermSumSetSubtermMask(TaoTerm term, PetscInt index, TaoTermMask mask)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermSumSetSubtermMask_C", (TaoTerm, PetscInt, TaoTermMask), (term, index, mask));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetSubtermMask_Sum(TaoTerm term, PetscInt index, TaoTermMask mask)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand       = &sum->terms[index];
  summand->mask = mask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumAddSubterm - Append a subterm to the terms being summed

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
. prefix  - the prefix used for configuring the term (if NULL, the index of the term will be used as a prefix, e.g. "0_", "1_", etc.)
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Output Parameter:
. index - the index of the newly added term (optional, pass NULL if not needed)

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumAddSubterm(TaoTerm term, const char prefix[], PetscReal scale, TaoTerm subterm, Mat map, PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (prefix) PetscAssertPointer(prefix, 2);
  PetscValidHeaderSpecific(subterm, TAOTERM_CLASSID, 4);
  PetscValidLogicalCollectiveReal(term, scale, 3);
  if (map) PetscValidHeaderSpecific(map, MAT_CLASSID, 5);
  if (index) PetscAssertPointer(index, 6);
  PetscTryMethod(term, "TaoTermSumAddSubterm_C", (TaoTerm, const char[], PetscReal, TaoTerm, Mat, PetscInt *), (term, prefix, scale, subterm, map, index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumAddSubterm_Sum(TaoTerm term, const char prefix[], PetscReal scale, TaoTerm subterm, Mat map, PetscInt *index)
{
  PetscInt n_terms_old;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms_old));
  PetscCall(TaoTermSumSetNumSubterms(term, n_terms_old + 1));
  PetscCall(TaoTermSumSetSubterm(term, n_terms_old, prefix, scale, subterm, map));
  if (index) *index = n_terms_old;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_Sum(TaoTerm term, PetscOptionItems *PetscOptionsObject)
{
  PetscInt    n_terms;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)term, &prefix));
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm sum options");
  PetscCall(PetscOptionsBoundedInt("-taoterm_sum_num_subterms", "The number of terms in the sum", "TaoTermSumSetNumSubterms", n_terms, &n_terms, NULL, 0));
  PetscCall(TaoTermSumSetNumSubterms(term, n_terms));
  for (PetscInt i = 0; i < n_terms; i++) {
    const char *subprefix;
    Mat         map;
    PetscReal   scale;
    TaoTerm     subterm;
    char        arg[256];
    PetscBool   flg;
    PetscEnum   masks[4];
    PetscInt    n_masks = PETSC_STATIC_ARRAY_LENGTH(masks);

    PetscCall(TaoTermSumGetSubterm(term, i, &subprefix, &scale, &subterm, &map));
    if (subterm == NULL) {
      PetscCall(TaoTermDuplicate(term, TAOTERM_DUPLICATE_SIZEONLY, &subterm));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)subterm, prefix));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)subterm, subprefix));
    } else PetscCall(PetscObjectReference((PetscObject)subterm));
    PetscCall(TaoTermSetFromOptions(subterm));

    PetscCall(PetscSNPrintf(arg, 256, "-taoterm_sum_%sscale", subprefix));
    PetscCall(PetscOptionsReal(arg, "The scale of the subterm in the sum", "TaoTermSumSetSubterm", scale, &scale, NULL));

    PetscCall(PetscSNPrintf(arg, 256, "-taoterm_sum_%smask", subprefix));
    PetscCall(PetscOptionsEnumArray(arg, "The mask of the subterm in the sum", "TaoTermSumSetSubtermMask", TaoTermMasks, masks, &n_masks, &flg));
    if (flg) {
      TaoTermMask mask = TAOTERM_MASK_NONE;

      for (PetscInt i = 0; i < n_masks; i++) {
        TaoTermMask this_mask = masks[i] ? (1 << (masks[i] - 1)) : TAOTERM_MASK_NONE;

        mask = mask | this_mask;
      }

      PetscCall(TaoTermSumSetSubtermMask(term, i, mask));
    }

    if (map) PetscCall(MatSetFromOptions(map));
    PetscCall(TaoTermSumSetSubterm(term, i, subprefix, scale, subterm, map));
    PetscCall(TaoTermDestroy(&subterm));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Sum(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;
  PetscBool   *is_dummy   = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

    PetscCall(TaoMappedTermObjective(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, value));
  }
  if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Sum(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;
  PetscBool   *is_dummy   = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

    PetscCall(TaoMappedTermGradient(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, g));
  }
  if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Sum(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;
  PetscBool   *is_dummy   = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

    PetscCall(TaoMappedTermObjectiveAndGradient(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, value, g));
  }
  if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_Sum(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;
  PetscBool   *is_dummy   = NULL;

  PetscFunctionBegin;
  PetscCall(TaoTermUpdateHessianShells(term, x, params, &H, &Hpre));
  if (!H && !Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

    PetscCall(TaoMappedTermHessian(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, H, Hpre == H ? NULL : Hpre));
  }
  if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessianMult_Sum(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;
  Mat         *hessians   = NULL;
  Vec         *Axs        = NULL;
  PetscBool   *is_dummy   = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscCall(TaoTermSumHessCacheGetHessians(term, x, params, &sum->hessian_cache, &hessians, &Axs));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

    PetscCall(TaoMappedTermHessianMult(summand, Axs[i] ? Axs[i] : x, sub_param, hessians[i], v, i == 0 ? INSERT_VALUES : ADD_VALUES, Hv));
  }
  if (params) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetUp_Sum(TaoTerm term)
{
  TaoTerm_Sum *sum          = (TaoTerm_Sum *)term->data;
  PetscBool    all_none     = PETSC_TRUE;
  PetscBool    any_required = PETSC_FALSE;
  PetscInt     k = 0, K = 0;
  Mat         *mats, new_parameters_factory;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(sum->n_terms, &mats));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm        *summand = &sum->terms[i];
    TaoTermParametersMode submode;

    PetscCall(TaoTermSetUp(summand->term));
    if (summand->map) PetscCall(MatSetUp(summand->map));
    PetscCall(TaoTermGetParametersMode(summand->term, &submode));
    if (submode == TAOTERM_PARAMETERS_REQUIRED) any_required = PETSC_TRUE;
    if (submode != TAOTERM_PARAMETERS_NONE) {
      PetscInt subk, subK;

      all_none = PETSC_FALSE;
      PetscCall(TaoTermGetParametersSizes(summand->term, &subk, &subK, NULL));
      k += subk;
      K += subK;
    }
    if (summand->term->parameters_mode != TAOTERM_PARAMETERS_NONE) {
      PetscCall(PetscObjectReference((PetscObject)summand->term->parameters_factory));
      mats[i] = summand->term->parameters_factory;
    } else {
      PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &mats[i]));
      PetscCall(MatSetType(mats[i], MATDUMMY));
      PetscCall(MatSetSizes(mats[i], 0, 0, 0, 0));
      PetscCall(MatSetUp(mats[i]));
    }
  }
  if (all_none) {
    term->parameters_mode = TAOTERM_PARAMETERS_NONE;
  } else if (any_required) {
    term->parameters_mode = TAOTERM_PARAMETERS_REQUIRED;
  } else {
    term->parameters_mode = TAOTERM_PARAMETERS_OPTIONAL;
  }
  PetscCall(TaoTermSetParametersSizes(term, k, K, PETSC_DECIDE));
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)term), sum->n_terms, NULL, 1, NULL, mats, &new_parameters_factory));
  PetscCall(MatSetVecType(new_parameters_factory, VECNEST));
  PetscCall(MatDestroy(&term->parameters_factory));
  term->parameters_factory = new_parameters_factory;
  for (PetscInt i = 0; i < sum->n_terms; i++) PetscCall(MatDestroy(&mats[i]));
  PetscCall(PetscFree(mats));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateVecs_Sum(TaoTerm term, Vec *solution_vec, Vec *parameters_vec)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;

  PetscFunctionBegin;
  if (solution_vec) PetscCall(MatCreateVecs(term->solution_factory, NULL, solution_vec));
  if (parameters_vec) {
    Vec *vecs;

    PetscCall(PetscCalloc1(sum->n_terms, &vecs));
    for (PetscInt i = 0; i < sum->n_terms; i++) {
      TaoMappedTerm        *summand = &sum->terms[i];
      TaoTermParametersMode submode;

      PetscCall(TaoTermGetParametersMode(summand->term, &submode));
      if (submode != TAOTERM_PARAMETERS_NONE) { PetscCall(TaoTermCreateVecs(summand->term, NULL, &vecs[i])); }
    }
    PetscCall(TaoTermSumParametersPack(term, vecs, parameters_vec));
    for (PetscInt i = 0; i < sum->n_terms; i++) PetscCall(VecDestroy(&vecs[i]));
    PetscCall(PetscFree(vecs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMSUM - A `TaoTerm` that is a sum of multiple other terms.

  Level: intermediate

  Note:
  The default Hessian creation mode (see `TaoTermGetCreateHessianMode()`) is `H == Hpre` and `TaoTermCreateHessianMatrices()`
  will create a `MATSHELL` for the Hessian.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermSumGetNumSubterms()`,
          `TaoTermSumSetNumSubterms()`,
          `TaoTermSumGetSubterm()`,
          `TaoTermSumSetSubterm()`,
          `TaoTermSumAddSubterm()`,
          `TaoTermSumSetSubtermHessianMatrices()`,
          `TaoTermSumGetSubtermMask()`,
          `TaoTermSumSetSubtermMask()`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Sum(TaoTerm term)
{
  TaoTerm_Sum *sum;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sum));
  term->data            = (void *)sum;
  term->parameters_mode = TAOTERM_PARAMETERS_OPTIONAL;

  term->ops->destroy               = TaoTermDestroy_Sum;
  term->ops->view                  = TaoTermView_Sum;
  term->ops->setfromoptions        = TaoTermSetFromOptions_Sum;
  term->ops->objective             = TaoTermObjective_Sum;
  term->ops->gradient              = TaoTermGradient_Sum;
  term->ops->objectiveandgradient  = TaoTermObjectiveAndGradient_Sum;
  term->ops->hessian               = TaoTermHessian_Sum;
  term->ops->hessianmult           = TaoTermHessianMult_Sum;
  term->ops->setup                 = TaoTermSetUp_Sum;
  term->ops->createvecs            = TaoTermCreateVecs_Sum;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatricesDefault;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetNumSubterms_C", TaoTermSumGetNumSubterms_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetNumSubterms_C", TaoTermSumSetNumSubterms_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubterm_C", TaoTermSumGetSubterm_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubterm_C", TaoTermSumSetSubterm_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumAddSubterm_C", TaoTermSumAddSubterm_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubtermHessianMatrices_C", TaoTermSumSetSubtermHessianMatrices_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubtermMask_C", TaoTermSumGetSubtermMask_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubtermMask_C", TaoTermSumSetSubtermMask_Sum));
  if (!term->H_mattype) PetscCall(PetscStrallocpy(MATSHELL, &term->H_mattype));
  if (!term->Hpre_mattype) PetscCall(PetscStrallocpy(MATAIJ, &term->Hpre_mattype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
