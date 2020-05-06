#if !defined (PETSCSNESDEF_H)
#define PETSCSNESDEF_H

#include "petsc/finclude/petscksp.h"

#define SNES type(tSNES)
#define PetscConvEst type(tPetscConvEst)

#define SNESType character*(80)
#define SNESLineSearchType character*(80)
#define SNESMSType character*(80)

#define SNESConvergedReason PetscEnum
#define SNESLineSearchReason PetscEnum
#define MatMFFD PetscFortranAddr
#define MatMFFDType PetscFortranAddr
#define SNESLineSearch PetscFortranAddr
#define SNESLineSearchOrder PetscEnum
#define SNESNormSchedule PetscEnum
#define SNESQNType PetscEnum
#define SNESQNRestartType PetscEnum
#define SNESQNCompositionType PetscEnum
#define SNESQNScaleType PetscEnum
#define SNESNCGType PetscEnum
#define SNESNGMRESRestartType PetscEnum
#define SNESNGMRESSelectType PetscEnum

#define SNESNEWTONLS         'newtonls'
#define SNESNEWTONTR         'newtontr'
#define SNESPYTHON           'python'
#define SNESNRICHARDSON      'nrichardson'
#define SNESKSPONLY          'ksponly'
#define SNESKSPTRANSPOSEONLY 'ksptransposeonly'
#define SNESVINEWTONRSLS     'vinewtonrsls'
#define SNESVINEWTONSSLS     'vinewtonssls'
#define SNESNGMRES           'ngmres'
#define SNESQN               'qn'
#define SNESSHELL            'shell'
#define SNESNCG              'ncg'
#define SNESFAS              'fas'
#define SNESMS               'ms'
#define SNESLINESEARCHBT                 'bt'
#define SNESLINESEARCHBASIC              'basic'
#define SNESLINESEARCHL2                 'l2'
#define SNESLINESEARCHCP                 'cp'
#define SNESLINESEARCHSHELL              'shell'
#define SNESLINESEARCHNCGLINEAR          'ncglinear'
#define SNES_LINESEARCH_ORDER_LINEAR    1
#define SNES_LINESEARCH_ORDER_QUADRATIC 2
#define SNES_LINESEARCH_ORDER_CUBIC     3
#define SNESMSM62       'm62'
#define SNESMSEULER     'euler'
#define SNESMSJAMESON83 'jameson83'
#define SNESMSVLTP21    'vltp21'
#define SNESMSVLTP31    'vltp31'
#define SNESMSVLTP41    'vltp41'
#define SNESMSVLTP51    'vltp51'
#define SNESMSVLTP61    'vltp61'

#endif
