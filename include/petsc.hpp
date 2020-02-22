
/*
    TODO; use exceptions to handle errors
*/

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <iostream>
#include <map>
#include <string>
#include <iterator>
#include <algorithm>

namespace PETSC {

typedef PetscScalar FieldValues;
template<typename T> using Parameter = T;
typedef enum {MeshStructured} MeshType;

void sPetscInitialize(void) __attribute__ ((constructor));
void sPetscFinalize(void) __attribute__ ((destructor));

void sPetscInitialize(void)
{
  PetscInitializeNoArguments();
  MPI_Comm_set_errhandler(PETSC_COMM_WORLD,MPI_ERRORS_ARE_FATAL);
  PetscPushErrorHandler(PetscAbortErrorHandler,NULL);
}

void sPetscFinalize(void)
{
  PetscPopErrorHandler();
  PetscFinalize();
}

typedef enum {NotSet,Linear,NonLinear,ODE} SolverType;

//  Used for passing information about AppParameters to Petsc object
class PetscParameterObj {
  public:
    PetscInt      min = PETSC_MIN_INT,max = PETSC_MAX_INT;
    const char    *name;
    std::size_t   offset;
    PetscDataType dtype;
    PetscParameterObj(PetscReal r,const char* iname,std::size_t ioffset) {this->name = iname; this->offset = ioffset; this->dtype = PETSC_REAL;}
    PetscParameterObj(PetscInt i,const char* iname,std::size_t ioffset) {this->name = iname; this->offset = ioffset; this->dtype = PETSC_INT;}
    PetscParameterObj(PetscInt i,const char* iname,std::size_t ioffset,PetscInt imin) {this->name = iname; this->offset = ioffset; this->dtype = PETSC_INT; this->min = imin;}
    PetscParameterObj(PetscInt i,const char* iname,std::size_t ioffset,PetscInt imin, PetscInt imax) {this->name = iname;this->offset = ioffset;this->dtype = PETSC_INT;this->min = imin;this->max = imax;}
    PetscParameterObj(const PetscParameterObj &m) {this->name = m.name;this->offset = m.offset;this->dtype = PETSC_INT;this->min = m.min;this->max = m.max;}
};
//  Used to store the AppParameters data in the PETSc object
typedef struct obj {
    int  cnt = 0;
    const PetscParameterObj *params[20];
    obj & operator=(const PetscParameterObj *m) { params[cnt++] = m; return *this;};
    obj & operator+=(const PetscParameterObj *m) { params[cnt++] = m; return *this;};
} ParameterNames;

  // The base implementation that is never used
  template<int dim,MeshType dmtype,typename strct,typename actx> class PETSc {
    public:
  };

  template<typename strct,typename actx>
  class  PETSc<2,MeshStructured,strct,actx> {

    typedef PetscErrorCode (*InitializeData)(PETSc<2,MeshStructured,strct,actx>*);

    typedef PetscErrorCode (*DMDASNESFunctionLocal)(DMDALocalInfo*,strct **,strct**,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDASNESGuessFunctionLocal)(DMDALocalInfo*,strct**,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDASNESJacobianLocal)(DMDALocalInfo,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDASNESJacobianMultLocal)(DMDALocalInfo*,strct**,PETSc<2,MeshStructured,strct,actx>*);

    typedef PetscErrorCode (*DMDATSRHSFunctionLocal)(DMDALocalInfo*,PetscReal,strct **,strct**,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDATSInitialConditionsLocal)(DMDALocalInfo*,PetscReal,strct**,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDATSRHSJacobianLocal)(DMDALocalInfo*,PetscReal,strct**,Mat,Mat,PETSc<2,MeshStructured,strct,actx>*);
    typedef PetscErrorCode (*DMDATSRHSJacobianMultLocal)(DMDALocalInfo*,PETSc<2,MeshStructured,strct,actx>*);

    public:
      PetscBool                     setupcalled = PETSC_FALSE;
      MPI_Comm                      comm = PETSC_COMM_WORLD;
      PetscInt                      dof = sizeof(strct)/sizeof(PetscScalar);
      const char *                  prefix = nullptr;
      const char*                   options = nullptr;
      std::array<const char*,20>    fieldNames = {0};
      ParameterNames                parameterNames;

      DM                            dm = nullptr;
      std::array<DMBoundaryType, 2> boundaryType = {DM_BOUNDARY_NONE,DM_BOUNDARY_NONE};
      DMDAStencilType               stencilType = DMDA_STENCIL_STAR;
      PetscInt                      stencilWidth = 1;

      InitializeData                initializeData = nullptr;

      SNES                          snes = nullptr;
      DMDASNESFunctionLocal         nonlinearFunction = nullptr;
      DMDASNESGuessFunctionLocal    guessFunction = nullptr;
      struct jacobian {
        DMDASNESJacobianLocal     j;
        DMDASNESJacobianMultLocal k;
        void operator=(DMDASNESJacobianLocal m) { this->j = m; }
        void operator=(DMDASNESJacobianMultLocal m) { this->k = m; }
      };

      TS                            ts = nullptr;
      DMDATSRHSFunctionLocal        RHSFunction = nullptr;
      DMDATSInitialConditionsLocal  initialConditions = nullptr;
      struct RHSJacobian {
        DMDATSRHSJacobianLocal     l;
        DMDATSRHSJacobianMultLocal m;
        void operator=(DMDATSRHSJacobianLocal m) { this->l = m; }
        void operator=(DMDATSRHSJacobianMultLocal m) { this->m = m; }
      } RHSJacobian;
      Vec                           solution = NULL;

      SolverType                    solvertype = NotSet;
      actx                          parameters;

    PETSc(std::array<const char*,20> fn) {fieldNames = fn;};
    void setUp(void) {

      if (options) PetscOptionsInsertString(NULL,options);
      PETSC_UNUSED int dummy = PetscOptionsBegin(comm,prefix,"Application options",NULL);
        for (int i=0; i< parameterNames.cnt; i++) {
          char   *name;
          size_t len;
          PetscStrlen(parameterNames.params[i]->name,&len);
          PetscMalloc1(len+1,&name);
          PetscStrcpy(name,"-");
          PetscStrcat(name,parameterNames.params[i]->name);
          if (parameterNames.params[i]->dtype == PETSC_REAL) {
            PetscOptionsReal(name,NULL,NULL,*(PetscReal*) ((char*)(this) + parameterNames.params[i]->offset),(PetscReal*) ((char*)(this) + parameterNames.params[i]->offset),NULL);
          }
          PetscFree(name);
        }
      dummy = PetscOptionsEnd();

      DMDACreate2d(comm,boundaryType[0],boundaryType[1],stencilType,5,5,PETSC_DECIDE,PETSC_DECIDE, dof,stencilWidth,NULL,NULL,&dm);
      if (prefix) DMSetOptionsPrefix(dm,prefix);
      DMSetFromOptions(dm);
      DMSetUp(dm);
      int i = 0;
      while (fieldNames[i++]);
      if (i-1 == dof) {
        const char **names;
        PetscMalloc1(dof+1,&names);
        for (int i=0; i<dof; i++) {names[i] = fieldNames[i];} names[dof] = NULL;
        DMDASetFieldNames(dm,names);
        PetscFree(names);
      }

      //   Determine solver type
      if (nonlinearFunction) solvertype = NonLinear;
      if (RHSFunction) {
        if (solvertype != NotSet) exit(1);
        solvertype = ODE;
      }

      if (solvertype == NonLinear) {
        SNESCreate(comm,&snes);
        if (prefix) SNESSetOptionsPrefix(snes,prefix);
        SNESSetDM(snes,dm);
        if (nonlinearFunction) {
          DMDASNESSetFunctionLocal(dm,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))nonlinearFunction,this);
        }
        if (guessFunction) {
          DMDASNESSetGuessFunctionLocal(dm,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*))guessFunction,this);
        }
        if (initializeData) {
          (*initializeData)(this);
        }
        SNESSetFromOptions(snes);
      } else if (solvertype == ODE) {
        TSCreate(comm,&ts);
        TSSetDM(ts,dm);
        // just now for testing
        TSSetType(ts,TSARKIMEX);
        TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);
        TSSetProblemType(ts,TS_NONLINEAR);
        TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
        TSSetProblemType(ts,TS_NONLINEAR);
        TSSetFromOptions(ts);
        if (RHSFunction) {
          DMDATSSetRHSFunctionLocal(dm,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo *,PetscReal,void *,void *,void *))RHSFunction,this);
        }
        if (RHSJacobian.l) {
          DMDATSSetRHSJacobianLocal(dm,(PetscErrorCode (*)(DMDALocalInfo *,PetscReal,void *,Mat,Mat,void *))RHSJacobian.l,this);
        }
        if (initializeData) {
          (*initializeData)(this);
        }
        if (initialConditions) {
          DMDATSSetInitialConditionsLocal(dm,(PetscErrorCode (*)(DMDALocalInfo *,PetscReal,void *,void*))initialConditions,this);
        }
        DMCreateGlobalVector(dm,&solution);
        TSSetSolution(ts,solution);
        TSSetFromOptions(ts);
      }  else exit(2);

      setupcalled = PETSC_TRUE;
    }
    ~PETSc() {VecDestroy(&solution);TSDestroy(&ts);SNESDestroy(&snes);DMDestroy(&dm);}

    void view() {
      if (!setupcalled) this->setUp();
      DMView(dm,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm)));
      if (solvertype == NonLinear) {
        SNESView(snes,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm)));
      } else if (solvertype == ODE) {
        TSView(ts,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm)));
      } else exit(4);
    }

    void view(const char *voptions) {
      if (!setupcalled) this->setUp();
      PetscOptionsInsertString(NULL,voptions);
      char *voptionskey;
      PetscToken token; PetscTokenCreate(voptions,' ',&token);PetscTokenFind(token,&voptionskey);
      char* flg; PetscStrstr(voptionskey,"solution",&flg);
      if (flg) {
        if (solvertype == NonLinear) {
          Vec vec_sol;
          SNESGetSolution(snes,&vec_sol);
          VecViewFromOptions(vec_sol,NULL,voptionskey);
        }
      } else {
        if (solvertype == NonLinear) {
          char* prefix = ((PetscObject)snes)->prefix; ((PetscObject)snes)->prefix = NULL;
          SNESViewFromOptions(snes,NULL,voptionskey);
          ((PetscObject)snes)->prefix = prefix;
        } else if (solvertype == ODE) {
          char* prefix = ((PetscObject)ts)->prefix; ((PetscObject)ts)->prefix = NULL;
          TSViewFromOptions(ts,NULL,voptionskey);
          ((PetscObject)ts)->prefix = prefix;
        } else exit(5);
      }
      PetscTokenDestroy(&token);
    }

    void solve() {
      if (!setupcalled) this->setUp();
      if (solvertype == NonLinear) {
        SNESSolve(snes,NULL,NULL);
      } else if (solvertype == ODE) {
        TSComputeInitialCondition(ts,solution); // TODO should be handled by TS
        TSSolve(ts,NULL);
      }
    }
};

#if defined(f00)
  template<typename strct,typename actx>
  class  PETSc<2,PetscMeshStructured,PetscODE,strct,actx> {
    public:

    typedef PetscErrorCode (*DMDATSRHSFunctionLocal)(DMDALocalInfo*,PetscReal,strct **,strct**,actx*);
    typedef PetscErrorCode (*DMDATSInitialConditionsLocal)(DMDALocalInfo*,strct**,actx*);
    typedef PetscErrorCode (*DMDATSRHSJacobianLocal)(DMDALocalInfo*,strct**,actx*);
    typedef PetscErrorCode (*DMDATSRHSJacobianMultLocal)(DMDALocalInfo*,actx*);
    typedef PetscErrorCode (*InitializeData)(DM,actx*);

    TS                           ts = nullptr;
    DM                           dm = nullptr;
    PetscInt                     dof = sizeof(strct)/sizeof(PetscScalar);
    MPI_Comm                     comm = PETSC_COMM_WORLD;
    PetscBool                    computeAdjoint = PETSC_FALSE;
    DMDATSRHSFunctionLocal       RHSFunction;
    DMDATSInitialConditionsLocal initialConditions;
    InitializeData               initializeData;
    actx                         parameters;

    struct RHSJacobian {
      DMDATSRHSJacobianLocal     j;
      DMDATSRHSJacobianMultLocal k;
      void operator=(DMDATSRHSJacobianLocal m) { this->j = m; }
      void operator=(DMDATSRHSJacobianMultLocal m) { this->k = m; }
    } RHSJacobian;

    Petsc() {};
    void setUp(void) {

      DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,5,5,PETSC_DECIDE,PETSC_DECIDE, dof,1,NULL,NULL,(DM*)&dm);
      DMSetFromOptions((DM)dm);
      DMSetUp((DM)dm);
      TSCreate(comm,&ts);
      TSSetDM(ts,dm);
      // just now for testing
      TSSetType(ts,TSARKIMEX);
      TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);
      TSSetProblemType(ts,TS_NONLINEAR);
      TSSetMaxTime(ts,2000.0);
      TSSetTimeStep(ts,.0001);
      TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);

      TSSetFromOptions(ts);
      if (RHSFunction) {
        DMDATSSetRHSFunctionLocal(dm,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo *,void *,void *,void *))RHSFunction);
      }
      if (RHSJacobian->j) {
        DMDATSSetRHSJacobianLocal(dm,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo *,void *,void *,void *))RHSJacobian->j);
      }
      if (initialConditions) {
        DMDATSSetInitialConditionsLocal(dm,(PetscErrorCode (*)(DMDALocalInfo *,void *,void *))initialConditions);
      }
      if (initializeData) {
        (*initializeData)(this);
      }
    }
    ~Petsc() {TSDestroy(&ts);DMDestroy(&dm);}

    void view() {
      DMView(dm,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm)));
      TSView(ts,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm)));
    }

    void solve() {TSSolve(ts,NULL);}
};
#endif

#define PetscParameter(PET,NAME,...)  new PetscParameterObj(PET.parameters.NAME,#NAME,offsetof(typeof(PET),parameters.NAME) __VA_OPT__())
}


