#include <petscconf.h>

#ifdef PETSC_HAVE_GRID

#include <petscgrid.h>
//#include "petsc_fermion_parameters.h"
#include "petsc_fermion.h"
#include <Grid/Grid.h>
//#include <Grid/qcd/utils/GaugeGroup.h>

WilsonParameters             TheWilsonParameters;
DomainWallParameters         TheDomainWallParameters;
MobiusDomainWallParameters   TheMobiusDomainWallParameters;

NAMESPACE_BEGIN(Grid);

static PetscErrorCode FormGauge(DM);

template<class vobj>
int PetscToGrid(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  PetscFunctionBeginUser;
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);

  const PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;


  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(psi, &psih));

  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =PetscRealPart(psih[off + d]);
      double im =PetscImaginaryPart(psih[off + d]);
      g_p[d]=ComplexD(re,im);
    }
  }
  assert(idx==lsites);
  vectorizeFromLexOrdArray(scalardata,g_psi);
  PetscFunctionReturn(0);
}

template<class vobj>
int GridToPetsc(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  PetscFunctionBeginUser;
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,g_psi);

  PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;


  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArray(psi, &psih));
  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =real(g_p[d]);
      double im =imag(g_p[d]);
      psih[off + d] = re + im * PETSC_i;
    }
  }
  assert(idx==lsites);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Sets the Guage links in a DM using the lattice gauge field
  generated by GRID
*/
static PetscErrorCode SetGauge_Grid(DM dm, LatticeGaugeField & Umu)
{
  typedef typename LatticeGaugeField::scalar_object sobj;
  GridBase *grid = Umu.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,Umu);

  DM           auxDM;
  Vec          auxVec, globalVector;
  PetscSection s;
  PetscScalar  *id;
  PetscInt     eStart, eEnd, low, high;

  PetscFunctionBegin;

  PetscSynchronizedPrintf(PETSC_COMM_WORLD, "LSITES: %lu\n", lsites);
  PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  // Get the ownership range of the global vector and reduce by degrees of freedom
  PetscCall(DMGetGlobalVector(auxDM, &globalVector));
  PetscCall(VecGetOwnershipRange(globalVector, &low, &high));
  low = low/9; high = high/9;// remove dofs and just use the edge numbering.
  PetscSynchronizedPrintf(PETSC_COMM_WORLD, "edge range on this process: %" PetscInt_FMT " %" PetscInt_FMT  "high-low: %" PetscInt_FMT "\n", low, high, high-low);
  PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));// these are edges, not vertices!!!!!!
  //PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Edges on this rank: %" PetscInt_FMT "\n", eEnd-eStart);
  //PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  PetscInt j=0;
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];
  //for (PetscInt i = eStart; i < eEnd; ++i) {
  for(PetscInt i = low; i < high; ++i) {
    //PetscSynchronizedPrintf(PETSC_COMM_WORLD, "index: %" PetscInt_FMT "\n", j);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    ColourMatrixD U = Grid_p[j];
    id = (PetscScalar *) &U;
    PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
    j++;
  }
  //PetscSynchronizedPrintf(PETSC_COMM_WORLD, " Set %" PetscInt_FMT" gauge links\n", j);
  //PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Grid lsites %ld\n", lsites);
  PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// We do not want a full 5D hypercubic mesh, the answer is to just copy the discretization across clones
// of the plex and set the gauge links manually that way.
static PetscErrorCode SetGauge_Grid5D(DM dm, LatticeGaugeField & Umu, PetscInt Ls, PetscReal m)
{
  PetscScalar  idm[9] = {-m, 0., 0., 0., -m, 0., 0., 0., -m};
  PetscScalar  id[9]  = { 1., 0., 0., 0., 1., 0., 0., 0., 1.};
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  *up;
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  typedef typename LatticeGaugeField::scalar_object sobj;
  GridBase *grid = Umu.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,Umu);

  PetscPrintf(PETSC_COMM_WORLD, "lsites: %ld\n", lsites);
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];
  for (PetscInt i = eStart; i < eEnd; ++i) {
    int d = i % 5;
    int s5=i/5;
    int ss=s5%Ls;
    int s4=s5/Ls;
    if ( d==0 ) { // Unit link in s-dim
      if ( ss == Ls-1 ) {
	      PetscCall(VecSetValuesSection(auxVec, s, i, idm, INSERT_VALUES));
      } else {
	      PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
      }
    } else {
      int jj=d-1 + 4*s4;
      //PetscPrintf(PETSC_COMM_WORLD, "edge: %d of %d\n", i, eEnd - eStart);
      ColourMatrixD U = Grid_p[jj];
      up = (PetscScalar *) &U;
      PetscCall(VecSetValuesSection(auxVec, s, i, up, INSERT_VALUES));
    }
  }
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode FormGauge(DM  dm, PetscReal shift, GRID_LOAD_TYPE type, int argc, char** argv, const char* filename) {
  FieldMetaData header;

  PetscFunctionBegin;
  Grid::Grid_init(&argc,&argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());//1 is Nd ?
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian         GRID(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGRID(&GRID);
  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&GRID);
  pRNG.SeedFixedIntegers(seeds);
  LatticeGaugeFieldD    Umu(&GRID);
  switch (type){
    case 0:
      SU3::ColdConfiguration(pRNG, Umu);
      break;
    case 1:
      // This needs to be built into the file with the configurations so that they don't get overwritten like i just did
      //SU3::TunedConfiguration(shift, pRNG, Umu);
      SU3::TepidConfiguration(pRNG,Umu);
      break;
    case 2:
      SU3::HotConfiguration(pRNG, Umu);
      break;
    case 3:
      std::cout<<GridLogMessage <<"Loading configuration from "<< filename <<std::endl;
      if (!filename) SETERRQ(MPI_COMM_WORLD, PETSC_ERR_ARG_NULL, "No filename provided for Grid. ");
      NerscIO::readConfiguration(Umu, header, filename);
      break;
    default:
      SETERRQ(MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No support for type %" PetscInt_FMT, type);
      break;
  }
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix
  U_GT = Umu;
  PetscCall(SetGauge_Grid(dm, Umu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckDwfWithGrid(DM dm, Mat dw, Vec psi,Vec res)
{
  DomainWallParameters p;
  p.M5   = 1.8;
  p.m    = 0.01;
  p.Ls   = 8;

  PetscFunctionBegin;
  SetDomainWallParameters(&p);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         GRID(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGRID(&GRID);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&GRID);
  pRNG.SeedFixedIntegers(seeds);

  LatticeGaugeFieldD Umu(&GRID);
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix

  std::string name ("ckpoint_EODWF_lat.125");
  std::cout<<GridLogMessage <<"Loading configuration from "<< name<<std::endl;
  FieldMetaData header;
  NerscIO::readConfiguration(Umu, header, name);
  U_GT = Umu;

  ////////////////////////////////////////////////////
  // DWF test
  ////////////////////////////////////////////////////
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(p.Ls,&GRID);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(p.Ls,&GRID);

  LatticeFermionD    g_src(FGrid);
  LatticeFermionD    g_res(FGrid); // Grid result
  LatticeFermionD    p_res(FGrid); // Petsc result
  LatticeFermionD    diff(FGrid); // Petsc result

  std::cout << "Setting gauge to Grid for Ls="<<p.Ls<<std::endl;
//  SetGauge_Grid5D(dm,U_GT,p.Ls,p.m);

  PetscToGrid(dm,psi,g_src);

  DomainWallFermionD DWF(U_GT,*FGrid,*FrbGrid,GRID,RBGRID,p.m,p.M5);

  //  DWF.Dhop(g_src,g_res,0); Passes with no 5d term
  RealD t0=usecond();
  DWF.M(g_src,g_res);
  RealD t1=usecond();
  //PetscCall(Ddwf(dw, psi, res)); // Applies DW
  PetscCall(MatMult(dw, psi, res));
  RealD t2=usecond();
  PetscToGrid(dm,res,p_res);

  diff = p_res - g_res;

  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "Grid  " << t1-t0 <<" us"<<std::endl;
  std::cout << "Petsc " << t2-t1 <<" us"<<std::endl;
  std::cout << "******************************"<<std::endl;

  DWF.Mdag(g_src,g_res);

  //PetscCall(DdwfDag(dw, psi, res)); // Applies DW
  PetscCall(MatMultTranspose(dw, psi, res));
  PetscToGrid(dm,res,p_res);

  diff = p_res - g_res;

  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwDagWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "******************************"<<std::endl;
  PetscPrintf(PETSC_COMM_WORLD, "pre norm\n");
  PetscCheck(norm2(diff) < 1.0e-7, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid check failed.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormGauge5D(DM  dm, GRID_LOAD_TYPE type, PetscBool isPV, PetscInt Ls, int argc, char** argv, const char* filename) {
  // These parameteres need to be moved out.
  DomainWallParameters p;
  FieldMetaData header;

  p.M5   = 1.8;
  if (isPV) p.m = 1.0;
  else p.m = 0.01;
  p.Ls   = 8;

  PetscFunctionBegin;
  if (!isPV) Grid::Grid_init(&argc,&argv);
  SetDomainWallParameters(&p);
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         GRID(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGRID(&GRID);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&GRID);
  pRNG.SeedFixedIntegers(seeds);

  LatticeGaugeFieldD Umu(&GRID);
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix
  std::cout << "Setting gauge to Grid for Ls="<<p.Ls<<std::endl;
  PetscPrintf(PETSC_COMM_WORLD, "ls in petsc %" PetscInt_FMT "\n", p.Ls);
  switch (type){
    case 0:
      SU3::ColdConfiguration(pRNG, Umu);
      break;
    case 1:
      SU3::TepidConfiguration(pRNG,Umu); /*random config*/
      break;
    case 2:
      SU3::HotConfiguration(pRNG, Umu);
      break;
    case 3:
      std::cout<<GridLogMessage <<"Loading configuration from "<< filename <<std::endl;
      if (!filename) SETERRQ(MPI_COMM_WORLD, PETSC_ERR_ARG_NULL, "No filename provided for Grid. ");
      NerscIO::readConfiguration(Umu, header, filename);
      break;
    default:
      SETERRQ(MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No support for type %" PetscInt_FMT, type);
      break;
  }
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(p.Ls,&GRID);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(p.Ls,&GRID);

  // We need to get this back. Store the grid object in a PETSC wrapper that persists like a PETSc object.
  LatticeFermionD    g_src(FGrid);
  PetscCall(SetGauge_Grid5D(dm, Umu, p.Ls, p.m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

NAMESPACE_END(Grid);

extern "C" {
  /*
    Sets Gauge links by using grid. Supported types are cold, tepid, and hot lattice configurations
    as well as checkpointed lattice files.
  */
  PetscErrorCode  PetscSetGauge_Grid(DM dm, PetscReal shift, GRID_LOAD_TYPE type, int argc, char** argv, const char* filename)
  {
    PetscFunctionBegin;
    PetscCall(Grid::FormGauge(dm, shift, type, argc, argv, filename));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode PetscSetGauge_Grid5D(DM dm, GRID_LOAD_TYPE type, PetscBool isPV, PetscInt Ls, int argc, char** argv, const char* filename){
    PetscFunctionBegin;
    PetscCall(Grid::FormGauge5D(dm, type, isPV, Ls, argc, argv, filename));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode PetscCheckDwfWithGrid(DM dm, Mat dw, Vec psi,Vec res) {
    PetscFunctionBegin;
    PetscCall(Grid::CheckDwfWithGrid(dm, dw, psi,res));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

}
#endif
