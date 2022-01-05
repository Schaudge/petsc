static char help[] = "Example of Kokkos shared memory multi-dimensional arrays and creating Kokkos arrays with raw C arrays\n";

#include <petscsys.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

typedef Kokkos::TeamPolicy<>::member_type team_member;
#define SZ 32
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       (*d_gIdx)[SZ][SZ], (*gIdx)[SZ][SZ],m=2,n=4,k=3;
  using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using real1D_scr_t = Kokkos::View<PetscScalar**, Kokkos::LayoutRight, scr_mem_t>;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  if (SZ < m) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"m = %D too large",m);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  if (SZ < n) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"n = %D too large",n);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = PetscMalloc(k*sizeof(*gIdx), &gIdx);CHKERRQ(ierr);
  for (int e = 0; e < k; ++e)
    for (int f = 0; f < n; ++f)
      for (int b = 0; b < m; ++b)
        gIdx[e][f][b] = 10000*e + 100*f + b;

  const Kokkos::View< PetscInt*[SZ][SZ], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_gidx_k ((PetscInt*)gIdx, k);
  Kokkos::View<PetscInt*[SZ][SZ], Kokkos::LayoutRight> *d_gidx_k = new Kokkos::View<PetscInt*[SZ][SZ], Kokkos::LayoutRight>("gIdx", k);
  Kokkos::deep_copy (*d_gidx_k, h_gidx_k);
  d_gIdx = (PetscInt (*)[SZ][SZ]) d_gidx_k->data();

  const int scr_bytes_fdf = real1D_scr_t::shmem_size(n,m);
  Kokkos::parallel_for("test", Kokkos::TeamPolicy<>(k, 16, 16).set_scratch_size(1, Kokkos::PerTeam(scr_bytes_fdf)), KOKKOS_LAMBDA (const team_member team) {
      const PetscInt loc_elem = team.league_rank();
      real1D_scr_t   coef_buff(team.team_scratch(1),n,m);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,n), [=] (int f) {
          PetscInt *const Idxs = &d_gIdx[loc_elem][f][0];
          //for (int f = 0; f < loc_Nf; ++f) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,m), [=] (int b) {
              //for (int b = 0; b < Nb; ++b) {
              PetscInt idx = Idxs[b];
              printf("%d,%d,%d) %d\n",loc_elem,f,b,idx);
              coef_buff(f,b) = (PetscScalar) idx;
            });
        });
    });
  ierr = PetscFree(gIdx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: kokkos

   test:
     suffix: 0
     requires: kokkos
     args: -n 2 -m 3 -k 4
     nsize:  1

TEST*/
