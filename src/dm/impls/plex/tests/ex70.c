static char help[] = "Test for Disjoint Union of DMPlexes \n\n";

#include <petscdmplex.h>


PetscErrorCode UnionTwoRefTriangles()
{
  DM           dm[2],dmunion;
  PetscSection unionmapping; 

  PetscFunctionBegin; 
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF,DM_POLYTOPE_TRIANGLE,&dm[0]));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF,DM_POLYTOPE_TRIANGLE,&dm[1]));

  PetscCall(DMPlexDisjointUnion_Topological_Section(dm,2,&dmunion,&unionmapping));

  PetscCall(DMViewFromOptions(dmunion,NULL,"-viewunion"));
  PetscCall(DMDestroy(&dm[0]));
  PetscCall(DMDestroy(&dm[1]));
  PetscCall(DMDestroy(&dmunion));
  PetscCall(PetscSectionDestroy(&unionmapping));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(UnionTwoRefTriangles()); 
  PetscCall(PetscFinalize());
}
