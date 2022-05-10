#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@C
  DMPlexDisjointUnion_Topological_Section - Computes the disjoint union of an array of DMPlexs 

  Input Parameters:
+ plexlist - An array of dmplex objects to apply the disjoint union to
. numplex  - The number of dmplexs in the array

  Output Parameters:
+ dmunion    - the dmplex formed by the disjoint union 
- stratumoff - A PetscSection encoding the disjoint union. Its chart is [0,numplex), with fields for every stratum in the input dmplexs. 
Each offset from PetscsectionGetFieldOffset(stratumoff,p,stratum,&offset) is the starting point of the pth plex's stratum in the new dmunion
DMPlex, with it's dof from PetscSectionGetFieldDof returning the number elements in that pth plex's stratum. 

Level: intermediate

.seealso: `DMPlexBuildFromDAG()`, `DMPlexBuildFromCellList()`
@*/
PetscErrorCode DMPlexDisjointUnion_Topological_Section(DM *plexlist, PetscInt numplex, DM *dmunion, PetscSection *stratumoff)
{
  PetscInt       p,i,j,k,depth,depth_temp,dim_top,dim_top_temp,pStart,pEnd,chartsize,stratum,totalconesize;
  PetscInt       *numpoints_g, *coneSize_g, *cones_g, *coneOrientations_g,coneSize,off,prevtotal;
  const PetscInt *cone,*coneOrientation;
  DMType         dmtype;
  MPI_Comm       comm,comm_prev;
  PetscSection   offsets;
  char           fieldname[64]; 
  DM             dm_sum;
  PetscBool      flag; 

  PetscFunctionBegin;
  /* input checks */
  if (numplex <= 0) PetscFunctionReturn(0);
  comm_prev = PetscObjectComm((PetscObject)plexlist[0]);
  for (i=0; i<numplex; i++) {
    comm = PetscObjectComm((PetscObject)plexlist[i]);
    PetscCall(DMGetType(plexlist[i],&dmtype));
    PetscCall(PetscStrncmp(dmtype,DMPLEX,sizeof(dmtype),&flag));
    if (!flag) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Wrong DM: %s Object must be DMPlex",dmtype);
    if(i == 0) continue; 
    else {
      PetscCheckSameComm((PetscObject)plexlist[i],1,(PetscObject)plexlist[i],2);
    }
  }
  /* Acquire maximum depth size across all dms and maximum topologial dimension chartsize */
  depth     = 0;
  dim_top   = 0;
  chartsize = 0;
  for(i=0; i<numplex; i++){
    PetscCall(DMPlexGetDepth(plexlist[i],&depth_temp));
    if (depth < depth_temp) depth = depth_temp;
    PetscCall(DMGetDimension(plexlist[i],&dim_top_temp));
    if (dim_top < dim_top_temp) dim_top = dim_top_temp;
    PetscCall(DMPlexGetChart(plexlist[i],&pStart,&pEnd));
    chartsize += (pEnd-pStart);
  }
  
  PetscCall(PetscMalloc1(chartsize,&coneSize_g));
  PetscCall(PetscCalloc1(depth+1,&numpoints_g));
  /* set up the stratum offset section */
  PetscCall(PetscSectionCreate(comm,&offsets));
  PetscCall(PetscSectionSetNumFields(offsets, depth+1));/* one field per stratum */
  PetscCall(PetscSectionSetChart(offsets,0,numplex));
  for (i=0; i<=depth; i++) {
    PetscCall(PetscSectionSetFieldComponents(offsets, i, 1));
    PetscCall(PetscSNPrintf(fieldname,sizeof(fieldname),"Stratum Depth %D",i));
    PetscCall(PetscSectionSetFieldName(offsets,i,fieldname));
  }
  /* Iterate through the meshes and compute the number of points at each stratum */
  for (i=0; i<numplex; i++) {
    PetscCall(DMPlexGetDepth(plexlist[i],&depth_temp));
    PetscCall(PetscSectionSetDof(offsets,i,depth_temp+1));
    for(stratum=0;stratum <= depth_temp; stratum++) {
      PetscCall(PetscSectionSetFieldDof(offsets,i,stratum,1));
      PetscCall(DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd));
      PetscCall(PetscSectionSetFieldOffset(offsets,i,stratum,numpoints_g[stratum]));
      numpoints_g[stratum] += (pEnd-pStart);
    }
  }
  /* Create the cone size information */
  totalconesize = 0;
  for (i=0; i<numplex; i++) {
    PetscCall(DMPlexGetDepth(plexlist[i],&depth_temp));
    for(stratum=0;stratum <= depth_temp; stratum++) {
      prevtotal=0;
      for(j=0; j<stratum; j++) prevtotal += numpoints_g[j];
      PetscCall(DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd));
      PetscCall(PetscSectionGetFieldOffset(offsets,i,stratum,&off));
      for(p=pStart; p<pEnd; p++) {
        PetscCall(DMPlexGetConeSize(plexlist[i],p,&coneSize));
        coneSize_g[p+off+prevtotal] = coneSize;
        totalconesize += coneSize;
      }
    }
  }
 /* create the cone and cone orientations */
  PetscCall(PetscMalloc2(totalconesize,&cones_g,totalconesize,&coneOrientations_g));
  k=0;
  for(stratum=0;stratum <= depth; stratum++) {
    prevtotal=0;
    for(j=0; j<stratum-1; j++) prevtotal += numpoints_g[j];
    for(i=0; i<numplex; i++){
      PetscCall(DMPlexGetDepth(plexlist[i],&depth_temp));
      if (stratum <= depth_temp) {
        if (stratum > 0) { /* stratum = 0 doesn't matter as the cones for stratum = 0 are empty */
          PetscCall(DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd));
          PetscCall(PetscSectionGetFieldOffset(offsets,i,stratum-1,&off));
          for(p=pStart; p<pEnd; p++) {
            PetscCall(DMPlexGetCone(plexlist[i],p,&cone));
            PetscCall(DMPlexGetConeOrientation(plexlist[i],p,&coneOrientation));
            PetscCall(DMPlexGetConeSize(plexlist[i],p,&coneSize));
            for(j=0; j<coneSize; j++) {
              coneOrientations_g[k] = coneOrientation[j];
              cones_g[k++] = cone[j]+off+prevtotal; /* account for the offset in the cone stratum (stratum -1) */
            }
          }
        }
      }
    }
  }

  PetscCall(DMPlexCreate(comm,&dm_sum));
  PetscCall(DMSetDimension(dm_sum,dim_top));
  PetscCall(DMPlexBuildFromDAG(dm_sum,depth,numpoints_g,coneSize_g,cones_g,coneOrientations_g));
  PetscCall(PetscFree(numpoints_g));
  PetscCall(PetscFree(coneSize_g));
  PetscCall(PetscFree2(cones_g,coneOrientations_g));
  *dmunion    = dm_sum;
  *stratumoff = offsets;
  PetscFunctionReturn(0);
}
