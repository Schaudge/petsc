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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* input checks */
  if (numplex <= 0) PetscFunctionReturn(0);
  comm_prev = PetscObjectComm((PetscObject)plexlist[0]);
  for (i=0; i<numplex; i++) {
    comm = PetscObjectComm((PetscObject)plexlist[i]);
    ierr = DMGetType(plexlist[i],&dmtype);CHKERRQ(ierr);
    ierr = PetscStrncmp(dmtype,DMPLEX,sizeof(dmtype),&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Wrong DM: %s Object must be DMPlex",dmtype);
    if(i == 0) continue; 
    else {
      PetscCheckSameComm((PetscObject)plexlist[i],1,(PetscObject)plexlist[i],2);
    }
  }
  /* Acquire maximum depth size and topological dimension across all dms, and total chartsize */
  depth     = 0;
  dim_top   = 0;
  chartsize = 0;
  for(i=0; i<numplex; i++){
    ierr = DMPlexGetDepth(plexlist[i],&depth_temp);CHKERRQ(ierr);
    if (depth < depth_temp) depth = depth_temp;
    ierr = DMGetDimension(plexlist[i],&dim_top_temp);CHKERRQ(ierr);
    if (dim_top < dim_top_temp) dim_top = dim_top_temp;
    ierr = DMPlexGetChart(plexlist[i],&pStart,&pEnd);CHKERRQ(ierr);
    chartsize += (pEnd-pStart);
  }
  
  ierr = PetscMalloc1(chartsize,&coneSize_g);CHKERRQ(ierr);
  ierr = PetscCalloc1(depth+1,&numpoints_g);CHKERRQ(ierr);
  /* set up the stratum offset section */
  ierr = PetscSectionCreate(comm,&offsets);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(offsets, depth+1);CHKERRQ(ierr);/* one field per stratum */
  ierr = PetscSectionSetChart(offsets,0,numplex);CHKERRQ(ierr);
  for (i=0; i<=depth; i++) {
    ierr = PetscSectionSetFieldComponents(offsets, i, 1);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fieldname,sizeof(fieldname),"Stratum Depth %D",i);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(offsets,i,fieldname);CHKERRQ(ierr);
  }
  /* Iterate through the meshes and compute the number of points at each stratum */
  for (i=0; i<numplex; i++) {
    ierr = DMPlexGetDepth(plexlist[i],&depth_temp);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(offsets,i,depth_temp+1);CHKERRQ(ierr);
    for(stratum=0;stratum <= depth_temp; stratum++) {
      ierr = DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(offsets,i,stratum,pEnd-pStart);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldOffset(offsets,i,stratum,numpoints_g[stratum]-pStart);CHKERRQ(ierr);
      numpoints_g[stratum] += (pEnd-pStart);
    }
  }
  /* Create the cone size information */
  totalconesize = 0;
  for (i=0; i<numplex; i++) {
    ierr = DMPlexGetDepth(plexlist[i],&depth_temp);CHKERRQ(ierr);
    for(stratum=0;stratum <= depth_temp; stratum++) {
      prevtotal = 0;
      for(j=0; j<stratum; j++) prevtotal += numpoints_g[j];
      ierr = DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(offsets,i,stratum,&off);CHKERRQ(ierr);
      for(p=pStart; p<pEnd; p++) {
        ierr = DMPlexGetConeSize(plexlist[i],p,&coneSize);CHKERRQ(ierr);
        coneSize_g[p+off+prevtotal] = coneSize;
        totalconesize += coneSize;
      }
    }
  }
 /* create the cone and cone orientations */
  ierr = PetscMalloc2(totalconesize,&cones_g,totalconesize,&coneOrientations_g);CHKERRQ(ierr);
  k=0;
  for(stratum=0;stratum <= depth; stratum++) {
    prevtotal = 0;
    for(j=0; j<stratum-1; j++) prevtotal += numpoints_g[j];
    for(i=0; i<numplex; i++){
      ierr = DMPlexGetDepth(plexlist[i],&depth_temp);CHKERRQ(ierr);
      if (stratum <= depth_temp) {
        if (stratum > 0) { /* stratum = 0 doesn't matter as the cones for stratum = 0 are empty */
          ierr = DMPlexGetDepthStratum(plexlist[i],stratum,&pStart,&pEnd);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(offsets,i,stratum-1,&off);CHKERRQ(ierr);
          for(p=pStart; p<pEnd; p++) {
            ierr = DMPlexGetCone(plexlist[i],p,&cone);CHKERRQ(ierr);
            ierr = DMPlexGetConeOrientation(plexlist[i],p,&coneOrientation);CHKERRQ(ierr);
            ierr = DMPlexGetConeSize(plexlist[i],p,&coneSize);CHKERRQ(ierr);
            for(j=0; j<coneSize; j++) {
              coneOrientations_g[k] = coneOrientation[j];
              cones_g[k++] = cone[j]+off+prevtotal; /* account for the offset in the cone stratum (stratum -1) */
            }
          }
        }
      }
    }
  }

  ierr = DMPlexCreate(comm,&dm_sum);CHKERRQ(ierr);
  ierr = DMSetDimension(dm_sum,dim_top);CHKERRQ(ierr);
  ierr = DMPlexBuildFromDAG(dm_sum,depth,numpoints_g,coneSize_g,cones_g,coneOrientations_g);CHKERRQ(ierr);
  ierr = PetscFree(numpoints_g);CHKERRQ(ierr);
  ierr = PetscFree(coneSize_g);CHKERRQ(ierr);
  ierr = PetscFree2(cones_g,coneOrientations_g);CHKERRQ(ierr);
  *dmunion    = dm_sum;
  if( stratumoff ) {
    *stratumoff = offsets;
  } else {
    ierr = PetscSectionDestroy(&offsets);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*@
  VecSectionCopy - Copies between a reduced vector and the appropriate elements of a full-space vector.

  Input Parameters:
+ vfull       - the full-space vector
. fullsec     - the section for the full vector space
. is          - An IS that maps the elements of the subsec chart to section for the full-space. 
                is[psub-pStart] = pfull for points psub in the subsec chart
. mode        - the direction of copying, SCATTER_FORWARD or SCATTER_REVERSE
. subsec      - the section for the reduced space
- vsub        - the reduced-space vector

  Notes:
    This is intended to be used in conjunction with PetscSectionCreateSubmeshSection() and other related
    functions. Providing a means to easily map vectors from submeshes to meshes and back, without having 
    to generate an intermediary IS. 

  Level: advanced

.seealso: `VecISSet()`, `VecISAXPY()`, `VecCopy()`, `PetscSectionCreateSubmeshSection()`
@*/

PetscErrorCode VecSectionCopy(Vec vfull,PetscSection fullsec, IS is, ScatterMode mode, PetscSection subsec, Vec vsub)
{
  PetscInt       nfull, nsub;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
  PetscValidHeaderSpecific(fullsec,PETSC_SECTION_CLASSID,2);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  PetscValidHeaderSpecific(subsec,PETSC_SECTION_CLASSID,5);
  PetscValidHeaderSpecific(vsub,VEC_CLASSID,6);
  ierr = VecGetSize(vfull, &nfull);CHKERRQ(ierr);
  ierr = VecGetSize(vsub, &nsub);CHKERRQ(ierr);

  if (nfull == nsub) { /* Also takes care of masked vectors */
    if (mode == SCATTER_FORWARD) {
      ierr = VecCopy(vsub, vfull);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(vfull, vsub);CHKERRQ(ierr);
    }
  } else {
    const PetscInt *id;
    PetscInt        p,i, n, m, pStart, pEnd,pfullStart,pfullEnd,subdof,fulldof,fulloff,suboff; 

    ierr = ISGetIndices(is, &id);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
    ierr = VecGetLocalSize(vsub, &m);CHKERRQ(ierr);
    /* Add debug only check that subsec is compatible with the vsub vec. That is the range of the section is [0,m), the domain 
    of vsub. Should be debug only as it will not be a "free" check */

    ierr = PetscSectionGetChart(subsec,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(fullsec,&pfullStart,&pfullEnd);CHKERRQ(ierr);
    //PetscCheck(pEnd-pStart == n,PETSC_COMM_SELF, PETSC_ERR_SUP, "IS local length %" PetscInt_FMT " not equal to Subsection Chart Size %" PetscInt_FMT, n, pEnd-pStart);
    id+=pStart; 
    if (mode == SCATTER_FORWARD) {
      PetscScalar       *y;
      const PetscScalar *x;

      ierr = VecGetArray(vfull, &y);CHKERRQ(ierr);
      ierr = VecGetArrayRead(vsub, &x);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        //PetscCheck(id[p] >= pfullStart && id[p] < pfullEnd,PETSC_COMM_SELF, PETSC_ERR_SUP, "The range of IS needs to be in the chart of fullsec. \n Subsec p: % " PetscInt_FMT "is[p]: % " PetscInt_FMT "fullsec chart start: % " PetscInt_FMT  " fullsec chart end: %"PetscInt_FMT, p,id[p],pfullStart,pfullEnd);
        ierr = PetscSectionGetDof(subsec,p,&subdof);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(fullsec,id[p],&fulldof);CHKERRQ(ierr);
        //PetscCheck(subdof == fulldof,PETSC_COMM_SELF,PETSC_ERR_SUP,"Nonequal dofs at corresponding points. \nsubsec has % " PetscInt_FMT " dofs at point %" PetscInt_FMT "\nfullsec has % " PetscInt_FMT "dofs at corresponding point % " PetscInt_FMT, p,subdof,id[p],fulldof);
        ierr = PetscSectionGetOffset(subsec,p,&suboff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(fullsec,id[p],&fulloff);CHKERRQ(ierr);
        for(i=0; i<subdof; i++) {
          y[fulloff+i] = x[suboff+i];
        }
      }
      ierr = VecRestoreArrayRead(vsub, &x);CHKERRQ(ierr);
      ierr = VecRestoreArray(vfull, &y);CHKERRQ(ierr);
    } else if (mode == SCATTER_REVERSE) {
      PetscScalar       *x;
      const PetscScalar *y;

      ierr = VecGetArrayRead(vfull, &y);CHKERRQ(ierr);
      ierr = VecGetArray(vsub, &x);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        //PetscCheck(id[p] >= pfullStart && id[p] < pfullEnd,PETSC_COMM_SELF, PETSC_ERR_SUP, "The range of IS needs to be in the chart of fullsec. \n Subsec p: % " PetscInt_FMT "is[p]: % " PetscInt_FMT "fullsec chart start: % " PetscInt_FMT  " fullsec chart end: %"PetscInt_FMT, p,id[p],pfullStart,pfullEnd);
        ierr = PetscSectionGetDof(subsec,p,&subdof);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(fullsec,id[p],&fulldof);CHKERRQ(ierr);
        //PetscCheck(subdof == fulldof,PETSC_COMM_SELF,PETSC_ERR_SUP,"Nonequal dofs at corresponding points. \nsubsec has % " PetscInt_FMT " dofs at point %" PetscInt_FMT "\nfullsec has % " PetscInt_FMT "dofs at corresponding point % " PetscInt_FMT, p,subdof,id[p],fulldof);
        ierr = PetscSectionGetOffset(subsec,p,&suboff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(fullsec,id[p],&fulloff);CHKERRQ(ierr);
        for(i=0; i<subdof; i++) {
          x[suboff+i] = y[fulloff+i];
        }
      }
      ierr = VecRestoreArray(vsub, &x);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(vfull, &y);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject) vfull), PETSC_ERR_ARG_WRONG, "Only forward or reverse modes are legal");
    id -= pStart; 
    ierr = ISRestoreIndices(is, &id);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexDisjointUnion_Geometric- Computes the disjoint union of an array of DMPlexs including the geometry

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
PetscErrorCode DMPlexDisjointUnion_Geometric_Section(DM *plexlist, PetscInt numplex, DM *dmunion, PetscSection *stratumoff)
{
  PetscInt       p,i,j,depth,pStart,pEnd,punionStart,punionEnd,stratum,off;
  PetscInt       maxorder = 0,formdegree,order,dim,dE,dEold,secStart,secEnd,off_stratum,off_union,dof; 
  PetscSection   coordsec_union,coordsec,offsets;
  DM             cdm, dm, cdmOld, dm_union,cdm_new;
  PetscBool      simplex;
  PetscClassId   classid;
  PetscFE        discOld,disc;
  PetscDualSpace dualspace;
  Vec            Coordunion,Coord;
  PetscScalar    *coordunion; 
  const PetscScalar *coord;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexDisjointUnion_Topological_Section(plexlist,numplex,&dm_union,&offsets);CHKERRQ(ierr);
  *dmunion = dm_union;
  /* 
    check that all geometries are described by PetscFE or implicit in the same derahm complex. If so, 
    a unifying PetscFE with maximum order amongst all geometries is created and all geometries are projected into it. 
  */
  for (i=0; i<numplex; i++) {
    dm = plexlist[i];
    ierr = DMGetCoordinateDM(dm, &cdmOld);CHKERRQ(ierr);
    /* Check current discretization is compatible */
    ierr = DMGetField(cdmOld, 0, NULL, (PetscObject*)&discOld);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId((PetscObject)discOld, &classid);CHKERRQ(ierr);
    if (classid != PETSCFE_CLASSID) {
      if (classid == PETSC_CONTAINER_CLASSID) {
        PetscFE        feLinear;

        /* Assume linear vertex coordinates */
        ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
        ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
        ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
        ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 1, PETSC_DETERMINE, &feLinear);CHKERRQ(ierr);
        ierr = DMSetField(cdmOld, 0, NULL, (PetscObject) feLinear);CHKERRQ(ierr);
        ierr = PetscFEDestroy(&feLinear);CHKERRQ(ierr);
        ierr = DMCreateDS(cdmOld);CHKERRQ(ierr);
        ierr = DMGetField(cdmOld, 0, NULL, (PetscObject*)&discOld);CHKERRQ(ierr);
      } else {
        const char *discname;

        ierr = PetscObjectGetType((PetscObject)discOld, &discname);CHKERRQ(ierr);
        SETERRQ1(PetscObjectComm((PetscObject)discOld), PETSC_ERR_SUP, "Discretization type %s not supported", discname);
      }
    }
    ierr = DMGetCoordinateDim(dm,&dE);CHKERRQ(ierr);
    if(i>0) {
      if(dE != dEold) {
        SETERRQ4(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Non-Equivalent geometric dimensions not supported, DM % " PetscInt_FMT "has dimension % " PetscInt_FMT " but DM % " PetscInt_FMT "has dimension %" PetscInt_FMT,i,dE,i-1,dEold);
      }
    }
    dEold = dE; 
    ierr = PetscFEGetDualSpace(discOld,&dualspace);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetOrder(dualspace,&order);CHKERRQ(ierr);
    maxorder = maxorder > order ? maxorder : order; 
    ierr = PetscDualSpaceGetDeRahm(dualspace,&formdegree);CHKERRQ(ierr);
    if(formdegree != 0) {
      SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only supports H^1 geometries currently, DM % "PetscInt_FMT "has formdegree % " PetscInt_FMT, i,formdegree); 
    }
  }
  /* create coordinates on the union dm */ 
  /* Assumes unified geometric dimension and H^1 geometry */
  ierr = DMGetDimension(*dmunion,&dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*dmunion,dE);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*dmunion,&cdm);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(*dmunion, &simplex);CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, maxorder, PETSC_DETERMINE, &disc);CHKERRQ(ierr);
  /* create a fresh clone of cdm to replace it. It appears that some default behavior in cdm is causing issues, it defaults 
  to a "Null" localsection, which causes the coordinate section to return nonsense for dmunion */

  ierr = DMClone(cdm,&cdm_new);CHKERRQ(ierr);
  ierr = DMSetField(cdm_new, 0, NULL, (PetscObject) disc);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&disc);CHKERRQ(ierr);
  ierr = DMCreateDS(cdm_new);CHKERRQ(ierr);
  ierr = DMGetField(cdm_new, 0, NULL, (PetscObject*)&disc);CHKERRQ(ierr);    
  ierr = DMSetCoordinateField(*dmunion, NULL);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(*dmunion, cdm_new);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(*dmunion,&coordsec_union);CHKERRQ(ierr); 
  ierr = DMCreateGlobalVector(cdm_new,&Coordunion);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm_new);CHKERRQ(ierr);
  ierr = VecGetArray(Coordunion,&coordunion);CHKERRQ(ierr);
  /* project the coordinate onto the new unifying PetscFE disc, and then map the coordinates to union dm */ 
  for (i=0; i<numplex; i++) {
    dm = plexlist[i];
    ierr = DMGetCoordinateDM(dm, &cdmOld);CHKERRQ(ierr);
    ierr = DMProjectCoordinates(dm,disc);CHKERRQ(ierr);
    /* now map the coordinates to the dmunion, knowing that both have the same coordinate discretization */
    ierr = DMGetCoordinates(dm,&Coord);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Coord,&coord);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm,&coordsec);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(coordsec,&secStart,&secEnd);CHKERRQ(ierr);
    /* Iterate through the stratums */
    ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
    for (stratum = 0; stratum <= depth; stratum++) {
      ierr = DMPlexGetDepthStratum(dm,stratum,&pStart,&pEnd);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(*dmunion,stratum,&punionStart,&punionEnd);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(offsets,i,stratum,&off_stratum);CHKERRQ(ierr);
      for (p=pStart;p<pEnd&&p<secEnd;p++) {
        if( p >= secStart) {
          ierr = PetscSectionGetFieldOffset(coordsec,p,0,&off);CHKERRQ(ierr); /* domain offset */
          ierr = PetscSectionGetFieldDof(coordsec,p,0,&dof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(coordsec_union,p+off_stratum+punionStart,0,&off_union);CHKERRQ(ierr); /*range offset */
          for (j=0; j<dof;j++) {
            coordunion[off_union+j] = coord[off+j];
          }
        }
      }
    }
    ierr = VecRestoreArrayRead(Coord,&coord);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(Coordunion,&coordunion);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*dmunion,Coordunion);CHKERRQ(ierr);
  ierr = VecDestroy(&Coordunion);CHKERRQ(ierr);
  if( stratumoff ) {
    *stratumoff = offsets;
  } else {
    ierr = PetscSectionDestroy(&offsets);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscSectionCreateDomainSuperSection - Create a Supersection from list of sections, along with domain mapping sections. 

  Input Parameters:
+ sectionlist - A len long array of petsctionsection to create a supersection from 
. len - The number of input sections 
. domain - A petscsection that selects the subcharts from the section list to apply the offsets
offsets from offset section to. The chart of this section is [0,len) and indexes the sections from section list. Each field contains a seperate subchart of 
for the section, and matches an offset from the offset. 
. offset  - A petscsection that determines how the charts from sectionlist[i] map into the supersection. Each field applies a seperate 
offset to the subcharts given by domain, each point refers to a different section from sectionlist. 

  Output Parameters:
+ supersection - The supersection formed from the section list, with charts mapped by domain, offset. 

Level: intermediate

Notes: Let $S_i$ denote section $i$ from sectionlist, $D$ the domain section and $O$ the offset section. Then the chart of $S_i$ is mapped to the chart 
of the supersection $S$ by the following: 

$S_i  \left ( [D^f(i).offset,D^f(i).offset + D^f(i).dof) \right) \to [O^f(i).offset, O^f(i).offset+ O^f.dof )$

where $[O^f(i).offset, O^f(i).offset+ O^f.dof )$ is subchart of $S$, and superscript $f$ refers to the $fth$ field of the sections. It is assumed 
that $O$ and $D$ are compatible, in that each (point,field) $(i,f)$, has the same number of dofs.

The resulting supersection $S$ will then maintain all data associated with the input sections points mapped to these new points in the supersection. 

Note also the $D$ is not specifically required to cover the input section charts, but any points not contained in $[D^f(i).offset,D^f(i).offset + D^f(i).dof)$, 
will have their components of the section not appear in the supersection.


Currently missing permutation migration and sf migration. LOCAL ONLY FOR NOW!!!!!!!!

Intended to be used with subdm, and superdm functions. Specififcally designed for use with DMPlexDisjointUnion_Topological_Section to map sections
from the input dms to the union dm. 

.seealso: ` DMPlexDisjointUnion_Topological_Section `
@*/
// PetscErrorCode  PetscSectionCreateDomainSuperSection(PetscSection *sectionlist, PetscInt len, PetscSection domain, PetscSection offset, PetscSection *supersection)
// {
//   PetscInt       p,i,j,k,depth,depth_temp,dim_top,dim_top_temp,pStart,pEnd,chartsize,stratum,totalconesize;
//   PetscInt       *numpoints_g, *coneSize_g, *cones_g, *coneOrientations_g,coneSize,off,prevtotal;
//   const PetscInt *cone,*coneOrientation;
//   DMType         dmtype;
//   MPI_Comm       comm,comm_prev;
//   PetscSection   offsets;
//   char           fieldname[64]; 
//   DM             dm_sum;
//   PetscBool      flag; 

//   PetscFunctionBegin;
  
//   PetscFunctionReturn(0);
// }

// Not sure if the above is actually needed.... I can get away without it for now I think. I only need this for visualization for now, and the 
// coordinate spaces can be generated by setting fe fields on the generated union dm. 