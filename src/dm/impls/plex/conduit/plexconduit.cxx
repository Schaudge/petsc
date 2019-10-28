#include <petsc/private/dmpleximpl.h>
#include <petsc/private/viewerconduitimpl.h>
#include <petsc/private/viewerimpl.h>
#include <conduit/conduit_blueprint.hpp>

static PetscErrorCode GetConduitTopologyName(PetscInt dim, PetscInt nvertex, const char **toponame)
{
  PetscFunctionBegin;
  switch (100*dim + nvertex) {
  case 203: *toponame = "tri"; break;
  case 204: *toponame = "quad"; break;
  case 304: *toponame = "tet"; break;
  case 308: *toponame = "hex"; break;
  default: SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for element of dimension %D with %D vertices",dim,nvertex);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Local_Conduit(Vec X,PetscViewer viewer) {
  PetscViewer_Conduit *cd = (PetscViewer_Conduit*)viewer->data;
  PetscErrorCode ierr;
  DM dm;
  PetscInt cycle,*elemcount,c,cStart,cEnd,v,vStart,vEnd,connsize,nelem,xlen,topo_dim,coord_dim;
  const PetscScalar *x;
  const PetscInt maxverts = 28; /* One more than max number of vertices per element (hex=27) */
  PetscReal time;
  PetscMPIInt rank;

  PetscFunctionBegin;
  if (!cd->mesh) {
    cd->mesh = new conduit::Node();
  }
  conduit::Node &mesh = *cd->mesh;

  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&topo_dim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&X);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm,&coord_dim);CHKERRQ(ierr);
  ierr = VecGetSize(X,&xlen);CHKERRQ(ierr);
  if (xlen % coord_dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Coordinate array of length %D not divisible by coordinate dimension %D",xlen,coord_dim);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  conduit::Node &coords = mesh["coordsets/coords"];
  coords["type"] = "explicit";
  coords["values/x"].set(x,xlen/coord_dim,0*sizeof(x[0]),coord_dim*sizeof(x[0]));
  if (coord_dim > 1) coords["values/y"].set(x,xlen/coord_dim,1*sizeof(x[0]),coord_dim*sizeof(x[0]));
  if (coord_dim > 2) coords["values/z"].set(x,xlen/coord_dim,2*sizeof(x[0]),coord_dim*sizeof(x[0]));
  if (coord_dim > 3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Coordinate dimension %D > 3",coord_dim);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  mesh["topologies/mesh/type"] = "unstructured";
  mesh["topologies/mesh/coordset"] = "coords";
  conduit::Node &elems = mesh["topologies/mesh/elements"];
  ierr = PetscCalloc1(maxverts, &elemcount);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    PetscInt closureSize, *closure, v, nverts;
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, (closure=NULL, &closure));CHKERRQ(ierr);
    for (v=nverts=0; v<2*closureSize; v+=2) {
      if (vStart <= closure[v] && closure[v] < vEnd) nverts++;
    }
    elemcount[nverts]++;
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }

  if (0) {
    { /* Create streams for each element type */
      conduit::Node &element_types = elems["element_types"];
      for (v=0,nelem=0,connsize=0; v<maxverts; v++) {
        if (elemcount[v]) {
          const char *toponame;
          PetscInt cellType;
          ierr = GetConduitTopologyName(topo_dim,v,&toponame);CHKERRQ(ierr);
          ierr = DMPlexVTKGetCellType_Internal(dm,topo_dim,v,&cellType);CHKERRQ(ierr);
          //conduit::Node &stream = element_types[toponame];
          conduit::Node &stream = element_types["quads"];
          stream["stream_id"] = cellType;
          stream["shape"] = toponame;
          nelem += elemcount[v];
          connsize += elemcount[v] * v;
        }
      }
    }

    { /* Create element connectivities */
      PetscInt *offsets,*stream_ids,*connectivity;
      ierr = PetscMalloc3(nelem,&offsets,nelem,&stream_ids,connsize,&connectivity);CHKERRQ(ierr);
      for (connsize=0,c=cStart; c<cEnd; c++) {
        PetscInt closureSize, *closure, v, nverts, cellType;
        ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, (closure=NULL, &closure));CHKERRQ(ierr);
        for (v=nverts=0; v<2*closureSize; v+=2) {
          if (vStart <= closure[v] && closure[v] < vEnd) nverts++;
        }
        ierr = DMPlexVTKGetCellType_Internal(dm,topo_dim,nverts,&cellType);CHKERRQ(ierr);
        stream_ids[c-cStart] = cellType;
        offsets[c-cStart] = connsize;
        for (v=0; v<2*closureSize; v+=2) {
          if (vStart <= closure[v] && closure[v] < vEnd) connectivity[connsize++] = closure[v] - vStart;
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      elems["element_index/stream_ids"].set(stream_ids, nelem);
      elems["element_index/offsets"].set(offsets, nelem);
      elems["stream"].set(connectivity, connsize);
      ierr = PetscFree3(offsets,stream_ids,connectivity);CHKERRQ(ierr);
    }
  } else {
    const char *toponame = NULL;
    for (v=0,connsize=0; v<maxverts; v++) {
      if (elemcount[v]) {
        if (!toponame) {ierr = GetConduitTopologyName(topo_dim,v,&toponame);CHKERRQ(ierr);}
        connsize += elemcount[v] * v;
      }
    }
    elems["shape"] = toponame;
    elems["connectivity"].set(conduit::DataType::int32(connsize));
    int32_t *connectivity = elems["connectivity"].value();

    for (connsize=0,c=cStart; c<cEnd; c++) {
      PetscInt closureSize, *closure, v;
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, (closure=NULL, &closure));CHKERRQ(ierr);
      for (v=0; v<2*closureSize; v+=2) {
        if (vStart <= closure[v] && closure[v] < vEnd) connectivity[connsize++] = closure[v] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(elemcount);CHKERRQ(ierr);

  conduit::Node &fields = mesh["fields"];
  conduit::Node &elem_id = fields["vertex"];
  elem_id["association"] = "vertex";
  elem_id["type"] = "scalar";
  elem_id["topology"] = "mesh";
  //elem_id["volume_dependent"] = "false";
  elem_id["values"].set(conduit::DataType::float64(vEnd-vStart));
  double *ids = elem_id["values"].value();
  for (v=vStart; v<vEnd; v++) {
    ids[v-vStart] = 1.*v;
  }

  // conduit::blueprint::mesh::examples::braid("quads", 3, 3, 0, mesh); mesh.print();
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm,&cycle,&time);CHKERRQ(ierr);
  mesh["state/cycle"] = cycle;
  mesh["state/time"] = time;
  mesh["state/domain_id"] = rank;
  PetscFunctionReturn(0);
}
