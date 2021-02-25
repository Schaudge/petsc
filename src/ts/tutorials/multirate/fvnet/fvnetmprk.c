#include "fvnet.h"

/* Specific multirate partitions for test examples. */
PetscErrorCode FVNetworkGenerateMultiratePartition_Preset(FVNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       id,e,eStart,eEnd,slow_edges_count = 0,fast_edges_count = 0,slow_edges_size = 0,fast_edges_size = 0;
  PetscInt       *slow_edges,*fast_edges;
  FVEdge         fvedge;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  switch (fvnet->networktype) {
    case 0: /* Mark the boundary edges as slow and the middle edge as fast */
    case 2: 
      /* Find the number of slow/fast edges */
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id || id == 2) {
          slow_edges_size++;
        } else if (id == 1) { 
          fast_edges_size++;
        }
      }
      /* Data will be owned and deleted by the IS*/
      ierr = PetscMalloc1(slow_edges_size,&slow_edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(fast_edges_size,&fast_edges);CHKERRQ(ierr);
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id || id == 2) {
          slow_edges[slow_edges_count] = e; 
          slow_edges_count++; 
        } else if (id == 1) { 
          fast_edges[fast_edges_count] = e; 
          fast_edges_count++; 
        }
      }
      /* Generate IS */
      ierr = ISCreateGeneral(MPI_COMM_SELF,slow_edges_size,slow_edges,PETSC_OWN_POINTER,&fvnet->slow_edges);CHKERRQ(ierr);
      ierr = ISCreateGeneral(MPI_COMM_SELF,fast_edges_size,fast_edges,PETSC_OWN_POINTER,&fvnet->fast_edges);CHKERRQ(ierr);
      break; 
    case 1: /* Mark the middle edge as slow */
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id) {
          slow_edges_size++;
        } 
      }
      /* Data will be owned and deleted by the IS*/
      ierr = PetscMalloc1(slow_edges_size,&slow_edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(fast_edges_size,&fast_edges);CHKERRQ(ierr);
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (!id) {
          slow_edges[slow_edges_count] = e; 
          slow_edges_count++; 
        }
      }
      /* Generate IS */
      ierr = ISCreateGeneral(MPI_COMM_SELF,slow_edges_size,slow_edges,PETSC_OWN_POINTER,&fvnet->slow_edges);CHKERRQ(ierr);
      ierr = ISCreateGeneral(MPI_COMM_SELF,fast_edges_size,fast_edges,PETSC_OWN_POINTER,&fvnet->fast_edges);CHKERRQ(ierr);
      break;
    case 3: 
       /* Find the number of slow/fast edges */
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if (id == 1 || id == 2) {
          slow_edges_size++;
        } else if (id == 0) { 
          fast_edges_size++;
        }
      }
      /* Data will be owned and deleted by the IS*/
      ierr = PetscMalloc1(slow_edges_size,&slow_edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(fast_edges_size,&fast_edges);CHKERRQ(ierr);
      for (e=eStart; e<eEnd; e++) {
        ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        id   = fvedge->id; 
        if ( id == 1 || id == 2) {
          slow_edges[slow_edges_count] = e; 
          slow_edges_count++; 
        } else if (id == 0) { 
          fast_edges[fast_edges_count] = e; 
          fast_edges_count++; 
        }
      }
      /* Generate IS */
      ierr = ISCreateGeneral(MPI_COMM_SELF,slow_edges_size,slow_edges,PETSC_OWN_POINTER,&fvnet->slow_edges);CHKERRQ(ierr);
      ierr = ISCreateGeneral(MPI_COMM_SELF,fast_edges_size,fast_edges,PETSC_OWN_POINTER,&fvnet->fast_edges);CHKERRQ(ierr);
      break; 
    default: 
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }
  PetscFunctionReturn(0);
}
/* Builds the vertex lists and marks the edges as buffer zones as neccessary. Assumes that fvnet->fast_edges and fvnet->slow_edges 
   have already been built. */
PetscErrorCode FVNetworkFinalizePartition(FVNetwork fvnet) 
{
  PetscErrorCode ierr;
  PetscInt       i,vfrom,vto,e,eStart,eEnd,v,vStart,vEnd,offsetf,ne,dof = fvnet->physics.dof;
  PetscInt       *buf_vert,*fast_vert,*slow_vert,*vert_sizes,*vert_count;
  PetscInt       buf_count = 0, buf_size = 0,edgemarking,numlvls = 2; /* Number of multirate levels, assumed two (fast/slow) for now */ 
  const PetscInt *cone,*edges;
  Junction       junction;
  FVEdge         fvedge;
  Vec            Ftmp = fvnet->Ftmp, localF = fvnet->localF; /* Used for passing marking data between processors */
  PetscScalar    *f;
  PetscBool      *hasmarkededge; 

  enum {SLOW=1,FAST=2}; /* for readability */
  
  PetscFunctionBegin;
  ierr = VecZeroEntries(Ftmp);CHKERRQ(ierr);
  ierr = VecZeroEntries(localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(fvnet->network,&vStart,&vEnd);CHKERRQ(ierr);
  /* We assume fvnet->slow_edges and fvnet->fast_edges are built. We communicate these edge markings into the 
     the junction edgemarkings data. This allows us to mark a junction as fast/slow/buffer */
  /* copy slow edge data */
  ierr = ISGetIndices(fvnet->slow_edges,&edges);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->slow_edges,&ne);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    e     = edges[i];
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vfrom] = SLOW;
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vto]   = SLOW; 
  }
  ierr = ISGetIndices(fvnet->fast_edges,&edges);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->fast_edges,&ne);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    e     = edges[i];
    ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0];
    vto   = cone[1];
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,vfrom,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vfrom] = FAST;
    ierr  = DMNetworkGetLocalVecOffset(fvnet->network,vto,FLUX,&offsetf);CHKERRQ(ierr);
    f[offsetf+fvedge->offset_vto]   = FAST; 
  }
  /* Now communicate the marking data to all processors */
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(fvnet->network,localF,ADD_VALUES,Ftmp);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(fvnet->network,Ftmp,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);
  /* Allocate size based on number of marking levels possible. Assume two (fast/slow) for now */
  ierr = PetscMalloc1(numlvls,&hasmarkededge);CHKERRQ(ierr);
  ierr = PetscMalloc2(numlvls,&vert_sizes,numlvls,&vert_count);CHKERRQ(ierr);
  for (i=0; i<numlvls; i++) {
    vert_sizes[i] = 0; 
    vert_count[i] = 0; 
  }
  /* Find the sizes of the vertex lists */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (i=0; i<numlvls; i++) {
      hasmarkededge[i] = PETSC_FALSE; 
    } 
    for (i=0; i<junction->numedges; i++) {
      edgemarking                  = f[offsetf+i*dof];
      hasmarkededge[edgemarking-1] = PETSC_TRUE; 
    }
    /* Add to slow/fast/... lists */
    for (i=0; i<numlvls; i++) {
      if (hasmarkededge[i]) vert_sizes[i]++; 
    }
    /* Add to buffer lists as necessary (assumes numlvl == 2 for now ) */
    if (hasmarkededge[0] && hasmarkededge[1]) {
      buf_size++; 
    }
  }
  /* Data will be owned and deleted by the IS */
  ierr = PetscMalloc1(vert_sizes[0],&slow_vert);CHKERRQ(ierr);
  ierr = PetscMalloc1(vert_sizes[1],&fast_vert);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size,&buf_vert);CHKERRQ(ierr);
  /* Build the vertex lists */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponent(fvnet->network,v,JUNCTION,NULL,(void**)&junction,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetLocalVecOffset(fvnet->network,v,FLUX,&offsetf);CHKERRQ(ierr);
    for (i=0; i<numlvls; i++) {
      hasmarkededge[i] = PETSC_FALSE; 
    } 
    for (i=0; i<junction->numedges; i++) {
      edgemarking                  = f[offsetf+i*dof];
      hasmarkededge[edgemarking-1] = PETSC_TRUE; 
    }
    /* Add to slow/fast/... lists (assumes numlvl = 2) */
    if (hasmarkededge[0]) {
      slow_vert[vert_count[0]] = v;
      vert_count[0]++; 
    }
    if (hasmarkededge[1]) {
      fast_vert[vert_count[1]] = v; 
      vert_count[1]++; 
    }    
    /* Add to buffer lists as necessary (assumes numlvl == 2 for now ) */
    if (hasmarkededge[0] && hasmarkededge[1]) {
      buf_vert[buf_count] = v; 
      buf_count++;
      ierr = DMNetworkGetSupportingEdges(fvnet->network,v,&ne,&edges);CHKERRQ(ierr);
      for (i=0; i<ne; i++) { /* Mark the connected slow edges as buffer edges */
        e     = edges[i];
        ierr  = DMNetworkGetConnectedVertices(fvnet->network,e,&cone);CHKERRQ(ierr);
        ierr  = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
        vfrom = cone[0]; 
        vto   = cone[1];
        if (v == vfrom) {
          if (f[offsetf+fvedge->offset_vfrom] == SLOW) fvedge->frombufferlvl = 1;
        } else if (v == vto) {
          if (f[offsetf+fvedge->offset_vto] == SLOW)   fvedge->tobufferlvl   = 1; 
        }
      }
    }
  }
  /* Generate IS */
  ierr = ISCreateGeneral(MPI_COMM_SELF,vert_sizes[0],slow_vert,PETSC_OWN_POINTER,&fvnet->slow_vert);CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,vert_sizes[1],fast_vert,PETSC_OWN_POINTER,&fvnet->fast_vert);CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,buf_size,buf_vert,PETSC_OWN_POINTER,&fvnet->buf_slow_vert);CHKERRQ(ierr);
  /* Free Data */
  ierr = PetscFree2(vert_sizes,vert_count);CHKERRQ(ierr); 
  ierr = PetscFree(hasmarkededge);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* Specifically for slow/slowbuffer/fast multirate partitioning. A general function is needed later. */
PetscErrorCode FVNetworkBuildMultirateIS(FVNetwork fvnet, IS *slow, IS *fast, IS *buffer) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,offset,e,dof = fvnet->physics.dof;
  PetscInt       *i_slow,*i_fast,*i_buf;
  const PetscInt *index; 
  PetscInt       slow_size = 0,fast_size = 0,buf_size = 0,size,bufferwidth = fvnet->bufferwidth; 
  PetscInt       slow_count = 0,fast_count = 0, buf_count = 0; 
  FVEdge         fvedge;

  PetscFunctionBegin; 
  /* Iterate through the marked slow edges */
  ierr = ISGetIndices(fvnet->slow_edges,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->slow_edges,&size);CHKERRQ(ierr);
  /* Find the correct sizes for the arrays */
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr); 
    if (!fvedge->tobufferlvl) {
      slow_size += dof*bufferwidth; 
    } else {
      buf_size  += dof*bufferwidth; 
    }
    if (!fvedge->frombufferlvl) {
      slow_size += dof*bufferwidth; 
    } else {
      buf_size  += dof*bufferwidth; 
    }
    slow_size   += dof*fvedge->nnodes - 2*dof*bufferwidth; 
  }
  ierr = PetscMalloc1(slow_size,&i_slow);CHKERRQ(ierr);
  ierr = PetscMalloc1(buf_size,&i_buf);CHKERRQ(ierr);
  /* Build the set of indices (global data indices) */
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    if (!fvedge->frombufferlvl) {
      for (j=0; j<bufferwidth; j++) {
        for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        }
      }
    } else {
      for (j=0; j<bufferwidth; j++) {
        for (k=0; k<dof; k++) {
          i_buf[buf_count] = offset+j*dof+k; 
          buf_count++;
        }
      }
    }
    for (j=bufferwidth; j<(fvedge->nnodes-bufferwidth); j++){
      for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        }
    }
    if(!fvedge->tobufferlvl) {
      for (j=fvedge->nnodes-bufferwidth; j<fvedge->nnodes; j++) {
        for (k=0; k<dof; k++) {
          i_slow[slow_count] = offset+j*dof+k; 
          slow_count++;
        } 
      }
    } else {
      for (j=fvedge->nnodes-bufferwidth; j<fvedge->nnodes; j++) {
        for (k=0; k<dof; k++) {
          i_buf[buf_count] = offset+j*dof+k; 
          buf_count++;
        }
      } 
    }
  }
  ierr = ISRestoreIndices(fvnet->slow_edges,&index);CHKERRQ(ierr);
  /* Repeat the same procedure for the fast edges. However a fast edge has no buffer data on it so this simplifies a bit. */
  /* Iterate through the marked fast edges */
  ierr = ISGetIndices(fvnet->fast_edges,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(fvnet->fast_edges,&size);CHKERRQ(ierr);
  /* Find the correct sizes for the arrays */
  for (i=0; i<size; i++) {
    e         = index[i];
    ierr      = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr); 
    fast_size += dof*fvedge->nnodes;
  }
  ierr = PetscMalloc1(fast_size,&i_fast);CHKERRQ(ierr); 
  for (i=0; i<size; i++) {
    e    = index[i];
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVecOffset(fvnet->network,e,FVEDGE,&offset);CHKERRQ(ierr);
    for (j=0; j<fvedge->nnodes; j++) {
      for (k=0; k<dof; k++) {
        i_fast[fast_count] = offset+j*dof+k; 
        fast_count++; 
      }
    }
  }
  ierr = ISRestoreIndices(fvnet->fast_edges,&index);CHKERRQ(ierr);
  /* Now Build the index sets */
  ierr = ISCreateGeneral(fvnet->comm,slow_size,i_slow,PETSC_COPY_VALUES,slow);CHKERRQ(ierr); 
  ierr = ISCreateGeneral(fvnet->comm,fast_size,i_fast,PETSC_COPY_VALUES,fast);CHKERRQ(ierr); 
  ierr = ISCreateGeneral(fvnet->comm,buf_size,i_buf,PETSC_COPY_VALUES,buffer);CHKERRQ(ierr); 
  /* Free Data */
  ierr = PetscFree(i_slow);CHKERRQ(ierr);
  ierr = PetscFree(i_fast);CHKERRQ(ierr);
  ierr = PetscFree(i_buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}