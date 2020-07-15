PetscErrorCode FVNetGenerateMultiratePartition_HValue(FVNetwork fvnet,PetscReal hcutoff) {
  PetscErrorCode ierr;
  PetscInt       i,vfrom,vto,type,offset,e,eStart,eEnd,ne,nv,dof = fvnet->physics.dof;
  PetscInt       *vtx,*edge; 
  PetscInt       *slow_edges,*fast_edges,*buf_slow_vert; 
  PetscInt       *slow_vert, *fast_vert;
  PetscInt       slow_edges_size = 0,fast_edges_size = 0,buf_slow_vert_size = 0,slow_vert_size = 0,fast_vert_size = 0;
  PetscScalar    *xarr,*u;
  Junction       junction;
  FVEdge         fvedge;
  const PetscInt *cone;

  PetscFunctionBegin;
  ierr = DMNetworkGetEdgeRange(fvnet->network,&eStart,&eEnd);CHKERRQ(ierr);
  /* First Pass is to count how many entries will be in each array */
  for(e=eStart; e<eEnd; e++){
    ierr = DMNetworkGetComponent(fvnet->network,e,FVEDGE,NULL,(void**)&fvedge);CHKERRQ(ierr);
    if (fvedge->h<hcutoff) {
      /* Mark as fast */
      fast_edges_size++; 
      fast_vert_size += 2; 
    } else if (fvedge->h>hcutoff) {
      
    }
  }

  PetscFunctionReturn(0);
}