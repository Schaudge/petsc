#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

typedef struct _UserCtx *User;
struct _UserCtx {
  DM        pack;
  Vec       U1loc,U2loc;
  PetscReal eddydiff1,eddydiff2;
  PetscReal diff1,diff2,rho1,c1,rho2,c2,hx1,hx2;
  PetscInt  m1,m2;
};

static PetscErrorCode FormRHSFunctionLocal1(User user,DMDALocalInfo *info,const PetscScalar u1[],PetscScalar f1[])
{
  PetscReal hx1 = user->hx1,hx2 = user->hx2,rho1 = user->rho1,rho2 = user->rho2,c1 = user->c1,c2 = user->c2,eddydiff1 = user->eddydiff1,diff1 = user->diff1,diff2 = user->diff2;
  PetscInt  i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == 0) f1[i] = eddydiff1*(u1[i+1] - 2.0*u1[i])/(hx1*hx1); /* zero boundary */
    else if (i == info->mx-1) { /* interface, assume the data is already in the ghost point */
      PetscScalar iflux1,iflux2;
      iflux1 = diff1*(u1[i] - u1[i-1])/hx1;
      iflux2 = diff2*(u1[i+1] - u1[i])/hx2;
      f1[i] = 2*(iflux1-iflux2)/(rho1*c1*hx1+ rho2*c2*hx2);
    } else f1[i] = eddydiff1*(u1[i-1] +  u1[i+1] - 2*u1[i])/(hx1*hx1);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunctionLocal2(User user,DMDALocalInfo *info,const PetscScalar u2[],PetscScalar f2[])
{
  PetscReal hx2 = user->hx2,eddydiff2 = user->eddydiff2;
  PetscInt  i;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    if (i == info->mx-1) f2[i] = eddydiff2*(u2[i-1] - 2.0*u2[i])/(hx2*hx2); /* zero boundary */
    else f2[i] = eddydiff2*(u2[i-1] + u2[i+1] - 2.0*u2[i])/(hx2*hx2);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction_All(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  User           user = (User)ctx;
  DM             da1,da2;
  DMDALocalInfo  info1,info2;
  VecScatter     scatter1,scatter2;
  IS             isin1to2,isout1to2,isin2to1,isout2to1;
  PetscScalar    *u1loc,*u2loc;
  PetscScalar    *f1,*f2;
  Vec            U1,U2,U1loc,U2loc,F1,F2;
  PetscInt       isize1,isize2,idx1[1],idx2[1];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&da1,&da2);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->pack,X,&U1,&U2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->pack,&U1loc,&U2loc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->pack,X,U1loc,U2loc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da1,U1loc,&u1loc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,U2loc,&u2loc);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->pack,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da1,F1,&f1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,F2,&f2);CHKERRQ(ierr);

  /* scatter from U1 interface to U2local boundary */
  isize1 = 0;
  isize2 = 0;
  if (info2.xs == 0) { /* first rank on DM 2 */
    idx2[isize2++] = info2.xs-info2.gxs-1; /* ghost boundary on DM 2, local index */
    idx1[isize1++] = info1.mx-1; /* interface node on DM 1, global index */
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,isize1,idx1,PETSC_COPY_VALUES,&isin1to2);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,isize2,idx2,PETSC_COPY_VALUES,&isout1to2);CHKERRQ(ierr);
  ierr = VecScatterCreate(U1,isin1to2,U2loc,isout1to2,&scatter1);CHKERRQ(ierr);

  /* scatter from U2 interface to U1local ghost boundary */
  isize1 = 0;
  isize2 = 0;
  if (info1.xs+info1.xm == info1.mx) { /* last rank on DM 1 */
    idx1[isize1++] = info1.xm+info1.xs-info1.gxs; /* ghost boundary on DM 1, local index */
    idx2[isize2++] = 0; /* interface node on DM2, global index */
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,isize1,idx1,PETSC_COPY_VALUES,&isout2to1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,isize2,idx2,PETSC_COPY_VALUES,&isin2to1);CHKERRQ(ierr);
  ierr = VecScatterCreate(U2,isin2to1,U1loc,isout2to1,&scatter2);CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter1,U1,U2loc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter1,U1,U2loc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter2,U2,U1loc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter2,U2,U1loc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter1);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter2);CHKERRQ(ierr);
  ierr = ISDestroy(&isin1to2);CHKERRQ(ierr);
  ierr = ISDestroy(&isout1to2);CHKERRQ(ierr);
  ierr = ISDestroy(&isin2to1);CHKERRQ(ierr);
  ierr = ISDestroy(&isout2to1);CHKERRQ(ierr);

  ierr = FormRHSFunctionLocal1(user,&info1,u1loc,f1);CHKERRQ(ierr);
  ierr = FormRHSFunctionLocal2(user,&info2,u2loc,f2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,F1,&f1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,F2,&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(user->pack,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,U1loc,&u1loc);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,U2loc,&u2loc);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->pack,&U1loc,&U2loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobianLocal_J11(User user,DMDALocalInfo *info,const PetscScalar u1[],Mat J11)
{
  PetscReal      hx1 = user->hx1,hx2 = user->hx2,rho1 = user->rho1,rho2 = user->rho2,c1 = user->c1,c2 = user->c2,eddydiff1 = user->eddydiff1,diff1 = user->diff1,diff2 = user->diff2;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row = i-info->gxs;
    if (i == 0) {
      PetscInt    cols[] = {row,row+1};
      PetscScalar vals[] = {-2.0*eddydiff1/(hx1*hx1),eddydiff1/(hx1*hx1)};
      ierr = MatSetValuesLocal(J11,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == info->mx-1) {
      PetscInt    cols[] = {row-1,row};
      PetscScalar vals[] = {-2.0*diff1/hx1/(rho1*c1*hx1+rho2*c2*hx2),2.0*(diff1/hx1+diff2/hx2)/(rho1*c1*hx1+rho2*c2*hx2)};
      ierr = MatSetValuesLocal(J11,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt    cols[] = {row-1,row,row+1};
      PetscScalar vals[] = {eddydiff1/(hx1*hx1),-2.0*eddydiff1/(hx1*hx1),eddydiff1/(hx1*hx1)};
      ierr = MatSetValuesLocal(J11,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobianLocal_J22(User user,DMDALocalInfo *info,const PetscScalar u2[],Mat J22)
{
  PetscReal      hx2 = user->hx2,eddydiff2 = user->eddydiff2;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    PetscInt    row = i-info->gxs;
    if (i == 0) {
      PetscInt    cols[] = {row,row+1};
      PetscScalar vals[] = {-2.0*eddydiff2/(hx2*hx2),eddydiff2/(hx2*hx2)};
      ierr = MatSetValuesLocal(J22,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == info->mx-1) {
      PetscInt    cols[] = {row-1,row};
      PetscScalar vals[] = {eddydiff2/(hx2*hx2),-2.0*eddydiff2/(hx2*hx2)};
      ierr = MatSetValuesLocal(J22,1,&row,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt    cols[] = {row-1,row,row+1};
      PetscScalar vals[] = {eddydiff2/(hx2*hx2),-2.0*eddydiff2/(hx2*hx2),eddydiff2/(hx2*hx2)};
      ierr = MatSetValuesLocal(J22,1,&row,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  info1: row DM
  info2: column DM
*/
static PetscErrorCode FormRHSJacobianLocal_J12(User user,DMDALocalInfo *info1,DMDALocalInfo *info2,const PetscScalar u1[],Mat J12)
{
  PetscReal      hx1 = user->hx1,hx2 = user->hx2,rho1 = user->rho1,rho2 = user->rho2,c1 = user->c1,c2 = user->c2,diff2 = user->diff2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!J12) PetscFunctionReturn(0); /* Not assembling this block */

  if (info1->xs+info1->xm == info1->mx) {
    PetscInt    row = info1->mx-1-info1->gxs,col = info2->gxm;
    PetscScalar val = -2.0*diff2/hx2/(rho1*c1*hx1+rho2*c2*hx2);
    ierr = MatSetValuesLocal(J12,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  info2: row DM
  info1: column DM
*/
static PetscErrorCode FormRHSJacobianLocal_J21(User user,DMDALocalInfo *info2,DMDALocalInfo *info1,const PetscScalar u2[],Mat J21)
{
  PetscReal      hx2 = user->hx2,eddydiff2 = user->eddydiff2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!J21) PetscFunctionReturn(0); /* Not assembling this block */

  if (info2->xs == 0) {
    PetscInt    row = info2->xs-info2->gxs,col = info1->gxm;
    PetscScalar val = eddydiff2/(hx2*hx2);
    ierr = MatSetValuesLocal(J21,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModifyCouplingSubMatrixL2G(User user,Mat J)
{
  DM                     da1,da2;
  DMDALocalInfo          info1,info2;
  IS                     *is;
  Mat                    J12,J21;
  ISLocalToGlobalMapping rl2g,cl2g,newcl2g;
  const PetscInt         *idx;
  PetscInt               *newidx,i,nlocal;
  PetscErrorCode         ierr;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&da1,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalISs(user->pack,&is);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(J,is[0],is[1],&J12);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(J,is[1],is[0],&J21);CHKERRQ(ierr);

  if (info1.xs+info1.xm == info1.mx) {
    ierr = MatGetLocalToGlobalMapping(J12,&rl2g,&cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(cl2g,NULL);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(cl2g,&nlocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nlocal+1,&newidx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idx);CHKERRQ(ierr);
    for (i=0; i<nlocal; i++) newidx[i] = idx[i];
    newidx[nlocal] = user->m1;
    ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,nlocal+1,newidx,PETSC_OWN_POINTER,&newcl2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(J12,rl2g,newcl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(newcl2g,NULL);CHKERRQ(ierr);
  }

  if (info2.xs == 0) {
    ierr = MatGetLocalToGlobalMapping(J21,&rl2g,&cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(cl2g,NULL);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(cl2g,&nlocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nlocal+1,&newidx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(cl2g,&idx);CHKERRQ(ierr);
    for (i=0; i<nlocal; i++) newidx[i] = idx[i];
    newidx[nlocal] = user->m1-1;
    ierr = ISLocalToGlobalMappingRestoreIndices(cl2g,&idx);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,nlocal+1,newidx,PETSC_OWN_POINTER,&newcl2g);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(J21,rl2g,newcl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(newcl2g,NULL);CHKERRQ(ierr);
  }
  ierr = MatRestoreLocalSubMatrix(J,is[0],is[1],&J12);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(J,is[1],is[0],&J21);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian_All(TS ts,PetscReal t,Vec X,Mat J,Mat JP,void *ctx)
{
  User           user = (User)ctx;
  DM             da1,da2;
  DMDALocalInfo  info1,info2;
  PetscBool      nest;
  IS             *is;
  Vec            U1loc,U2loc;
  Mat            J11,J12,J21,J22;
  PetscScalar    *u1,*u2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&da1,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->pack,&U1loc,&U2loc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->pack,X,U1loc,U2loc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da1,U1loc,&u1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,U2loc,&u2);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalISs(user->pack,&is);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(JP,is[0],is[0],&J11);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(JP,is[0],is[1],&J12);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(JP,is[1],is[0],&J21);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(JP,is[1],is[1],&J22);CHKERRQ(ierr);
  ierr = FormRHSJacobianLocal_J11(user,&info1,u1,J11);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)JP,MATNEST,&nest);CHKERRQ(ierr);
  if (!nest) {
    /*
       DMCreateMatrix_Composite()  for a nested matrix does not generate off-block matrices that one can call MatSetValuesLocal() on, it just creates dummy
       matrices with no entries; there cannot insert values into them. Commit b6480e041dd2293a65f96222772d68cdb4ed6306
       changed Mat_Nest() from returning NULL pointers for these submatrices to dummy matrices because PCFIELDSPLIT could not
       handle the returned null matrices.
    */
    ierr = FormRHSJacobianLocal_J12(user,&info1,&info2,u1,J12);CHKERRQ(ierr);
    ierr = FormRHSJacobianLocal_J21(user,&info2,&info1,u2,J21);CHKERRQ(ierr);
  }
  ierr = FormRHSJacobianLocal_J22(user,&info2,u2,J22);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(JP,is[0],is[0],&J11);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(JP,is[0],is[1],&J12);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(JP,is[1],is[0],&J21);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(JP,is[1],is[1],&J22);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,U1loc,&u1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,U2loc,&u2);CHKERRQ(ierr);

  ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
  ierr = PetscFree(is);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(user->pack,&U1loc,&U2loc);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (JP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != JP) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitial_Coupled(User user,Vec U)
{
  DM             da1,da2;
  DMDALocalInfo  info1,info2;
  Vec            U1,U2;
  PetscScalar    *u1,*u2,hx1 = user->hx1,hx2 = user->hx2;
  PetscInt       i;
  PetscReal      sigma = 500.0;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCompositeGetEntries(user->pack,&da1,&da2);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->pack,U,&U1,&U2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da1,U1,&u1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,U2,&u2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  for (i=info1.xs; i<info1.xs+info1.xm; i++) {
    u1[i] = 50.0*PetscExpReal(-0.5*(1.0-i*hx1)*(1.0-i*hx1)/(sigma*sigma));
  }
  for (i=info2.xs; i<info2.xs+info2.xm; i++) {
    u2[i] = 50.0*PetscExpReal(-0.5*(i+1)*hx2*(i+1)*hx1)/(sigma*sigma);
  }
  ierr = DMDAVecRestoreArray(da1,U1,&u1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,U2,&u2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(user->pack,U,&U1,&U2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->pack,U,user->U1loc,user->U2loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  struct _UserCtx user;                   /* user-defined application context */
  TS              ts;                     /* timestepping context */
  DM              da1,da2;                /* data management context */
  Mat             JP;                      /* matrix data structure */
  Vec             U,F;                    /* approximate solution vector */
  IS              *isg;
  PetscReal       dt = 0.2;               /* default step size */
  PetscReal       time_total_max = 1.0;   /* default max total time */
  PetscInt        time_steps_max = 100;   /* default max timesteps */
//  PetscDraw       draw;                   /* drawing context */
  PetscInt        m1,m2;
  PetscMPIInt     size;
  PetscBool       pass_dm = PETSC_TRUE;
//  TSProblemType   tsproblem = TS_LINEAR;
  PetscErrorCode  ierr;

  /*
     Initialize program and set problem parameters
  */
  ierr        = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

  m1             = 60;
  m2             = 60;
  ierr           = PetscOptionsGetInt(NULL,NULL,"-m1",&m1,NULL);CHKERRQ(ierr);
  ierr           = PetscOptionsGetInt(NULL,NULL,"-m2",&m2,NULL);CHKERRQ(ierr);
  //ierr           = PetscOptionsHasName(NULL,NULL,"-debug",&user.debug);CHKERRQ(ierr);
  ierr           = PetscOptionsGetBool(NULL,NULL,"-pass_dm",&pass_dm,NULL);CHKERRQ(ierr);
  user.m1        = m1;
  user.m2        = m2;
  user.hx1       = 1.0/(m1-1);
  user.hx2       = 1.0/m2;
  user.rho1      = 1.0;
  user.c1        = 1.0;
  user.diff1     = 1.0;
  user.rho2      = 1.0;
  user.c2        = 1.0;
  user.diff2     = 1.0;
  user.eddydiff1 = user.diff1/(user.rho1*user.c1);
  user.eddydiff2 = user.diff2/(user.rho2*user.c2);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving a coupling problem, number of processors = %d\n",size);CHKERRQ(ierr);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are M
     total grid values spread equally among all the processors.
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,m1,1,1,NULL,&da1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da1);CHKERRQ(ierr);
  ierr = DMSetUp(da1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,m2,1,1,NULL,&da2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);

  /*
    Create DMComposite to glue the two DMs together.
  */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&user.pack);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.pack,"pack_");CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,da2);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,0,"u1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,0,"u2");CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.pack);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
  // ierr = DMCreateGlobalVector(user.da1,&u1);CHKERRQ(ierr);
  // ierr = DMCreateLocalVector(user.da1,&user.u_local1);CHKERRQ(ierr);
  // ierr = DMCreateGlobalVector(user.da2,&u2);CHKERRQ(ierr);
  // ierr = DMCreateLocalVector(user.da2,&user.u_local2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.pack,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&F);CHKERRQ(ierr);

  ierr = DMCompositeGetGlobalISs(user.pack,&isg);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user.pack,&user.U1loc,&user.U2loc);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user.pack,U,user.U1loc,user.U2loc);CHKERRQ(ierr);

  ierr = FormInitial_Coupled(&user,U);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.pack);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.pack,&JP);CHKERRQ(ierr);

  ierr = ModifyCouplingSubMatrixL2G(&user,JP);CHKERRQ(ierr);

  /* This example does not correctly allocate off-diagonal blocks. These options allows new nonzeros (slow). */
  ierr = MatSetOption(JP,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(JP,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,F,FormRHSFunction_All,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,JP,JP,FormRHSJacobian_All,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,time_total_max);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,time_steps_max);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  if (!pass_dm) { /* Manually provide index sets and names for the splits */
    SNES snes;
    KSP  ksp;
    PC   pc;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u1",isg[0]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetIS(pc,"u2",isg[1]);CHKERRQ(ierr);
  } else {
    /* The same names come from the options prefix for da1 and da2. This option can support geometric multigrid inside
     * of splits, but it requires using a DM (perhaps your own implementation). */
    SNES snes;
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,user.pack);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = ISDestroy(&isg[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&isg[1]);CHKERRQ(ierr);
  ierr = PetscFree(isg);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&JP);CHKERRQ(ierr);
  ierr = DMDestroy(&da1);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = DMDestroy(&user.pack);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
}
