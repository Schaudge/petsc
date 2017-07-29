#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar u,v,h;
} Field;

typedef struct {
  PetscReal EarthRadius;
  PetscReal Gravity;
  PetscReal AngularSpeed;
  PetscReal alpha,phi;
  PetscInt  p;
  PetscInt  q;
} Model_SW;

PetscErrorCode InitialConditions(DM da,Vec U,Model_SW *sw)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      a,omega,g,phi,u0,dlat,lat,lon;
  Field          **uarr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  phi   = sw->phi;
  u0    = 20.0;

  ierr = DMDAVecGetArray(da,U,&uarr);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  dlat = PETSC_PI/(PetscReal)(My);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2; /* shift half grid size to avoid North pole and South pole */
    for (i=xs; i<xs+xm; i++) { /* longitude */
      lon = i*dlat; /* dlon = dlat */
      uarr[j][i].u = -3.*u0*PetscSinReal(lat)*PetscCosReal(lat)*PetscCosReal(lat)*PetscSinReal(lon)+u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lon);
      uarr[j][i].v = u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lon);
      uarr[j][i].h = (phi+2.*omega*a*u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lat)*PetscSinReal(lon))/g;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  Model_SW      *sw = (Model_SW*)ptr;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,p,q,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  PetscScalar    fc,fnq,fsq,uc,ue,uep,uwp,uw,un,us,unq,uepnq,uwpnq,usq,uepsq,uwpsq,vc,ve,vw,vep,vwp,vn,vnq,vs,vsq,vepnq,vwpnq,vepsq,vwpsq,hc,he,hw,hep,hwp,hn,hs,hnq,hsq;
  Field          **uarr,**farr;
  Vec            localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  p     = sw->p;
  q     = sw->q;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&farr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected,e.g. v(-1) = v(+1)
     Height is assumed to be constant, e.g. h(-1) = h(0)
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      uarr[-1][i].u = -uarr[0][i].u;
      uarr[-1][i].v = -uarr[0][i].v;
      uarr[-1][i].h = uarr[0][i].h;
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      uarr[My][i].u = -uarr[My-1][i].u;
      uarr[My][i].v = -uarr[My-1][i].v;
      uarr[My][i].h = uarr[My-1][i].h;
    }
  }
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+qh*dlat);
    fsq = 2.*omega*PetscSinReal(lat-qh*dlat);
    for (i=xs; i<xs+xm; i++) { /* longitude */
      uc    = uarr[j][i].u;
      ue    = uarr[j][i+1].u;
      uep   = uarr[j][i+ph].u;
      uwp   = uarr[j][i-ph].u;
      uw    = uarr[j][i-1].u;
      un    = uarr[j+1][i].u;
      us    = uarr[j-1][i].u;
      unq   = uarr[j+qh][i].u;
      uepnq = uarr[j+qh][i+ph].u;
      uwpnq = uarr[j+qh][i-ph].u;
      usq   = uarr[j-qh][i].u;
      uepsq = uarr[j-qh][i+ph].u;
      uwpsq = uarr[j-qh][i-ph].u;

      vc    = uarr[j][i].v;
      ve    = uarr[j][i+1].v;
      vw    = uarr[j][i-1].v;
      vep   = uarr[j][i+ph].v;
      vwp   = uarr[j][i-ph].v;
      vn    = uarr[j+1][i].v;
      vnq   = uarr[j+qh][i].v;
      vs    = uarr[j-1][i].v;
      vsq   = uarr[j-qh][i].v;
      vepnq = uarr[j+qh][i+ph].v;
      vwpnq = uarr[j+qh][i-ph].v;
      vepsq = uarr[j-qh][i+ph].v;
      vwpsq = uarr[j-qh][i-ph].v;

      hc  = uarr[j][i].h;
      he  = uarr[j][i+1].h;
      hw  = uarr[j][i-1].h;
      hep = uarr[j][i+ph].h;
      hwp = uarr[j][i-ph].h;
      hn  = uarr[j+1][i].h;
      hs  = uarr[j-1][i].h;
      hnq = uarr[j+qh][i].h;
      hsq = uarr[j-qh][i].h;

      farr[j][i].u = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ue-uw)+vc*(un-us)+2.*g/(p*PetscCosReal(lat))*(hep-hwp))
                    +(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*vc
                    +alpha/2.*(fc+uep/a*PetscTanReal(lat))*vep
                    +alpha/2.*(fc+uwp/a*PetscTanReal(lat))*vwp;
      farr[j][i].v = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ve-vw)+vc*(vn-vs)+2.*g/q*(hnq-hsq))
                    -(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*uc
                    -alpha/2.*(fnq+unq/a*PetscTanReal(lat+qh*dlat))*unq
                    -alpha/2.*(fsq+usq/a*PetscTanReal(lat-qh*dlat))*usq;
      farr[j][i].h = -1./(2.*a*dlat)*(
                     uc/PetscCosReal(lat)*(he-hw)
                    +vc*(hn-hs)
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(uep-uwp)+alpha/2.*(uepnq-uwpnq+uepsq-uwpsq))/p
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(vnq*PetscCosReal(lat+qh*dlat)-vsq*PetscCosReal(lat-qh*dlat))+alpha/2.*(vepnq*PetscCosReal(lat+qh*dlat)-vepsq*PetscCosReal(lat-qh*dlat)+vwpnq*PetscCosReal(lat+qh*dlat)-vwpsq*PetscCosReal(lat-qh*dlat)))/q
                     );
    }
  }
  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ptx)
{
  Model_SW       *sw=(Model_SW*)ptx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  Field          **uarr;
  Vec            localU;
  MatStencil     stencil[19],rowstencil;
  PetscScalar    entries[19];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected,e.g. v(-1) = v(+1)
     Height is assumed to be constant, e.g. h(-1) = h(0)
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      uarr[-1][i].u = -uarr[0][i].u;
      uarr[-1][i].v = -uarr[0][i].v;
      uarr[-1][i].h = uarr[0][i].h;
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      uarr[My][i].u = -uarr[My-1][i].u;
      uarr[My][i].v = -uarr[My-1][i].v;
      uarr[My][i].h = uarr[My-1][i].h;
    }
  }

  for (i=0; i<19; i++) stencil[i].k = 0;
  rowstencil.k = 0;
  rowstencil.c = 0;
  for (j=ys; j<ys+ym; j++) {
    PetscReal fc;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);

    stencil[0].j  = j+1;
    stencil[1].j  = j;
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = j;
    stencil[6].j  = j-1;
    stencil[7].j  = j;
    stencil[8].j  = j;
    stencil[9].j  = j;
    stencil[10].j = j;
    stencil[11].j = j;

    /* Relocate the ghost points at north and south boundaries */
    if (j==0) stencil[6].j = j;
    if (j==My-1) stencil[0].j = j;

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal vc,vep,vwp,uc,ue,uw,un,us,uep,uwp;
      uc  = uarr[j][i].u;
      ue  = uarr[j][i+1].u;
      uw  = uarr[j][i-1].u;
      un  = uarr[j+1][i].u;
      us  = uarr[j-1][i].u;
      uep = uarr[j][i+ph].u;
      uwp = uarr[j][i-ph].u;
      vc  = uarr[j][i].v;
      vep = uarr[j][i+ph].v;
      vwp = uarr[j][i-ph].v;

      stencil[0].i  = i;    stencil[0].c  = 0; entries[0]  = -1./(2.*a*dlat)*vc; /* un */
      stencil[1].i  = i-ph; stencil[1].c  = 0; entries[1]  = alpha/2./a*PetscTanReal(lat)*vwp; /* uwp */
      stencil[2].i  = i-1;  stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat)*uc/PetscCosReal(lat); /* uw */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat)/PetscCosReal(lat)*(ue-uw)+(1.-alpha)/a*PetscTanReal(lat)*vc; /* uc */
      stencil[4].i  = i+1;  stencil[4].c  = 0; entries[4]  = -entries[2]; /* ue */
      stencil[5].i  = i+ph; stencil[5].c  = 0; entries[5]  = alpha/2./a*PetscTanReal(lat)*vep; /* uep */
      stencil[6].i  = i;    stencil[6].c  = 0; entries[6]  = -entries[0]; /* us */
      stencil[7].i  = i-ph; stencil[7].c  = 1; entries[7]  = alpha/2.*(fc+uwp/a*PetscTanReal(lat)); /* vwp */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat)*(un-us)+(1.-alpha)*(fc+uc/a*PetscTanReal(lat)); /* vc */
      stencil[9].i  = i+ph; stencil[9].c  = 1; entries[9]  = alpha/2.*(fc+uep/a*PetscTanReal(lat)); /* vep */
      stencil[10].i = i-ph; stencil[10].c = 2; entries[10] = 1./(2.*a*dlat)*2.*g/(sw->p*PetscCosReal(lat));/* hwp */
      stencil[11].i = i+ph; stencil[11].c = 2; entries[11] = -entries[10]; /* hep */

      /* flip the sign */
      if (j==0) entries[6] = -entries[6];
      if (j==My-1) entries[0] = -entries[0];

      rowstencil.i = i;
      //for (int k=0;k<19;k++) entries[k] += 30000+10*k;
      ierr = MatSetValuesStencil(A,1,&rowstencil,12,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 1;
  for (j=ys; j<ys+ym; j++) {
    PetscReal fc,fnq,fsq;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+qh*dlat);
    fsq = 2.*omega*PetscSinReal(lat-qh*dlat);

    stencil[0].j = j+1;
    stencil[1].j = j;
    stencil[2].j = j-1;
    stencil[3].j = j+1;
    stencil[4].j = j;
    stencil[5].j = j;
    stencil[6].j = j;
    stencil[7].j = j-1;
    stencil[8].j = j+qh;
    stencil[9].j = j-qh;

    /* Relocate the ghost points at north and south boundaries */
    if (j == 0) {
      stencil[2].j = j;
      stencil[7].j = j;
      stencil[9].j = j;
    }
    if (j == My-1) {
      stencil[0].j = j;
      stencil[3].j = j;
      stencil[8].j = j;
    }

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,unq,usq,ve,vs,vw,vn,vc;
      uc  = uarr[j][i].u;
      unq = uarr[j+qh][i].u;
      usq = uarr[j-qh][i].u;
      ve  = uarr[j][i+1].v;
      vs  = uarr[j-1][i].v;
      vw  = uarr[j][i-1].v;
      vn  = uarr[j+1][i].v;
      vc  = uarr[j][i].v;

      stencil[0].i = i;   stencil[0].c = 0; entries[0] = -alpha/2.*(fnq+2.*unq/a*PetscTanReal(lat+qh*dlat)); /* unq */
      stencil[1].i = i;   stencil[1].c = 0; entries[1] = -1./(2.*a*dlat*PetscCosReal(lat))*(ve-vw)-(1.-alpha)*(fc+2.*uc/a*PetscTanReal(lat)); /* uc */
      stencil[2].i = i;   stencil[2].c = 0; entries[2] = -alpha/2.*(fsq+2.*usq/a*PetscTanReal(lat-qh*dlat)); /* usq */
      stencil[3].i = i;   stencil[3].c = 1; entries[3] = -1./(2.*a*dlat)*vc; /* vn */
      stencil[4].i = i-1; stencil[4].c = 1; entries[4] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* vw */
      stencil[5].i = i;   stencil[5].c = 1; entries[5] = -1./(2.*a*dlat)*(vn-vs); /* vc */
      stencil[6].i = i+1; stencil[6].c = 1; entries[6] = -entries[4]; /* ve */
      stencil[7].i = i;   stencil[7].c = 1; entries[7] = -entries[3]; /* vs */
      stencil[8].i = i;   stencil[8].c = 2; entries[8] = -1./(2.*a*dlat*sw->q)*2.*g; /* hnq */
      stencil[9].i = i;   stencil[9].c = 2; entries[9] = -entries[8]; /* hsq */

      /* flip the sign */
      if (j == 0) {
        entries[2] = -entries[2];
        entries[7] = -entries[7];
      }
      if (j == My-1) {
        entries[0] = -entries[0];
        entries[3] = -entries[3];
      }
      rowstencil.i = i;
      //for (int k=0;k<19;k++) entries[k] += 50000+100*k;
      ierr = MatSetValuesStencil(A,1,&rowstencil,10,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 2;
  for (j=ys; j<ys+ym; j++) {
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */

    stencil[0].j  = j+qh;
    stencil[1].j  = j+qh;
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = j-qh;
    stencil[6].j  = j-qh;
    stencil[7].j  = j+qh;
    stencil[8].j  = j+qh;
    stencil[9].j  = j+qh;
    stencil[10].j = j;
    stencil[11].j = j-qh;
    stencil[12].j = j-qh;
    stencil[13].j = j-qh;
    stencil[14].j = j+1;
    stencil[15].j = j;
    stencil[16].j = j;
    stencil[17].j = j;
    stencil[18].j = j-1;

    /* Relocate the ghost points at north and south boundaries */
    if (j == 0) {
      stencil[5].j  = j;
      stencil[6].j  = j;
      stencil[11].j = j;
      stencil[12].j = j;
      stencil[13].j = j;
      stencil[18].j = j;
    }
    if (j == My-1) {
      stencil[0].j  = j;
      stencil[1].j  = j;
      stencil[7].j  = j;
      stencil[8].j  = j;
      stencil[9].j  = j;
      stencil[14].j = j;
    }

    rowstencil.j  = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,uep,uwp,uepnq,uwpnq,uepsq,uwpsq,vc,vnq,vsq,vepnq,vwpnq,vepsq,vwpsq,hc,he,hw,hs,hn;
      uc    = uarr[j][i].u;
      uep   = uarr[j][i+ph].u;
      uwp   = uarr[j][i-ph].u;
      uepnq = uarr[j+qh][i+ph].u;
      uwpnq = uarr[j+qh][i-ph].u;
      uepsq = uarr[j-qh][i+ph].u;
      uwpsq = uarr[j-qh][i-ph].u;
      vc    = uarr[j][i].v;
      vnq   = uarr[j+qh][i].v;
      vsq   = uarr[j-qh][i].v;
      vepnq = uarr[j+qh][i+ph].v;
      vwpnq = uarr[j+qh][i-ph].v;
      vepsq = uarr[j-qh][i+ph].v;
      vwpsq = uarr[j-qh][i-ph].v;
      hc    = uarr[j][i].h;
      he    = uarr[j][i+1].h;
      hw    = uarr[j][i-1].h;
      hs    = uarr[j-1][i].h;
      hn    = uarr[j+1][i].h;

      stencil[0].i  = i-ph; stencil[0].c  = 0; entries[0]  = 1./(2.*a*dlat*PetscCosReal(lat)*2.*sw->p)*2.*hc*alpha; /* uwpnq */
      stencil[1].i  = i+ph; stencil[1].c  = 0; entries[1]  = -entries[0]; /* uepnq */
      stencil[2].i  = i-ph; stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat*PetscCosReal(lat)*sw->p)*2.*hc*(1.-alpha); /* uwp */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat*PetscCosReal(lat))*(he-hw); /* uc */
      stencil[4].i  = i+ph; stencil[4].c  = 0; entries[4]  = -entries[2]; /* uep */
      stencil[5].i  = i-ph; stencil[5].c  = 0; entries[5]  = entries[0]; /* uwpsq */
      stencil[6].i  = i+ph; stencil[6].c  = 0; entries[6]  = -entries[0]; /* uepsq */
      stencil[7].i  = i-ph; stencil[7].c  = 1; entries[7]  = -1./(2.*a*dlat*PetscCosReal(lat)*2.*sw->q)*2.*hc*alpha*PetscCosReal(lat+qh*dlat); /* vwpnq */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat*PetscCosReal(lat)*sw->q)*2.*hc*(1.-alpha)*PetscCosReal(lat+qh*dlat); /* vnq */
      stencil[9].i  = i+ph; stencil[9].c  = 1; entries[9]  = entries[7]; /* vepnq */
      stencil[10].i = i;    stencil[10].c = 1; entries[10] = -1./(2.*a*dlat)*(hn-hs); /* vc */
      stencil[11].i = i-ph; stencil[11].c = 1; entries[11] = 1./(2.*a*dlat*PetscCosReal(lat)*2.*sw->q)*(2.*hc*alpha*PetscCosReal(lat-qh*dlat)); /* vwpsq */
      stencil[12].i = i;    stencil[12].c = 1; entries[12] = 1./(2.*a*dlat*PetscCosReal(lat)*sw->q)*2.*hc*(1.-alpha)*PetscCosReal(lat-qh*dlat); /* vsq */
      stencil[13].i = i+ph; stencil[13].c = 1; entries[13] = entries[11]; /* vepsq */
      stencil[14].i = i;    stencil[14].c = 2; entries[14] = -1./(2.*a*dlat)*vc; /* hn */
      stencil[15].i = i-1;  stencil[15].c = 2; entries[15] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* hw */
      stencil[16].i = i;    stencil[16].c = 2; entries[16] = -1./(2.*a*dlat)*(2./PetscCosReal(lat)*((1.-alpha)*(uep-uwp)+alpha/2.*(uepnq-uwpnq+uepsq-uwpsq))/sw->p + 2./PetscCosReal(lat)*((1.-alpha)*(vnq*PetscCosReal(lat+qh*dlat)-vsq*PetscCosReal(lat-qh*dlat))+alpha/2.*(vepnq*PetscCosReal(lat+qh*dlat)-vepsq*PetscCosReal(lat-qh*dlat)+vwpnq*PetscCosReal(lat+qh*dlat)-vwpsq*PetscCosReal(lat-qh*dlat)))/sw->q); /* hc */
      stencil[17].i = i+1;  stencil[17].c = 2; entries[17] = -entries[15]; /* he */
      stencil[18].i = i;    stencil[18].c = 2; entries[18] = -entries[14]; /* hs */

      /* flip the sign */
      if (j == 0) {
        entries[5]  = -entries[5];
        entries[6]  = -entries[6];
        entries[11] = -entries[11];
        entries[12] = -entries[12];
        entries[13] = -entries[13];
      }
      if (j == My-1) {
        entries[0] = -entries[0];
        entries[1] = -entries[1];
        entries[7] = -entries[7];
        entries[8] = -entries[8];
        entries[9] = -entries[9];
      }
      rowstencil.i = i;
      //for (int k=0;k<19;k++) entries[k] += 70000+1000*k;
      ierr = MatSetValuesStencil(A,1,&rowstencil,19,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  Vec            U;
  DM             da;
  Model_SW       sw;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  sw.Gravity      = 9.8;
  sw.EarthRadius  = 6.37e6;
  sw.alpha        = 1./3.;
  sw.phi          = 5.768e4;
  sw.AngularSpeed = 7.292e-5;
  sw.p            = 4;
  sw.q            = 2;

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,150,75,PETSC_DECIDE,PETSC_DECIDE,3,2,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"h");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&sw);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&sw);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);

  ierr = InitialConditions(da,U,&sw);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  ierr = TSSetMaxSteps(ts,3600);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,32.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
