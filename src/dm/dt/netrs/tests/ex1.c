static const char help[] = "Test function for 1-2 junction for NetRS. In conjunction with a matlab script build error phase \
plots for the linearized solver and plots the performance of the various error estimators. Useful for testing new \
netrs implementations and error estimators WIP";

#include <petscriemannsolver.h>
#include <petscnetrs.h>
#include "../fluxfun.h"  /*please remove this after generating a proper flux physics class. This is terrible */ 

/* all this will do is read in the user inputs for the network riemann problems and then will solve it, outputting diagnoistic into 
various files in the directory ex1. This is intended to be called in a matlab loop for plotting phase error plots */ 


/* currently this is hard coded for the SWE equations */ 
int main(int argc,char *argv[])
{
    PetscErrorCode ierr; 
    RiemannSolver  rs; 
    NetRS          netrs; 
    PetscReal      ph,pu,d1h,d2h,d1u,d2u; /* riemann data for the parent and daugther branches */ 
    MPI_Comm       comm = PETSC_COMM_SELF;
    FluxFunction   fluxfun; 
    PetscInt       i,numedges = 3;
    FILE           *outputp,*outputd1,*outputd2;
    PetscReal      *err, *flux, u[6],fluxlin[6],*ustar_exact;
    EdgeDirection  dir[3] = {EDGEIN,EDGEOUT,EDGEOUT};
    PetscScalar    *eig; 

    ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
    /* default riemann problem */
    ph = 2; pu = 0; d1h = 1; d1u = 0; d2h = 1; d2u = 0;     

    /* Command Line Options */
    ierr = PetscOptionsBegin(comm,NULL,"Ex1 Options","");CHKERRQ(ierr);
        ierr = PetscOptionsReal("-ph","parent height","",ph,&ph,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-pu","parent momentum","",pu,&pu,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-d1h","daugher 1 height","",d1h,&d1h,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-d1u","daugher 1 momentum","",d1u,&d1u,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-d2h","dauther 2 height","",d2h,&d2h,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-d2u","daugher 2 momentum","",d2u,&d2u,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    /* Input Sanitization */
    if (ph <= 0 || d1h <=0 || d2h <= 0) {
        SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"All input water heights must be strictly positive.");
    }
    u[0] = ph; u[1] = pu; u[2] = d1h; u[3] = d1u; u[4] = d2h; u[5] = d2u; 

    ierr = PhysicsCreate_Shallow(&fluxfun);CHKERRQ(ierr);

    /* Set up Riemann Solver */
    ierr = RiemannSolverCreate(comm,&rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(rs,fluxfun->user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(rs,fluxfun->fluxeig);CHKERRQ(ierr);
    ierr = RiemannSolverSetRoeAvgFunct(rs,fluxfun->roeavg);CHKERRQ(ierr);
    ierr = RiemannSolverSetRoeMatrixFunct(rs,fluxfun->roemat);CHKERRQ(ierr);
    ierr = RiemannSolverSetEigBasis(rs,fluxfun->eigbasis);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(rs,1,fluxfun->dof,fluxfun->flux);CHKERRQ(ierr); 
    ierr = RiemannSolverSetLaxCurve(rs,fluxfun->laxcurve);CHKERRQ(ierr);
    ierr = RiemannSolverSetUp(rs);CHKERRQ(ierr);

    /* Setup NetRS */
    ierr = NetRSCreate(comm,&netrs);CHKERRQ(ierr);
    ierr = NetRSSetApplicationContext(netrs,fluxfun->user);CHKERRQ(ierr);
    ierr = NetRSSetRiemannSolver(netrs,rs);CHKERRQ(ierr);
    ierr = NetRSSetNumEdges(netrs,numedges);CHKERRQ(ierr);

    /* Clean up the Output Directory */
    ierr = PetscRMTree("ex1output");CHKERRQ(ierr);
    ierr = PetscMkdir("ex1output");CHKERRQ(ierr);

    /* Note : Should have function to return all available error estimator (once they are a class) and all available netrs/rs 
    to automate these tests */

    /* Compute diagnostics for the linearized solver */ 
    ierr = PetscFOpen(comm,"ex1output/linearized_p.csv","a",&outputp);CHKERRQ(ierr);
    ierr = PetscFOpen(comm,"ex1output/linearized_d1.csv","a",&outputd1);CHKERRQ(ierr);
    ierr = PetscFOpen(comm,"ex1output/linearized_d2.csv","a",&outputd2);CHKERRQ(ierr);
    /* name of all error estimators goes here */
    ierr = PetscFPrintf(comm,outputp,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n");CHKERRQ(ierr);

    /* iterate through the available error estimators and compute the result */

    ierr = NetRSSetErrorEstimate(netrs,NetRSRoeErrorEstimate);CHKERRQ(ierr);
    ierr = NetRSSetType(netrs,NETRSLINEAR);CHKERRQ(ierr); 
    ierr = NetRSSetUp(netrs);CHKERRQ(ierr);

    /* Roe Err Estimate */
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);

    /* Lax Err Estimate */
    ierr = NetRSSetErrorEstimate(netrs,NetRSLaxErrorEstimate);CHKERRQ(ierr);
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);
    
    /* Taylor Err Estimate */
    ierr = NetRSSetErrorEstimate(netrs,NetRSTaylorErrorEstimate);CHKERRQ(ierr);
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);

    /* Exact Flux Difference */ 
    for(i=0; i<numedges; i++) {fluxlin[i] = flux[i];}

    ierr = NetRSSetType(netrs,NETRSEXACTSWE);CHKERRQ(ierr); 
    ierr = NetRSSetUp(netrs);
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr);
    /* L^2 flux error */
    for(i=0; i<numedges; i++) 
    {
        fluxlin[i*fluxfun->dof]   -= flux[i*fluxfun->dof];
        fluxlin[i*fluxfun->dof+1] -= flux[i*fluxfun->dof+1];
    }
    for(i=0; i<numedges; i++) 
    {
    err[i] = PetscSqrtReal(PetscSqr(fluxlin[i*fluxfun->dof])+PetscSqr(fluxlin[i*fluxfun->dof+1]));
    }
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);

    /* Exact Star State Difference */
    ierr = PetscMalloc1(6,&ustar_exact);CHKERRQ(ierr);
    ierr = NetRSSetType(netrs,NETRSEXACTSWESTAR);CHKERRQ(ierr); 
    ierr = NetRSSetUp(netrs);
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr); /* flux contains the star state instead of the flux */
    
    for(i=0; i<6; i++)
    {
        ustar_exact[i] = flux[i];
    }
    ierr = NetRSSetType(netrs,NETRSLINEARSTAR);CHKERRQ(ierr); 
    ierr = NetRSSetUp(netrs);
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr); /* flux contains the star state instead of the flux */
    for(i=0; i<6; i++)
    {
        flux[i] -= ustar_exact[i];
    }
    for(i=0; i<3; i++)
    {
        err[i] = PetscSqrtReal(PetscSqr(flux[i*fluxfun->dof])+PetscSqr(flux[i*fluxfun->dof+1]));
    }
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);
    /* Compute properties of the computed states */
    /* Rarefaction vs shock check */
    for(i=0;i<3; i++)
    {
       err[i] = (ustar_exact[i*fluxfun->dof] < u[i*fluxfun->dof]) ? 1 : -1; /*  -1: Shock wave. 1: Rarefaction Wave*/ 
    }
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);

    /* fluvial state check */
    for(i=0;i<3; i++)
    {
        RiemannSolverComputeEig(rs,ustar_exact+i*fluxfun->dof,&eig);CHKERRQ(ierr);
        err[i] = (PetscSign(eig[0]) == PetscSign(eig[1])) ? 0 : 1; 
    }
    ierr = PetscFPrintf(comm,outputp,"%e,",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e,",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e,",err[2]);CHKERRQ(ierr);

    /* Neg Height */
    ierr = NetRSEvaluate(netrs,u,dir,&flux,&err,NULL);CHKERRQ(ierr); /* flux contains the star state instead of the flux */
    for(i=0;i<3; i++)
    {
        err[i] = (flux[i*fluxfun->dof] <=0) ? 1 : 0; 
    }

    ierr = PetscFPrintf(comm,outputp,"%e",err[0]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd1,"%e",err[1]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,outputd2,"%e",err[2]);CHKERRQ(ierr);

    ierr = PetscFClose(comm,outputp);CHKERRQ(ierr);
    ierr = PetscFClose(comm,outputd1);CHKERRQ(ierr);
    ierr = PetscFClose(comm,outputd2);CHKERRQ(ierr);

    ierr = RiemannSolverDestroy(&rs);CHKERRQ(ierr); 
    ierr = NetRSDestroy(&netrs);CHKERRQ(ierr);
    ierr = fluxfun->destroy(&fluxfun);
}