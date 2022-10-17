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

    PetscCall(PetscInitialize(&argc,&argv,0,help));
    /* default riemann problem */
    ph = 2; pu = 0; d1h = 1; d1u = 0; d2h = 1; d2u = 0;     

    /* Command Line Options */
    PetscOptionsBegin(comm,NULL,"Ex1 Options","");
        PetscCall(PetscOptionsReal("-ph","parent height","",ph,&ph,NULL));
        PetscCall(PetscOptionsReal("-pu","parent momentum","",pu,&pu,NULL));
        PetscCall(PetscOptionsReal("-d1h","daugher 1 height","",d1h,&d1h,NULL));
        PetscCall(PetscOptionsReal("-d1u","daugher 1 momentum","",d1u,&d1u,NULL));
        PetscCall(PetscOptionsReal("-d2h","dauther 2 height","",d2h,&d2h,NULL));
        PetscCall(PetscOptionsReal("-d2u","daugher 2 momentum","",d2u,&d2u,NULL));
    PetscOptionsEnd();

    /* Input Sanitization */
    if (ph <= 0 || d1h <=0 || d2h <= 0) {
        SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"All input water heights must be strictly positive.");
    }
    u[0] = ph; u[1] = pu; u[2] = d1h; u[3] = d1u; u[4] = d2h; u[5] = d2u; 

    PetscCall(PhysicsCreate_Shallow(&fluxfun));

    /* Set up Riemann Solver */
    PetscCall(RiemannSolverCreate(comm,&rs));
    PetscCall(RiemannSolverSetApplicationContext(rs,fluxfun->user));
    PetscCall(RiemannSolverSetFromOptions(rs));
    PetscCall(RiemannSolverSetFluxEig(rs,fluxfun->fluxeig));
    PetscCall(RiemannSolverSetRoeAvgFunct(rs,fluxfun->roeavg));
    PetscCall(RiemannSolverSetRoeMatrixFunct(rs,fluxfun->roemat));
    PetscCall(RiemannSolverSetEigBasis(rs,fluxfun->eigbasis));
    PetscCall(RiemannSolverSetFlux(rs,1,fluxfun->dof,fluxfun->flux)); 
    PetscCall(RiemannSolverSetLaxCurve(rs,fluxfun->laxcurve));
    PetscCall(RiemannSolverSetUp(rs));

    /* Setup NetRS */
    PetscCall(NetRSCreate(comm,&netrs));
    PetscCall(NetRSSetApplicationContext(netrs,fluxfun->user));
    PetscCall(NetRSSetRiemannSolver(netrs,rs));
    PetscCall(NetRSSetNumEdges(netrs,numedges));

    /* Clean up the Output Directory */
    PetscCall(PetscRMTree("ex1output"));
    PetscCall(PetscMkdir("ex1output"));

    /* Note : Should have function to return all available error estimator (once they are a class) and all available netrs/rs 
    to automate these tests */

    /* Compute diagnostics for the linearized solver */ 
    PetscCall(PetscFOpen(comm,"ex1output/linearized_p.csv","a",&outputp));
    PetscCall(PetscFOpen(comm,"ex1output/linearized_d1.csv","a",&outputd1));
    PetscCall(PetscFOpen(comm,"ex1output/linearized_d2.csv","a",&outputd2));
    /* name of all error estimators goes here */
    PetscCall(PetscFPrintf(comm,outputp,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n"));
    PetscCall(PetscFPrintf(comm,outputd1,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n"));
    PetscCall(PetscFPrintf(comm,outputd2,"Roe,Lax,Taylor,ExactFlux,ExactStar,WaveType,Fluvial,NegHeight\n"));

    /* iterate through the available error estimators and compute the result */

    PetscCall(NetRSSetErrorEstimate(netrs,NetRSRoeErrorEstimate));
    PetscCall(NetRSSetType(netrs,NETRSLINEAR)); 
    PetscCall(NetRSSetUp(netrs));

    /* Roe Err Estimate */
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL));
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));

    /* Lax Err Estimate */
    PetscCall(NetRSSetErrorEstimate(netrs,NetRSLaxErrorEstimate));
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL));
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));
    
    /* Taylor Err Estimate */
    PetscCall(NetRSSetErrorEstimate(netrs,NetRSTaylorErrorEstimate));
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL));
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));

    /* Exact Flux Difference */ 
    for(i=0; i<numedges; i++) {fluxlin[i] = flux[i];}

    PetscCall(NetRSSetType(netrs,NETRSEXACTSWE)); 
    PetscCall(NetRSSetUp(netrs));
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL));
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
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));

    /* Exact Star State Difference */
    PetscCall(PetscMalloc1(6,&ustar_exact));
    PetscCall(NetRSSetType(netrs,NETRSEXACTSWESTAR)); 
    PetscCall(NetRSSetUp(netrs));
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL)); /* flux contains the star state instead of the flux */
    
    for(i=0; i<6; i++)
    {
        ustar_exact[i] = flux[i];
    }
    PetscCall(NetRSSetType(netrs,NETRSLINEARSTAR)); 
    PetscCall(NetRSSetUp(netrs));
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL)); /* flux contains the star state instead of the flux */
    for(i=0; i<6; i++)
    {
        flux[i] -= ustar_exact[i];
    }
    for(i=0; i<3; i++)
    {
        err[i] = PetscSqrtReal(PetscSqr(flux[i*fluxfun->dof])+PetscSqr(flux[i*fluxfun->dof+1]));
    }
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));
    /* Compute properties of the computed states */
    /* Rarefaction vs shock check */
    for(i=0;i<3; i++)
    {
       err[i] = (ustar_exact[i*fluxfun->dof] < u[i*fluxfun->dof]) ? 1 : -1; /*  -1: Shock wave. 1: Rarefaction Wave*/ 
    }
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));

    /* fluvial state check */
    for(i=0;i<3; i++)
    {
        PetscCall(RiemannSolverComputeEig(rs,ustar_exact+i*fluxfun->dof,&eig));
        err[i] = (PetscSign(eig[0]) == PetscSign(eig[1])) ? 0 : 1; 
    }
    PetscCall(PetscFPrintf(comm,outputp,"%e,",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e,",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e,",err[2]));

    /* Neg Height */
    PetscCall(NetRSEvaluate(netrs,u,dir,&flux,&err,NULL)); /* flux contains the star state instead of the flux */
    for(i=0;i<3; i++)
    {
        err[i] = (flux[i*fluxfun->dof] <=0) ? 1 : 0; 
    }

    PetscCall(PetscFPrintf(comm,outputp,"%e",err[0]));
    PetscCall(PetscFPrintf(comm,outputd1,"%e",err[1]));
    PetscCall(PetscFPrintf(comm,outputd2,"%e",err[2]));

    PetscCall(PetscFClose(comm,outputp));
    PetscCall(PetscFClose(comm,outputd1));
    PetscCall(PetscFClose(comm,outputd2));

    PetscCall(RiemannSolverDestroy(&rs)); 
    PetscCall(NetRSDestroy(&netrs));
    fluxfun->destroy(&fluxfun);
}
