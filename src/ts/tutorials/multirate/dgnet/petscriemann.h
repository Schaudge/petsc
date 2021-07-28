
typedef struct _p_PetscRiemann* PetscRiemann; 
/* Name Objects similarly to petscds */
/* Flux function in balance law evaluated at a point, specifically in an equation of the form 
   \partial_t u + \div (F(u)) = G(u,x) (a Balance law)
   
   where F : \mathbb{R}^dim \to \mathbb{R}^m 
   dim : dimension of the equation (1,2,3 dimensional in general)
   m   : Size of the system of equations 

   This function is then the specification for F. Currently only using it for dim = 1 problems 
   so there is no assumptions on the format of the output (yet!) in terms of assumed m x dim or dim x m output
   structure for the c array. 
*/

/* 
    Note : Perhaps this would be better as a class as well, instead of a patterned function call? Frankly though 
    I won't know unless I start working with UFL, TSFC, Firedrake fenics etc... to get a sense of what is 
    needed/convenient. Will work for now. 
*/
typedef void (*PetscPointFlux)(void*,const PetscReal*,PetscReal*);