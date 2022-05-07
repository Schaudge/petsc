/* Simple Routines for Building Graphs to test the DGNet functions on. 
    Just includes construction routines for basic graphs as well as graph imbeddings into R^d, d=2 for all 
    imbeddings so far. 

    Designed to seperate out the DGNet specific construction routines from the dmnetwork 
    construction routines 

    These routines are generic on DMNetwork, but are designed to be used for DGNet 
*/


#if !defined(__NETCONSTRUCT_H)
#define __NETCONSTRUCT_H
#include <petscdmnetwork.h>


typedef enum {LINE,REVERSELINE,PERIODICLINE,PARENT_DAUGHTER} NetworkType;


/* Network Type: Enumeration on manually created network options */
/* Create Single SubNetwork DMNetwork with the following graph choices */ 
/* 
   This routine relies on the fact that I can register components after setting the DMnetwork layout. 
   This contradicts the documention on DMNetworkLayoutSetUp() but seems to work for some reason 
*/

#endif
