/*
Header file including necessary nvml headers.
*/

#ifndef INCLNVML
#define INCLNVML

#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <petscsys.h>

extern PETSC_VISIBILITY_PUBLIC void nvmlAPIRun();
extern PETSC_VISIBILITY_PUBLIC void nvmlAPIEnd();
void nvmlAPIstart();
double nvmlAPIstop();
void *powerPollingFunc(void *ptr);
double nvmlAPIcumul();
double nvmlAPIidle();
int getNVMLError(nvmlReturn_t resultToCheck);

#endif
