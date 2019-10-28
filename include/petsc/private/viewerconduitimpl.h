#ifndef _viewerconduitimpl_h
#define _viewerconduitimpl_h

#include <petscviewer.h>
#include <conduit/conduit.hpp>

typedef struct {
  const char *filename;
  PetscFileMode filemode;
  conduit::Node *mesh;
} PetscViewer_Conduit;

#endif
