/* Modified from petsc/src/snes/examples/tutorials/network/water/waterreaddata.c */

#include "washdata.h"
#include <string.h>
#include <ctype.h>
#include <dirent.h>

int LineStartsWith(const char *a, const char *b)
{
  if (strncmp(a, b, strlen(b)) == 0) return 1;
  return 0;
}

int CheckDataSegmentEnd(const char *line)
{
  if (LineStartsWith(line,"[JUNCTIONS]") || \
     LineStartsWith(line,"[INFLOWS]") || \
     LineStartsWith(line,"[STAGES]") ||	\
     LineStartsWith(line,"[RESERVOIRS]") || \
     LineStartsWith(line,"[TANKS]") || \
     LineStartsWith(line,"[PIPES]") || \
     LineStartsWith(line,"[PUMPS]") || \
     LineStartsWith(line,"[CURVES]") || \
     LineStartsWith(line,"[VALVES]") || \
     LineStartsWith(line,"[PATTERNS]") || \
     LineStartsWith(line,"[VALVES]") || \
     LineStartsWith(line,"[QUALITY]") || \
     LineStartsWith(line,"\n") || LineStartsWith(line,"\r\n")) {
    return 1;
  }
  return 0;
}

/* Gets the file pointer positiion for the start of the data segment and the
   number of data segments (lines) read
*/
PetscErrorCode GetDataSegment(FILE *fp,char *line,fpos_t *data_segment_start_pos,PetscInt *ndatalines)
{
  PetscInt data_segment_end;
  PetscInt nlines=0;

  PetscFunctionBegin;
  data_segment_end = 0;
  fgetpos(fp,data_segment_start_pos);
  if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
  while (LineStartsWith(line,";")) {
    fgetpos(fp,data_segment_start_pos);
    if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
  }
  while (!data_segment_end) {
    if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data segment from file");
    nlines++;
    data_segment_end = CheckDataSegmentEnd(line);
  }
  *ndatalines = nlines;
  PetscFunctionReturn(0);
}

PetscErrorCode WaterReadData(WATERDATA *water,const char *filename)
{
  FILE              *fp;
  PetscErrorCode    ierr;
  VERTEX_Water      vert;
  EDGE_Water        edge;
  fpos_t            junc_start_pos,flow_start_pos,stage_start_pos,res_start_pos,tank_start_pos,pipe_start_pos,pump_start_pos;
  fpos_t            curve_start_pos,title_start_pos;
  char              line[MAXLINE];
  PetscInt          i,j,nv=0,ne=0,ncurve=0,ntitle=0,nlines,ndata,curve_id;
  JunctionData      *junction=NULL;
  Inflow            *inflow=NULL;
  Stage             *stage=NULL;
  Reservoir         *reservoir=NULL;
  Tank              *tank=NULL;
  PipeData           *pipe=NULL;
  PumpData           *pump=NULL;
  PetscScalar        curve_x,curve_y;
  double             v1,v2,v3,v4,v5,v6,v7=0.0;
  int                id,id_max,pattern,node1,node2;
  PetscTable         table;

  PetscFunctionBegin;
  water->nvertex = water->nedge = 0;
  fp = fopen(filename,"rb");

  /* Check for valid file; if not, load a default data file */
  if (!fp) {
    const char filename[PETSC_MAX_PATH_LEN]= "../cases/sample1.inp";
    fp = fopen(filename,"rb");
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nRead default file %s\n",filename);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nRead %s\n",filename);CHKERRQ(ierr);
  }

  /* Read file and get line numbers for different data segments */
  while (fgets(line,MAXLINE,fp)) {

    if (strstr(line,"[TITLE]")) {
      GetDataSegment(fp,line,&title_start_pos,&ntitle);
    }

    if (strstr(line,"[JUNCTIONS]")) {
      GetDataSegment(fp,line,&junc_start_pos,&nlines);
      water->nvertex += nlines;
      water->njunction = nlines;
    }

    if (strstr(line,"[INFLOWS]")) {
      GetDataSegment(fp,line,&flow_start_pos,&nlines);
      water->nvertex += nlines;
      water->ninflow = nlines;
    }

    if (strstr(line,"[STAGES]")) {
      GetDataSegment(fp,line,&stage_start_pos,&nlines);
      water->nvertex += nlines;
      water->nstage = nlines;
    }

    if (strstr(line,"[RESERVOIRS]")) {
      GetDataSegment(fp,line,&res_start_pos,&nlines);
      water->nvertex += nlines;
      water->nreservoir = nlines;
    }

    if (strstr(line,"[TANKS]")) {
      GetDataSegment(fp,line,&tank_start_pos,&nlines);
      water->nvertex += nlines;
      water->ntank = nlines;
    }

    if (strstr(line,"[PIPES]")) {
      GetDataSegment(fp,line,&pipe_start_pos,&nlines);
      water->nedge += nlines;
      water->npipe = nlines;
    }

    if (strstr(line,"[PUMPS]")) {
      GetDataSegment(fp,line,&pump_start_pos,&nlines);
      water->nedge += nlines;
      water->npump  = nlines;
    }

    if (strstr(line,"[CURVES]")) {
      GetDataSegment(fp,line,&curve_start_pos,&ncurve);
    }
  }

  /* Allocate vertex and edge data structs */
  ierr = PetscCalloc1(water->nvertex,&water->vertex);CHKERRQ(ierr);
  ierr = PetscCalloc1(water->nedge,&water->edge);CHKERRQ(ierr);
  vert = water->vertex;
  edge = water->edge;

  /* Junctions */
  id_max = -1;
  if (water->njunction) {
    fsetpos(fp,&junc_start_pos);
    for (i=0; i < water->njunction; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read junction from file");
      vert[nv].type = JUNCTION;
      junction      = &vert[nv].junc;
      ndata = sscanf(line,"%d %lf %lf %d",&id,&v1,&v2,&pattern);if (ndata < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read junction data");
      vert[nv].id = id;
      if (id > id_max) id_max = id;
      junction->dempattern = pattern;
      vert[nv].elev        = (PetscScalar)v1;
      junction->demand     = (PetscScalar)v2;
      junction->demand    *= GPM_CFS;
      junction->id         = vert[nv++].id;
    }
  }

  /* Inflows */
  if (water->ninflow) {
    fsetpos(fp,&flow_start_pos);
    for (i=0; i < water->ninflow; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read reservoir from file");
      vert[nv].type = INFLOW;
      inflow        = &vert[nv].inflow;
      ndata = sscanf(line,"%d %lf",&id,&v1);if (ndata < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read reservoir data");
      vert[nv].id = id;
      if (id > id_max) id_max = id;
      vert[nv].elev = (PetscScalar)v1;
      inflow->flow  = (PetscScalar)v2;
      inflow->id    = vert[nv++].id;
      //printf("id %d, elev %g, flow %g\n", inflow->id,inflow->elev,inflow->flow);
    } 
  }

  /* Stages */
  if (water->nstage) {
    fsetpos(fp,&stage_start_pos);
    for (i=0; i < water->nstage; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read reservoir from file");
      vert[nv].type = STAGE;
      stage         = &vert[nv].stage;
      ndata = sscanf(line,"%d %lf",&id,&v1);if (ndata < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read reservoir data");
      vert[nv].id = id;
      if (id > id_max) id_max = id;
      vert[nv].elev = (PetscScalar)v1;
       stage->head  = (PetscScalar)v2;
      stage->id     = vert[nv++].id;
    }
  }
  
  /* Reservoirs */
  if (water->nreservoir) {
    fsetpos(fp,&res_start_pos);
    for (i=0; i < water->nreservoir; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read reservoir from file");
      vert[nv].type = RESERVOIR;
      reservoir     = &vert[nv].res;
      ndata = sscanf(line,"%d %lf %lf",&id,&v1,&v2);if (ndata < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read reservoir data");
      vert[nv].id = id;
      if (id > id_max) id_max = id;
      vert[nv].elev   = (PetscScalar)v1;
      reservoir->head = (PetscScalar)v2;
      reservoir->id   = vert[nv++].id;
      /* printf("RESERVOIR id %d, elev %g, head %g\n",id,(PetscScalar)v1,reservoir->head); */
    }
  }

  /* Tanks */
  if (water->ntank) {
    int curve;
    fsetpos(fp,&tank_start_pos);
    for (i=0; i < water->ntank; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data tank from file");
      vert[nv].type = TANK;
      tank = &vert[nv].tank;
      ndata = sscanf(line,"%d %lf %lf %lf %lf %lf %lf %d",&id,&v1,&v2,&v3,&v4,&v5,&v6,&curve);if (ndata < 8) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read tank data");
      vert[nv].id       = id;
      if (id > id_max) id_max = id;
      tank->volumecurve = curve;
      vert[nv].elev   = (PetscScalar)v1;
      tank->head      = (PetscScalar)v2;
      tank->initlvl   = (PetscScalar)v3;
      tank->minlvl    = (PetscScalar)v4;
      tank->maxlvl    = (PetscScalar)v5;
      tank->diam      = (PetscScalar)v6;
      //tank->minvolume = (PetscScalar)v7;
      tank->id        = vert[nv++].id;
    }
  }
  /* printf("  id_max %d\n",id_max); */
  ierr = PetscTableCreate(water->nvertex,id_max,&table);CHKERRQ(ierr);
  water->table = table;

  for (i=0; i < water->nvertex; i++) {
    ierr = PetscTableAdd(table,vert[i].id,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Pipes */
  if (water->npipe) {
    fsetpos(fp,&pipe_start_pos);
    for (i=0; i < water->npipe; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data pipe from file");
      edge[ne].type = EDGE_TYPE_PIPE;
      pipe = &edge[ne].pipe;
      ndata = sscanf(line,"%d %d %d %lf %lf %lf %lf %lf %lf %lf %s",&id,&node1,&node2,&v1,&v2,&v3,&v4,&v5,&v6,&v7,pipe->stat);
      pipe->id        = id;
      pipe->node1     = node1;
      pipe->node2     = node2;
      pipe->length    = (PetscScalar)v1;
      pipe->width     = (PetscScalar)v2;
      pipe->roughness = (PetscScalar)v3;
      pipe->slope     = (PetscScalar)v4;
      pipe->qInitial  = (PetscScalar)v5;
      pipe->hInitial  = (PetscScalar)v6;
      pipe->minorloss = (PetscScalar)v7;
      edge[ne++].id   = pipe->id;
      if (strcmp(pipe->stat,"OPEN") == 0) pipe->status = PIPE_STATUS_OPEN;
      if (ndata < 8) {
        strcpy(pipe->stat,"OPEN"); /* default OPEN */
        pipe->status = PIPE_STATUS_OPEN;
      }
      if (ndata < 7) pipe->minorloss = 0.;
      pipe->n = 1.85;
      pipe->k = 4.72*pipe->length/(PetscPowScalar(pipe->roughness,pipe->n)*PetscPowScalar(0.0833333*pipe->width,4.87));
    }
  }

  /* Pumps */
  if (water->npump) {
    int paramid;
    fsetpos(fp,&pump_start_pos);
    for (i=0; i < water->npump; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data pump from file");
      edge[ne].type = EDGE_TYPE_PUMP;
      pump          = &edge[ne].pump;
      ndata = sscanf(line,"%d %d %d %s %d",&id,&node1,&node2,pump->param,&paramid);if (ndata != 5) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read pump data");
      pump->id      = id;
      pump->node1   = node1;
      pump->node2   = node2;
      pump->paramid = paramid;
      edge[ne++].id = pump->id;
    }
  }

  /* Curves */
  if (ncurve) {
    int icurve_id;
    fsetpos(fp,&curve_start_pos);
    for (i=0; i < ncurve; i++) {
      if (!fgets(line,MAXLINE,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Cannot read data curve from file");
      ndata = sscanf(line,"%d %lf %lf",&icurve_id,&v1,&v2);if (ndata != 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to read curve data");
      curve_id = icurve_id;
      curve_x  = (PetscScalar)v1;
      curve_y  = (PetscScalar)v2;
      /* Check for pump with the curve_id */
      for (j=water->npipe; j<water->npipe+water->npump; j++) {
        if (water->edge[j].pump.paramid == curve_id) {
          PetscCheck(pump->headcurve.npt < 3,PETSC_COMM_SELF,0,"Pump %d [%d --> %d]: No support for more than 3-pt head-flow curve",pump->id,pump->node1,pump->node2);

          pump = &water->edge[j].pump;
          pump->headcurve.flow[pump->headcurve.npt] = curve_x*GPM_CFS;
          pump->headcurve.head[pump->headcurve.npt] = curve_y;
          pump->headcurve.npt++;
          break;
        }
      }
    }
  }
  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode GetListofEdges_Water(WATERDATA *water,PetscInt *edgelist)
{
  PetscErrorCode ierr;
  PetscInt       i,j,node1,node2;
  PipeData       *pipe;
  PumpData       *pump;
  PetscBool      netview=PETSC_FALSE;
  PetscTable     table = water->table;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,NULL, "-water_view",&netview);CHKERRQ(ierr);
  /* printf("water->nedge %d, nvertex %d\n",water->nedge,water->nvertex); */
  for (i=0; i<water->nedge; i++) {
    if (water->edge[i].type == EDGE_TYPE_PIPE) {
      pipe  = &water->edge[i].pipe;
      node1 = pipe->node1;
      node2 = pipe->node2;
      if (netview) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"edge %d, pipe v[%d] -> v[%d]\n",i,node1,node2);CHKERRQ(ierr);
      }
    } else {
      pump  = &water->edge[i].pump;
      node1 = pump->node1;
      node2 = pump->node2;
      if (netview) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"edge %d, pump v[%d] -> v[%d]\n",i,node1,node2);CHKERRQ(ierr);
      }
    }

    ierr = PetscTableFind(table,node1,&j);CHKERRQ(ierr);
    j--;
    if (j > -1 && j < water->nvertex) {
      edgelist[2*i] = j;
    } else PetscCheck(j <= -1 || j >= water->nvertex,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"edge %"  PetscInt_FMT " does not have node1 %" PetscInt_FMT,i,node1);

    ierr = PetscTableFind(table,node2,&j);CHKERRQ(ierr);
    j--;
    if (j > -1 && j < water->nvertex) {
      edgelist[2*i+1] = j;
    } else PetscCheck(j <= -1 || j >= water->nvertex,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"edge %" PetscInt_FMT " does not have node2 %" PetscInt_FMT,i,node2);
  }
  ierr = PetscTableDestroy(&water->table);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode WashReadInputFile(PetscInt nsubnet,char filename[][PETSC_MAX_PATH_LEN])
{
  PetscErrorCode    ierr;
  FILE              *fp;
  PetscInt          fNum = 0,len;
  char              tmpfile[MAXLINE];

  PetscFunctionBegin;
  /* opening file for reading */
  fp = fopen(filename[0],"rb");
  if (fp == NULL) {
    perror("");
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error opening file");
  }
  while (fgets(tmpfile,MAXLINE,fp) && fNum<nsubnet) {
    len = strlen(tmpfile)-1;
    if (tmpfile[len]=='\n') tmpfile[len]='\0';
    if (tmpfile[0] == '#') continue;
    ierr = PetscStrcpy(filename[fNum],tmpfile);CHKERRQ(ierr);
    //printf("fNum %d,  fname %s\n",fNum,filename[fNum]);
    fNum++;
  }
  fclose(fp);

  PetscCheck(fNum >= nsubnet,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "num of filenames %" PetscInt_FMT " < nsubnet %" PetscInt_FMT,fNum,nsubnet);
  PetscFunctionReturn(0);
}
