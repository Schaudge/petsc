# -*- mode: makefile-gmake -*-
#
#    Contains rules that work on a single directory (which are called from $PETSC_DIR/makefile) to
#       *  generate .md files for manual pages from comments in the source code using Sowing
#       *  find the use of PETSc routines, for example KSPSolve(), in tutorials and saves their source
#       *  find the implementations of methods, for example KSPSolve_GMRES(), in the source code and saves their locations
#       *  convert PETSc source code to HTML
#       These are called the classic documentation but they are integral to the PETSc documentation

#
# The following additional variables are used by PETSc documentation targets
#
# LIBNAME    - library name
# SOURCE     - source files
# SOURCEALL  - sources and includes
# SOURCED    - sources/includes [but not Fortran - for doc parsing]
#
# Note that EXAMPLESALL is only used in the tutorial directories and SOURCED only in the non-tutorials and tests directories
#
LIBNAME     = ${INSTALL_LIB_DIR}/${LIBBASE}.${AR_LIB_SUFFIX}
SOURCE      = `ls *.c *.cxx *.F *.F90 *.cu *.cpp           2> /dev/null`
SOURCEALL   = `ls *.c *.cxx *.F *.F90 *.cu *.cpp *.h *.hpp 2> /dev/null`
SOURCED     = `ls *.c *.cxx           *.cu *.cpp *.h *.hpp 2> /dev/null`
EXAMPLESALL = `ls *.c *.cxx *.F *.F90 *.cu *.cpp           2> /dev/null`

# This is included in this file so it may be used from any source code PETSc directory
libs: ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/files ${PETSC_DIR}/${PETSC_ARCH}/tests/testfiles
	+@r=`echo "${MAKEFLAGS}" | grep ' -j'`; \
        if [ "$$?" = 0 ]; then make_j=""; else make_j="-j${MAKE_NP}"; fi; \
	r=`echo "${MAKEFLAGS}" | grep ' -l'`; \
        if [ "$$?" = 0 ]; then make_l=""; else make_l="-l${MAKE_LOAD}"; fi; \
        cmd="${OMAKE_PRINTDIR} -f gmakefile $${make_j} $${make_l} ${MAKE_PAR_OUT_FLG} V=${V} libs"; \
        cd ${PETSC_DIR} && echo $${cmd} && exec $${cmd}

# Performs the specified action on all source/include directories except output; used by c2html and cleanhtml
tree: ${ACTION}
	-@for dir in `ls -d */ 2> /dev/null` foo ;  do \
            if [[ $${dir} != "doc/" && $${dir} != "output/" ]]; then \
              if [[ -f $${dir}makefile ]]; then \
	        (cd $$dir ; ${OMAKE} ACTION=${ACTION} PETSC_ARCH=${PETSC_ARCH}  LOC=${LOC} tree) ; \
              fi; \
           fi; \
	 done

#   Rule for generating html code from C and Fortran
#   Can run (and is) in parallel
html:
	-@export htmlmap_tmp=$$(mktemp) ;\
          petsc_dir=$$(realpath ${PETSC_DIR}); LOCDIR=$$(pwd | sed s"?$${petsc_dir}/??"g)/; \
          sed -e s?man+manualpages/?man+HTML_ROOT/manualpages/? ${HTMLMAP} > $$htmlmap_tmp ;\
          cat ${PETSC_DIR}/doc/manualpages/mpi.www.index >> $$htmlmap_tmp ;\
          ROOT=`echo $${LOCDIR} | sed -e s?/[-a-z_0-9]*?/..?g -e s?src/??g -e s?include/??g` ;\
          loc=`pwd | sed -e s?\$${PETSC_DIR}?$${LOC}/?g -e s?/disks??g`;  \
          ${MKDIR} -p $${loc} ;\
          current_git_sha=$$(git rev-parse HEAD) ;\
          rel_dir=$$(echo ${PWD} | sed "s%^${PETSC_DIR}/%%") ;\
          for i in ${SOURCEALL} ${EXAMPLESALL} foo ; do\
            if [ -f $$i ]; then \
              idir=`dirname $$i`;\
              if [ ! -d $${loc}/$${idir} ]; then ${MKDIR} -p $${loc}/$${idir}; fi ; \
              iroot=`echo $$i | sed -e "s?[a-z.]*/??g"`;\
              IROOT=`echo $${i} | sed -e s?[.][.]??g` ;\
              if [ "$${IROOT}" != "$${i}" ] ; then \
                IROOT=".."; \
              else \
                IROOT=$${ROOT};\
              fi;\
              ${RM} $${loc}/$$i.html; \
              echo "<center><a href=\"https://gitlab.com/petsc/petsc/-/blob/$$current_git_sha/$$rel_dir/$$i\">Actual source code: $${iroot}</a></center><br>" > $${loc}/$$i.html; \
              sed -E "s/PETSC[A-Z]*_DLLEXPORT//g" $$i | \
              ${C2HTML} -n | \
              awk '{ sub(/<pre width="80">/,"<pre width=\"80\">\n"); print }' | \
              ${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/fixinclude.py $$i $${PETSC_DIR} | \
              grep -E -v '(PetscValid|#if !defined\(__|#define __|#undef __|EXTERN_C )' | \
              ${MAPNAMES} -map $$htmlmap_tmp -inhtml | sed -e s?HTML_ROOT?$${IROOT}?g >> $${loc}/$$i.html ; \
            fi; \
          done ;\
          loc=`pwd | sed -e s?\$${PETSC_DIR}?$${LOC}/?g -e s?/disks??g`; ${RM} $${loc}/index.html; \
          if [ -f ${PETSC_DIR}/doc/manualpages/MANSECHeaders/${MANSEC} ] ; then \
            cat ${PETSC_DIR}/doc/manualpages/MANSECHeaders/${MANSEC} | sed -e "s?<A HREF=\"PETSC_DIR[a-z/]*\">Examples</A>?<A HREF=\"$${ROOT}/manualpages/${MANSEC}\">Manual pages</A>?g" -e "s?PETSC_DIR?$${ROOT}/?g"> $${loc}/index.html; \
          else \
            touch $${loc}/index.html; \
          fi; \
          echo "<p>" >> $${loc}/index.html ;\
          loc=`pwd | sed -e s?\$${PETSC_DIR}?$${LOC}/?g -e s?/disks??g`;\
          if [ "${EXAMPLESC}" != "" ] ; then \
            for file in ${EXAMPLESC} foo ; do \
              if [ -f $$file ]; then \
                cmess=`grep "static\( const\)\? char help" $${file} | cut -d\" -f2 | cut -d\. -f1`; \
                echo "<a href=\"$${file}.html\">$${file}: $${cmess}</a><br>" >> $${loc}/index.html;\
              fi; \
            done ;\
          else \
            for file in `ls -d */ 2> /dev/null` foo; do \
              if [ -d $$file ]; then \
                echo "<a href=\"$${file}/\">$${file}/</a><br>" >> $${loc}/index.html; \
              fi; \
            done; \
            echo " " >> $${loc}/index.html; \
            for file in ${SOURCEALL} foo ; do \
              if [ -f $$file ]; then \
                echo "<a href=\"$${file}.html\">$${file}</a><br>" >> $${loc}/index.html; \
              fi; \
            done; \
          fi ;\
          ${RM} $$htmlmap_tmp

cleanhtml:
	-@${RM} index.html *.{c,cxx,cu,F,F90,h,h90,m}.html *.{c,cxx,cu}.gcov.html

