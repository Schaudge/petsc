#!/bin/bash
set -e
set -x
git fetch -q origin maint
maint=$(git rev-parse FETCH_HEAD)
git fetch -q origin master
master=$(git rev-parse FETCH_HEAD)
if [ $(git merge-base --octopus $maint $master HEAD) =  $(git merge-base $master HEAD) ]
 then dest=$maint; deststr=origin/maint
 else dest=$master; deststr=origin/master
fi
if [ -z "$(git diff HEAD...$dest .gitlab-ci.yml)" ]
  then printf "Success! Using current CI settings as in gitlab-ci.yml in $deststr!\n"
  else printf "ERROR! Using old CI settings in gitlab-ci.yml! Please rebase to $deststr to use current CI settings.\n"; exit 1
fi

