#! /bin/bash

#COBALT -t 180
#COBALT -q default
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A $COBALT_PROJ

echo "Start"
echo "-n total mpi ranks (-N*-n)"
echo "-N ranks per node"
echo "-cc how ranks are bound to cores, depth allows specification"
echo "-d hardware threads per rank"
echo "-j hardware threads per core (max 4)"

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++                   DEPLOYED VERSION                    +++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

filtINSERT='filtoutINSERT.txt'
filtADDVAL='filtoutADDVAL.txt'
nprocess='nprocess.txt'
rawlog='rawlog.txt'
cellprank='cellprank.txt'
packsizeINSERT='packsizeINSERT.txt'
packsizeADDVAL='packsizeADDVAL.txt'

echo "resetting full log and filtered outputs..."
>./$filtINSERT
>./$filtADDVAL
>./$nprocess
>./$rawlog
>./$cellprank
>./$packsizeINSERT
>./$packsizeADDVAL
echo "done"

maxcount=18
counter=11
cells=8000000
echo "Max number of iterations:		$(($maxcount-$counter))"
echo "looping..."
echo "-----------"
until [ $counter -gt $maxcount ]
do
    ranks=$((2**$counter))
    cellprank=$(bc <<< "scale=3; $cells/$ranks")
    echo "counter:			   	$counter"
    echo "current number of MPI ranks:         	$ranks"
    echo "approx cells/rank:                    $cellprank"
    echo "start time:                      	$(date -u)"
    SECONDS=0
    aprun -n $ranks -N 256 -cc depth -d 1 -j 4 ./exspeedtest -f ssthimble8M.msh4 -speed -log_view >> ./$rawlog
    echo "+++++++++++++++++++++++++++ End of Log ++++++++++++++++++++++++++++++++++">>./$rawlog
    ((counter++))
    duration=$SECONDS
    echo "end time:                      	$(date -u)"
    echo "runtime:                         	$(($duration / 60)) minutes and $(($duration % 60)) seconds"
    echo "$cellprank">>./$cellprank
    echo "-----------"
done
echo "--------------------------- Successful exit! ---------------------------">>./rawlog.txt
echo "done"
echo "grepping..."
grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtINSERT
grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtADDVAL
grep "./exspeedtest on a" --line-buffered ./$rawlog | awk '{print $8}' >> ./$nprocess
grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeINSERT
grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeADDVAL
echo "done"
