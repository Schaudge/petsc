#! /bin/bash

#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A $COBALT_PROJ

echo "Start"
echo "-n total mpi ranks (-N*-n)"
echo "-N ranks per node"
echo "-cc how ranks are bound to cores, depth allows specification"
echo "-d hardware threads per rank"
echo "-j hardware threads per core (max 4)"
echo "aprun -n 256 -N 256 -cc depth -d 1 -j 4 ./exspeedtest -f ssthimble1M.msh4 -disp -log_view"

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++                     DEBUG VERSION                     +++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

mkdir -p
filtINSERT='filtoutdebugINSERT.txt'
filtADDVAL='filtoutdebugADDVAL.txt'
nprocess='nprocessdebug.txt'
rawlog='rawlogdebug.txt'
cellprank='cellprankdebug.txt'
packsizeINSERT='packsizedebugINSERT.txt'
packsizeADDVAL='packsizedebugADDVAL.txt'

echo "resetting full log and filtered outputs..."
>./$filtINSERT
>./$filtADDVAL
>./$nprocess
>./$rawlog
>./$cellprank
>./$packsizeINSERT
>./$packsizeADDVAL
echo "done"

maxcount=9
counter=0
cells=1000000
echo "Max number of Iterations:		$(($maxcount-$counter))"
echo "looping..."
echo "-----------"
until [ $counter -gt $maxcount ]
do
    #ranks=$((2**$counter))
    ranks=256
    cellprank=$(bc <<< "scale=3; $cells/$ranks")
    echo "counter:                         	$counter"
    echo "current number of MPI ranks:     	$ranks"
    echo "approx cells/rank:			$cellprank"
    echo "start time:                      	$(date -u)"
    SECONDS=0
    aprun -n $ranks -N 256 -cc depth -d 1 -j 4 ./exspeedtest -f ssthimble1M.med -speed -log_view >> ./$rawlog
    echo "+++++++++++++++++++++++++++++ End of Log +++++++++++++++++++++++++++++++">>./$rawlog
    ((counter++))
    duration=$SECONDS
    echo "end time:			 	$(date -u)"
    echo "runtime:                         	$(($duration / 60)) minutes and $(($duration % 60)) seconds"
    echo "$cellprank">>./$cellprank
    echo "-----------"
done
echo "--------------------------- Successful exit! ---------------------------">>./$rawlog
echo "done"
echo "grepping..."
grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtINSERT
grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtADDVAL
grep "./exspeedtest on a" --line-buffered ./$rawlog | awk '{print $8}' >> ./$nprocess
grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeINSERT
grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeADDVAL
echo "done"
