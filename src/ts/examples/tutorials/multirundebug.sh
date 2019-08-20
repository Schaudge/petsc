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

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++                     DEBUG VERSION                     +++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo "Creating Directories"
now=$(date +"%m_%d_%Y")
dirname="med_out_debug_$now"
mkdir -p ./$dirname
filtINSERT="./${dirname}/filtoutdebugINSERT.txt"
filtADDVAL="./${dirname}/filtoutdebugADDVAL.txt"
nprocess="./${dirname}/nprocessdebug.txt"
rawlog="./${dirname}/rawlogdebug.txt"
cellprank="./${dirname}/cellprankdebug.txt"
packsizeINSERT="./${dirname}/packsizedebugINSERT.txt"
packsizeADDVAL="./${dirname}/packsizedebugADDVAL.txt"

echo "resetting full log and filtered outputs..."
>./$filtINSERT
>./$filtADDVAL
>./$nprocess
>./$rawlog
>./$cellprank
>./$packsizeINSERT
>./$packsizeADDVAL
echo "done"

maxcount=0
counter=0
cells=1000000
echo "Max number of Iterations:		$(($maxcount-$counter))"
echo "looping..."
echo "-----------"
until [ $counter -gt $maxcount ]
do
    #ranks=$((2**$counter))
    ranks=256
    cellprankval=$(bc <<< "scale=3; $cells/$ranks")
    echo "counter:                         	$counter"
    echo "current number of MPI ranks:     	$ranks"
    echo "approx cells/rank:			$cellprankval"
    echo "start time:                      	$(date -u)"
    SECONDS=0
    aprun -n $ranks -cc depth -d 1 -j 4 ./exspeedtest -speed -f ssthimble8M.med -log_view >> ./$rawlog
    #aprun -n $ranks -N 256 -cc depth -d 1 -j 4 ./exspeedtest -speed -dim $dim -level $lvl -n $nface -nf $nfield -maxcom -$maxcom -log_view >> ./$rawlog
    echo "+++++++++++++++++++++++++++++ End of Log +++++++++++++++++++++++++++++++">>./$rawlog
    ((counter++))
    duration=$SECONDS
    echo "end time:			 	$(date -u)"
    echo "runtime:                         	$(($duration / 60)) minutes and $(($duration % 60)) seconds"
    echo "$cellprankval">>./$cellprank
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
