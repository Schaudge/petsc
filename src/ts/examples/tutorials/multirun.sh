#! /bin/bash

#COBALT -t 60
#COBALT -q default
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A $COBALT_PROJDM

echo "Start"
echo "-n total mpi ranks (-N*-n)"
echo "-N ranks per node"
echo "-cc how ranks are bound to cores, depth allows specification"
echo "-d hardware threads per rank"
echo "-j hardware threads per core (max 4)"

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++                   DEPLOYED VERSION                    +++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo "Creating Directories"
now=$(date +"%m_%d_%Y")
dirname="med_out_dep_$now"
mkdir -p ./$dirname
filtINSERT="./${dirname}/filtoutINSERT.txt"
filtADDVAL="./${dirname}/filtoutADDVAL.txt"
nprocess="./${dirname}/nprocess.txt"
rawlog="./${dirname}/rawlog.txt"
cellprank="./${dirname}/cellprank.txt"
packsizeINSERT="./${dirname}/packsizeINSERT.txt"
packsizeADDVAL="./${dirname}/packsizeADDVAL.txt"
runERROR="./${dirname}/runerror.txt"

echo "Resetting full log and filtered outputs..."
>./$filtINSERT
>./$filtADDVAL
>./$nprocess
>./$rawlog
>./$cellprank
>./$packsizeINSERT
>./$packsizeADDVAL
>./$runERROR
echo "done"

maxcount=18
counter=10
cells=2000000
ranks=$((2**$counter))
failcount=0
runcount=0
maxfail=5
maxruncount=1
failsafe=$(($maxfail+$maxruncount))
echo "Max number of Iterations:         	$(((($maxcount-$counter))*$maxruncount))"
echo "looping..."
echo "-----------"
until [ $counter -gt $maxcount ]
do
    cellprankval=$(bc <<< "scale=3; $cells/$ranks")
    echo "counter:			   	$counter"
    echo "run count:				$runcount"
    echo "fail count:				$failcount"
    echo "current number of MPI ranks:		$ranks"
    echo "approx cells/rank:			$cellprankval"
    echo "start time:				$(date -u)"
    runFlag="+++++++++++++++++++++++++++++ RUN $counter RANKS $ranks +++++++++++++++++++++++++++++"
    runCleanFlag="_________________________ RUN $counter RANKS $ranks CLEAN _________________________"
    echo "$runFlag">>./$runERROR
    SECONDS=0
    aprun -n $ranks -cc depth -d 1 -j 4 ./exspeedtest -speed -f ssthimble2M.med -log_view 2>> ./$runERROR 1>> ./$rawlog
    duration=$SECONDS
    echo "$runCleanFlag">>./$runERROR
    line=$(grep -A1 "$runFlag" ./$runERROR | tail -n 1 | grep -q "$runCleanFlag")
    if [ $? -eq 0 ] || [ $failcount -eq $maxfail ]; then
	if [ $failcount -eq $maxfail ]; then
            echo "=============================================================================================">>./$runERROR
            echo -e "\t\t\t\t RUN $counter RANKS $ranks ERROR BUT EXITING ANYWAY">>./$runERROR
            echo "=============================================================================================">>./$runERROR
            ((counter++))
            runcount=0
            failcount=0
            ranks=$((2**$counter))
	    echo "+++++++++++++++++++++++++++ End of RUN $counter RANKS $ranks Attempt +++++++++++++++++++++++++++++">>./$rawlog
        else
            echo "+++++++++++++++++++++++++++++ RUN $counter RANKS $ranks NO ERROR +++++++++++++++++++++++++++++">>./$runERROR
	    echo "$cellprankval">>./$cellprank
	fi;
	if [ $runcount -eq $maxruncount ]; then
            ((counter++))
            runcount=0
            failcount=0
            ranks=$((2**$counter))
	    echo "+++++++++++++++++++++++++++ End of RUN $counter RANKS $ranks Attempt +++++++++++++++++++++++++++++">>./$rawlog
        else
            ((runcount++))
        fi;
    else
        echo "Command failed"
	((failcount++))
        ranks=$(($ranks-1))
    fi
    if [ $(($failcount+$runcount)) -eq $failsafe ]; then
	echo "=============================================================================================">>./$runERROR
        echo -e "\t\t\t\t RUN $counter RANKS $ranks FAILSAFE">>./$runERROR
        echo "=============================================================================================">>./$runERROR
    fi
    echo "end time:				$(date -u)"
    echo "runtime:                         	$(($duration / 60)) minutes and $(($duration % 60)) seconds"
    echo "-----------"
done
echo "--------------------------- Successful exit! ---------------------------">>./$rawlog
trap finish EXIT
function finish {
    echo "grepping..."
    grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtINSERT
    grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $4}' >> ./$filtADDVAL
    grep "./exspeedtest on a" --line-buffered ./$rawlog | awk '{print $8}' >> ./$nprocess
    grep "CommINSERT" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeINSERT
    grep "CommADDVAL" --line-buffered ./$rawlog | awk '{print $9}' >> ./$packsizeADDVAL
    echo "done"
}
