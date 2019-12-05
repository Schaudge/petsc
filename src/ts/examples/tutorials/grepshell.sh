#! /bin/bash

for logfile in "$@"
do
    filename=$(basename -- "$logfile");
    filepath=$(dirname -- "$logfile");
    ID=${filename#*_};
    ID=${ID%.*};
    echo "$filename";
    echo "$filepath";
    echo "$logfile";
    echo "$ID"
    if [ "$filename" = "rawlog_${ID}.txt" ]; then
        filtINSERT="${filepath}/filtoutINSERT_${ID}.txt";
        filtADDVAL="${filepath}/filtoutADDVAL_${ID}.txt";
        nprocess="${filepath}/nprocess_${ID}.txt";
        cellprank="${filepath}/cellprank_${ID}.txt"
        packsizeINSERT="${filepath}/packsizeINSERT_${ID}.txt";
        packsizeADDVAL="${filepath}/packsizeADDVAL_${ID}.txt";
        VecDotTime="${filepath}/vecdottime_${ID}.txt";
        VecDotFlops="${filepath}/vecdotflops_${ID}.txt";
        ZeroVecDotTime="${filepath}/zero_vecdottime_${ID}.txt"
        ZeroVecDotFlops="${filepath}/zero_vecdotflops_${ID}.txt";
        NodeNum="${filepath}/nodenum_${ID}.txt";
        CellNum="${filepath}/cellnum_${ID}.txt";
        Overlap="${filepath}/overlap_${ID}.txt";
        Order="${filepath}/feorder_${ID}.txt";
        CompNum="${filepath}/compnum_${ID}.txt";
        GVS="${filepath}/globvecsize_${ID}.txt";
        echo "Do you wish to overwrite?";
        select yne in "Yes" "No" "Exit"; do
            case $yne in
                Yes ) > $filtINSERT;
                      > $filtADDVAL;
                      > $cellprank;
                      > $packsizeINSERT;
                      > $packsizeADDVAL;
                      > $nprocess;
                      > $VecDotTime;
                      > $VecDotFlops;
                      > $ZeroVecDotTime;
                      > $ZeroVecDotFlops;
                      > $NodeNum;
                      > $CellNum;
                      > $Overlap;
                      > $Order;
                      > $CompNum;
                      > $GVS;
                      break;;
                No ) "GREPPED">> $filtINSERT;
                     "GREPPED">> $filtADDVAL;
                     "GREPPED">> $cellprank;
                     "GREPPED">> $packsizeINSERT;
                     "GREPPED">> $packsizeADDVAL;
                     "GREPPED">> $nprocess;
                     "GREPPED">> $VecDotTime;
                     "GREPPED">> $VecDotFlops;
                     "GREPPED">> $ZeroVecDotTime;
                     "GREPPED">> $ZeroVecDotFlops;
                     "GREPPED">> $NodeNum;
                     "GREPPED">> $CellNum;
                     "GREPPED">> $Overlap;
                     "GREPPED">> $Order;
                     "GREPPED">> $CompNum;
                     "GREPPED">> $GVS;
                     break;;
                Exit ) exit;;
            esac
        done
    else
        echo "Wrong input file $filename! Exiting";
        exit 1
    fi

    echo "grepping...";
    grep "CommINSERT" --line-buffered $logfile | awk '{print $4}' >> $filtINSERT;
    grep "CommADDVAL" --line-buffered $logfile | awk '{print $4}' >> $filtADDVAL;
    grep "./exspeedtest on a" --line-buffered $logfile | awk '{print $8}' >> $nprocess;
    grep "CommINSERT" --line-buffered $logfile | awk '{print $9}' >> $packsizeINSERT;
    grep "CommADDVAL" --line-buffered $logfile | awk '{print $9}' >> $packsizeADDVAL;
    grep "CommGlblVecDot" --line-buffered $logfile | awk '{print $4}' >> $VecDotTime;
    grep "CommGlblVecDot" --line-buffered $logfile | awk '{print $6}' >> $VecDotFlops;
    grep "CommZEROVecDot" --line-buffered $logfile | awk '{print $4}' >> $ZeroVecDotTime;
    grep "CommZEROVecDot" --line-buffered $logfile | awk '{print $6}' >> $ZeroVecDotFlops;
    grep "Global Node Num" --line-buffered $logfile | sed 's:.*>::' >> $NodeNum;
    grep "Global Cell Num" --line-buffered $logfile | sed 's:.*>::' >> $CellNum;
    grep "overlap" --line-buffered $logfile | sed 's:.*>::' >> $Overlap;
    grep "petscspace_degree" --line-buffered $logfile | awk '{print $2}' >> $Order;
    grep "num_fields" --line-buffered $logfile | awk '{print $2}' >> $CompNum;
    grep "Global Vector Size" --line-buffered $logfile | awk '{print $4}' >> $GVS;
    echo "done";
done
