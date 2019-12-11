#! /bin/bash

clear_files() {
    > $filtINSERT;
    > $filtADDVAL;
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
    > $SFPackINSERT;
    > $SFUnPackINSERT;
    > $SFPackADDVAL;
    > $SFUnPackADDVAL;
}

append_files() {
    "GREPPED">> $filtINSERT;
    "GREPPED">> $filtADDVAL;
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
    "GREPPED">> $SFPackINSERT;
    "GREPPED">> $SFUnPackINSERT;
    "GREPPED">> $SFPackADDVAL;
    "GREPPED">> $SFUnPackADDVAL;
}

auto_flag=0;
while getopts ":a" opt; do
    case $opt in
        a)
            echo "auto flag triggered!" >&2;
            auto_flag=1;
            shift;
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2;
            break;;
    esac
done
for logfile in "$@"
do
    filename=$(basename -- "$logfile");
    filepath=$(dirname -- "$logfile");
    ID=${filename#*_};
    ID=${ID%.*};
    echo "File:             ${logfile}";
    echo "Logfile dir name: ${filepath}";
    echo "Logfile name:     ${filename}";
    echo "Logfile ID:       ${ID}"
    if [ "$filename" = "rawlog_${ID}.txt" ]; then
        filtINSERT="${filepath}/filtoutINSERT_${ID}.txt";
        filtADDVAL="${filepath}/filtoutADDVAL_${ID}.txt";
        nprocess="${filepath}/nprocess_${ID}.txt";
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
        SFPackINSERT="${filepath}/sfpackINSERT_${ID}.txt";
        SFUnPackINSERT="${filepath}/sfunpackINSERT_${ID}.txt";
        SFPackADDVAL="${filepath}/sfpackADDVAL_${ID}.txt";
        SFUnPackADDVAL="${filepath}/sfunpackADDVAL_${ID}.txt";
        if [ "$auto_flag" -eq "0" ]; then
            echo "Do you wish to overwrite?";
            select yne in "Yes" "No" "Exit"; do
                case $yne in
                    Yes ) clear_files;
                          break;;
                    No ) append_files;
                         break;;
                    Exit ) exit;;
                esac
            done
        else
            clear_files;
        fi
    else
        echo "Wrong input file $filename! Exiting";
        exit 1
    fi
    echo "=== Grepping ${filename} ===";
    echo "Populating ${filtINSERT}";
    grep "CommINSERT" --line-buffered $logfile | awk '{print $4}' >> $filtINSERT;
    echo "Populating ${filtADDVAL}";
    grep "CommADDVAL" --line-buffered $logfile | awk '{print $4}' >> $filtADDVAL;
    echo "Populating ${nprocess}";
    grep "/exspeedtest" --line-buffered $logfile | awk '{print $8}' >> $nprocess;
    echo "Populating ${packsizeINSERT}";
    grep "CommINSERT" --line-buffered $logfile | awk '{print $9}' >> $packsizeINSERT;
    echo "Populating ${packsizeADDVAL}";
    grep "CommADDVAL" --line-buffered $logfile | awk '{print $9}' >> $packsizeADDVAL;
    echo "Populating ${VecDotTime}";
    grep "CommGlblVecDot" --line-buffered $logfile | awk '{print $4}' >> $VecDotTime;
    echo "Populating ${VecDotFlops}";
    grep "CommGlblVecDot" --line-buffered $logfile | awk '{print $6}' >> $VecDotFlops;
    echo "Populating ${ZeroVecDotTime}";
    grep "CommZEROVecDot" --line-buffered $logfile | awk '{print $4}' >> $ZeroVecDotTime;
    echo "Populating ${ZeroVecDotFlops}";
    grep "CommZEROVecDot" --line-buffered $logfile | awk '{print $6}' >> $ZeroVecDotFlops;
    echo "Populating ${NodeNum}";
    grep "Global Node Num" --line-buffered $logfile | sed 's:.*>::' >> $NodeNum;
    echo "Populating ${CellNum}";
    grep "Global Cell Num" --line-buffered $logfile | sed 's:.*>::' >> $CellNum;
    echo "Populating ${Overlap}";
    grep "overlap" --line-buffered $logfile | sed 's:.*>::' >> $Overlap;
    echo "Populating ${Order}";
    grep "petscspace_degree" --line-buffered $logfile | awk '{print $2}' >> $Order;
    echo "Populating ${CompNum}";
    grep "num_fields" --line-buffered $logfile | awk '{print $2}' >> $CompNum;
    echo "Populating ${GVS}";
    grep "GLOBAL Vector GLOBAL Size" --line-buffered $logfile | awk '{print $5}' >> $GVS;
    echo "Populating ${SFPackINSERT}";
    grep "CommINSERT" -B 5 --line-buffered $logfile | grep "SFPack" --line-buffered | awk '{printf ("%s\n",$4)}' >> $SFPackINSERT;
    echo "Populating ${SFUnPackINSERT}";
    grep "CommINSERT" -B 5 --line-buffered $logfile | grep "SFUnpack" --line-buffered | awk '{printf ("%s\n",$4)}' >> $SFUnPackINSERT;
    echo "Populating ${SFPackADDVAL}";
    grep "CommADDVAL" -B 5 --line-buffered $logfile | grep "SFPack" --line-buffered | awk '{printf ("%s\n",$4)}' >> $SFPackADDVAL;
    echo "Populating ${SFUnPackADDVAL}";
    grep "CommADDVAL" -B 5 --line-buffered $logfile | grep "SFUnpack" --line-buffered | awk '{printf ("%s\n",$4)}' >> $SFUnPackADDVAL;
    echo "=== Done ===";
done
