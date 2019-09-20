#! /bin/bash

filename=$(basename -- "$1")
filepath=$(dirname -- "$1")

echo "$1"
if [ "$filename" = "rawlog.txt" ]; then
    filtINSERT="./${filepath}/filtoutINSERT.txt"
    filtADDVAL="./${filepath}/filtoutADDVAL.txt"
    nprocess="./${filepath}/nprocess.txt"
    packsizeINSERT="./${filepath}/packsizeINSERT.txt"
    packsizeADDVAL="./${filepath}/packsizeADDVAL.txt"
    VecDotTime="./${filepath}/vecdottime.txt"
    VecDotFlops="./${filepath}/vecdotflops.txt"
    NodeNum="./${filepath}/nodenum.txt"
    CellNum="./${filepath}/cellnum.txt"
    Fields="./${filepath}/fields.txt"
    Overlap="./${filepath}/overlap.txt"
    echo "Do you wish to overwrite?"
    select yne in "Yes" "No" "Exit"; do
        case $yne in
            Yes ) >./$filtINSERT;
                >./$filtADDVAL;
                >./$packsizeINSERT;
                >./$packsizeADDVAL;
                >./$nprocess;
                >./$VecDotTime;
                >./$VecDotFlops;
                >./$NodeNum;
                >./$CellNum;
                >./$Fields;
                >./$Overlap;
                break;;
            No ) "GREPPED">>./$filtINSERT;
                "GREPPED">>./$filtADDVAL;
                "GREPPED">>./$packsizeINSERT;
                "GREPPED">>./$packsizeADDVAL;
                "GREPPED">>./$nprocess;
                "GREPPED">>./$VecDotTime;
                "GREPPED">>./$VecDotFlops;
                "GREPPED">>./$NodeNum;
                "GREPPED">>./$CellNum;
                "GREPPED">>./$Fields;
                "GREPPED">>./$Overlap;
                break;;
            Exit ) exit;;
        esac
    done
else
    echo "Wrong input file $filename! Exiting"
    exit 1
fi

echo "grepping..."
grep "CommINSERT" --line-buffered ./$1 | awk '{print $4}' >> ./$filtINSERT
grep "CommADDVAL" --line-buffered ./$1 | awk '{print $4}' >> ./$filtADDVAL
grep "./exspeedtest on a" --line-buffered ./$1 | awk '{print $8}' >> ./$nprocess
grep "CommINSERT" --line-buffered ./$1 | awk '{print $9}' >> ./$packsizeINSERT
grep "CommADDVAL" --line-buffered ./$1 | awk '{print $9}' >> ./$packsizeADDVAL
grep "CommGlblVecDot" --line-buffered ./$1 | awk '{print $4}' >> ./$VecDotTime
grep "CommGlblVecDot" --line-buffered ./$1 | awk '{print $6}' >> ./$VecDotFlops
grep "Global Node Num" --line-buffered ./$1 | sed 's:.*>::' >> ./$NodeNum
grep "Global Cell Num" --line-buffered ./$1 | sed 's:.*>::' >> ./$CellNum
grep "Fields" --line-buffered ./$1 | sed 's:.*>::' | sed 's:(.*::' >> ./$Fields
grep "overlap" --line-buffered ./$1 | sed 's:.*>::' >> ./$Overlap
echo "done"
