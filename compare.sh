#!/bin/bash
for dir in “/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia_From_Moira_2020/Moira_Annotation_Backup/R*”     # list directories in the form "/tmp/dirname/"
do
    #cd ${dir%*}      # remove the trailing "/"
    echo $dir
    num=$dir | sed “s/.*files\/R//“
    #num=$dir | sed "s/.*files\/R/"
    echo $num      # remove the trailing "/"
    diff “$dir/BatSearch.csv” “/am/state-opera/home1/listanvirg/Documents/Bat_TESTS/Test_1/R”"$num"“_Results.csv”  > out
    errFN=$(grep "<.*Moira" out | wc -l)
    errFP=$(grep ">.*Moira" out | wc -l)
    nP=$(grep "Moira" BatSearch.csv | wc -l)
    count=$(grep "bmp" BatSearch.csv | wc -l)
    echo ${dir%*} $errFN $errFP $nP $count
    #cd ..
done
