#!/bin/bash
#for dir in /media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia\ \(From\ Moira\ 2020\)/Raw\ files/R2*     # list directories in the form "/tmp/dirname/"
#do
    #cd ${dir%*}      # remove the trailing "/"
    #num=$dir | sed "s/.*files\/R/"
    #echo $num      # remove the trailing "/"
    #diff “$dir/BatSearch.csv” “/am/state-opera/home1/listanvirg/Documents/Bat_TESTS/Test_1/R$num_Results.csv” > out
    #errFN=$(grep "<.*Moira" out | wc -l)
    #errFP=$(grep ">.*Moira" out | wc -l)
    #nP=$(grep "Moira" BatSearch.csv | wc -l)
    #count=$(grep "bmp" BatSearch.csv | wc -l)
    #echo ${dir%*} $errFN $errFP $nP $count
    #cd ..
#done


d1=“/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia_From_Moira_2020/Raw_files/”
d2=“/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia_From_Moira_2020/Raw_files/”
echo $d1
for dir in “$d1”/R*
do
    #echo $dir
    num=$dir | sed “s/.*files\/R//“; echo $dir
    diff “$dir/test” “$d2/R1/test”
done
