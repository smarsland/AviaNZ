#!/bin/bash

for dir in /media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia (From Moira 2020)/Raw \ files/R*     # list directories in the form "/tmp/dirname/"
do
    cd ${dir%*}      # remove the trailing "/"
    diff BatSearch.csv Results.csv > out
    errFN=$(grep "<.*Moira" out | wc -l)
    errFP=$(grep ">.*Moira" out | wc -l)
    nP=$(grep "Moira" BatSearch.csv | wc -l)
    count=$(grep "bmp" BatSearch.csv | wc -l)
    echo ${dir%*} $errFN $errFP $nP $count
    cd ..
done
