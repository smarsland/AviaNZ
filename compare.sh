#!/bin/bash

for dir in R*     # list directories in the form "/tmp/dirname/"
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
