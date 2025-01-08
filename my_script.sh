#!/bin/bash

for year in {1991..1992}

do 
    rsync -a --info=progress2 "/mnt/s3fs/CHRTOUT/$year" "/media/volume/Imp_Data/CHRTOUT"
    echo "$year transfer for CHRTOUT has been complete"
    rsync -a --info=progress2 "/mnt/s3fs/FORCING/$year" "/media/volume/Imp_Data/FORCING"
    echo "$year transfer for FORCING has been complete"
done