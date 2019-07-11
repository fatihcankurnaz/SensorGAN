#!/usr/bin/env bash




for FILE in "/home/fatih/LidarLabelsCameraViewTest/2011_09_26_drive_0022_sync/"*; do
    b=$(basename $FILE)
    #echo "$b"
    value=$(echo $b | cut -f2 -d_)
    var3="/home/fatih/SegmentedInputTest/2011_09_26_drive_0022_sync/segmented_"
    var4="$var3$value"
    #echo "$var4"
    if [ ! -f $var4 ]; then
        echo "$var4"
        echo "$FILE"
        rm $FILE

    fi

done
#while [ $end_counter -le 12901 ]
#do
#    if test -f "$FILE"; then
#        echo "$FILE exist"
#    fi
#done
