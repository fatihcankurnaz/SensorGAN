#!/usr/bin/env bash

start_counter=7001
end_counter=7051
while [ $end_counter -le 12901 ]
do
echo "##################"
echo "##################"
echo "##################"
echo "##################"
echo "START VALUE"
echo "##################"
echo "##################"
echo "##################"
echo "##################"

echo $start_counter
echo "##################"
echo "##################"
echo "##################"
echo "##################"
echo "END VALUE"
echo "##################"
echo "##################"
echo "##################"
echo "##################"
echo $end_counter
python save_segmented_versions.py $start_counter $end_counter
((start_counter=end_counter))
((end_counter=start_counter+50))

done

echo "##################"
echo "##################"
echo "##################"
echo "##################"
echo "LAST SET 12901-12920"
echo "##################"
echo "##################"
echo "##################"
echo "##################"

python save_segmented_versions.py 12901 12920
