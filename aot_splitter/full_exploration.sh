#!/bin/bash
sub=( "bramble-1-1" "bramble-1-3" "bramble-1-4" "bramble-2-1" "bramble-2-2" "bramble-2-4" "bramble-2-5" "bramble-2-6" "bramble-4-1" "bramble-4-2" "bramble-4-3" "bramble-4-5" "bramble-4-6")
for model in "tcn modules" "resnet18 children" "vit modules"
do
    waiters=()
    for s in ${sub[@]}
    do
        ./subcluster_exploration.sh $model $s &
        waiters+=($!)
    done
    wait ${waiters[@]}
    echo "model ${model} done for all subclusters!"
    sleep 60
done

