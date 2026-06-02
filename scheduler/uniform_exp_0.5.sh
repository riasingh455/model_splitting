#!/bin/bash

f_10 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
}


f_9 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
}


f_8 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
}


f_7 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
}


f_6 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
}


f_5 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
}


f_4 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
}


f_3 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
}


f_2 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[14]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[14]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[10]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[10]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[11]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[11]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[13]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[13]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[12]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[12]}_5_4_2.log &" &
    wait
}


f_1 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[9]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[9]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[5]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[5]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[6]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[6]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[8]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[8]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[7]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[7]}_5_4_2.log &" &
    wait
}


f_0 () {
    dt=$(date -d '+60 seconds' +%s)
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 0 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[4]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[4]}_5_0_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 1 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[0]}_5_1_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 2 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[1]}_5_2_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 3 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[3]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[3]}_5_3_2.log &" &
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
    command="[[ $(ps -aex | grep 'onnxtest' | wc -l) -gt 1 ]] && wait $(ps -aex | grep 'onnxtest' | awk '{print $1}');python3 onnxtest_w_json_bs.py 30 ./resnet18_splits_5_2 4 4 1 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[2]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos${nodes[2]}_5_4_2.log &" &
    wait
}


rng_gen () {
        len=$1
        rng_arr=()
        while [[ ${#rng_arr[@]} -lt $conc_val ]]
        do
                rng_arr+=($[ $RANDOM % $len ])
                rng_arr=( $(for i in ${rng_arr[@]}; do echo $i; done | sort -u) )
        done
        echo "${rng_arr[@]}"
}

node_select () {
        # valid_nodes=($(cat ${path_prefix}/full_sweep/${node_prefix}_middling))
        #hardcodeed to ${stuck_dev} here
        all_nodes=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep -v "${stuck_dev}" | grep "${node_prefix}" | awk '{print$1}')
        #node_list=( $(for i in ${all_nodes[@]}; do [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${i}" | wc -l) -gt 0 ]] && echo "${i}"; done) )
	node_list=()
        for n_check in ${all_nodes[@]};
	do
		# if [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${n_check}" | wc -l) -gt 0 ]]
		#if [[ $(cat ${path_prefix}/aot_splitter/${node_prefix}.model.perf | grep -w "${n_check}" | wc -l) -gt 0 ]]
		if [[ $(cat ${path_prefix}/aot_splitter/${node_prefix}.red_blue.op | grep -w "${n_check}" | wc -l) -gt 0 ]]
		then
			hname=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check 'hostname')
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
			if [[ $n_check == $hname ]]
			then
				node_list+=($n_check)
            		fi
		fi
	done
	#echo "list ${node_list[@]}"

        ind_arr=( $(rng_gen "${#node_list[@]}") )
        node_select=( $(for i in ${ind_arr[@]}; do echo "${node_list[$i]}"; done) )
        echo "${node_select[@]}"
}

model_type=$1
model_split=$2
node_prefix=$3
inter=$4
path_prefix="/home/animesh/test_model_split/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] || [[ -z $inter ]]
# if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]]
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y) <split version> (int, 0(nw), 1(comp))" && exit
	# echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y)" && exit
fi


conc_val=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep "${node_prefix}" | awk '{print$1}' | wc -l)
#stuck_dev="bramble-4-1-29"

for repeat in {1..10}
do
    nodes=( $(node_select) )
    #hardcoded python commands and waits and node selections go here!
    #don't forget to add done at the end!
    sleep 0.0
    f_0 &

    sleep 0.5
    f_1 &

    sleep 1.0
    f_2 &

    sleep 1.5
    f_3 &

    sleep 2.0
    f_4 &

    sleep 2.5
    f_5 &

    sleep 3.0
    f_6 &

    sleep 3.5
    f_7 &

    sleep 4.0
    f_8 &

    sleep 4.5
    f_9 &

    sleep 5.0
    f_10 &
done
