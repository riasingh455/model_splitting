#!/bin/bash
pids=()

./roofline_pi_controller.sh resnet18 children bramble-1-1 > test_bramble-1-1.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-1-2 > test_bramble-1-2.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-1-3 > test_bramble-1-3.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-1-4 > test_bramble-1-4.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-2-1 > test_bramble-2-1.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-2-5 > test_bramble-2-5.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-2-6 > test_bramble-2-6.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-4-5 > test_bramble-4-5.out &
pids+=($!)

./roofline_pi_controller.sh resnet18 children bramble-4-6 > test_bramble-4-6.out &
pids+=($!)

wait ${pids[@]}