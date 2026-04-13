#!/bin/bash

nohup ./roofline_pi_controller_conc_no_slurm_no_temp.sh resnet18 children bramble-2-5 1 > test_bramble-2-5.out 

nohup ./roofline_pi_controller_conc_no_slurm_no_temp.sh resnet18 children bramble-2-5 4 > test_bramble-2-5.out


