import os

path = '../logs/full_pipeline/bramble-4-1/'

#key is the device name
resnet = {}
tcn = {}
vit = {}


def average_automation(model, device_name):
    model_path = os.path.join(path, model)
    print(model_path)

    device = device_name        #change device name here
    average= []    #each index is a time slice, and each value is an average of all the runs in that time slice
    for timeslice in range(8):
        max_times = []
        counter = 0
        timeslice_path = os.path.join(model_path, str(timeslice))

        for run in range(1,11):
            run_path = os.path.join(timeslice_path, str(run))
            if not os.path.isdir(run_path):
                continue

            for filename in os.listdir(run_path):
                if filename.endswith(device + ".log"):
                    file_path = os.path.join(run_path, filename)

                    avg_times = []
                
                    with open(file_path, "r") as f:
                        for line_num, line in enumerate(f):
                            if line_num == 7:
                                line = line.strip()
                                parts = line.split()
                                avg_time = float(parts[1])
                                avg_times.append(avg_time)
                                break

                    if avg_times:
                        max_time = max(avg_times)
                        max_times.append(max_time)
                        counter += 1

        sum = 0
        if(counter):
            for i in max_times:
                sum += i                   
            final_avg = sum / counter
            average.append(final_avg)
        else:
            average.append(-1)      #to indicate slices that don't have an average for the device
    
    return average 


devices = ['2_0','2_1','5_0','5_1','5_2','5_3','5_4']
for i in devices:
    resnet[i] = average_automation('resnet18_children_onnx', i)
    tcn[i] = average_automation('tcn_modules_onnx', i)
    vit[i] = average_automation('vit_modules_onnx', i)

print("resnet")
print(resnet)
print("\ntcn")
print(tcn)
print("\nvit")
print(vit)


