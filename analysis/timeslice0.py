import os

path = '../logs/full_pipeline/bramble-4-1/'
max_times = []
cores = []
counter = 0
device = '2_0'

model_path = os.path.join(path, 'resnet18_children_onnx')
print(model_path)

timeslice_path = os.path.join(model_path, '0')

for run in os.listdir(timeslice_path):
    run_path = os.path.join(timeslice_path, run)
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
                        core = float(parts[0])
                        avg_time = float(parts[1])
                        avg_times.append(avg_time)
                        cores.append(core)
                        break

            if avg_times:
                max_time = max(avg_times)
                max_times.append(max_time)
                counter += 1

sum = 0
for i in max_times:
    sum += i                   # i is the value, not the index
final_avg = sum / counter
print(final_avg)
print(sum)
print(counter)
print(max_times)
print(cores)
print(len(max_times))


