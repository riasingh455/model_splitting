import onnxruntime as ort
import sys
# import onnxruntime.quantization #import QuantFormat, QuantType, quantize_dynamic
import numpy as np
import multiprocessing as mp
from ast import literal_eval
from datetime import datetime
import time

def shape_shifter(numpy_shape, numpy_arr):
    slices=[] 
    split_t=[]
    #for each tuple shape get full tuple slice
    for k in numpy_shape[1]:
        c = 1
        for l in k:
            c*=l
        slices.append(c)
    counter=0 #to include the first slice
    for sl in range(len(slices)):
        split_t.append(numpy_arr[counter:counter+slices[sl]].reshape(tuple(numpy_shape[1][sl])))
        counter+=slices[sl]
    # final_arr = tuple(split_t)
    return split_t

def silly_test(core, path, rank, iters, shape_dict):
    # input_name = session.get_inputs()[0].name
    # print(shape_dict[rank])
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 1
    sess_opt.inter_op_num_threads=1
    sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ['CPUExecutionProvider']
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    x=np.empty(shape_dict[rank][0], dtype=np.float32)
    #network stuff technically here somewhere?
    #shapeshift to match input layer of model slice
    if len(shape_dict[rank])>2:
        x=shape_shifter(shape_dict[rank][2:], x)

    ort_sess = ort.InferenceSession(f'{path}/exe_split_{rank}.pte_quant.onnx', sess_options=sess_opt, providers=providers)
    input_name = ort_sess.get_inputs()[0].name
    ts = datetime.now()
    print(f"{ts} Sync done -> model run start", flush=True)
    #print sync message -> do network sync here though
    total_start = time.perf_counter()
    # print([i.name for i in ort_sess.get_inputs()])
    input_map = {i.name: x[ind] for ind, i in enumerate(ort_sess.get_inputs())}
    for _ in range(iters):
        outputs = ort_sess.run(None, input_map)

        # print(core, outputs[0])
    total_end = time.perf_counter()
    print(core, round(total_end-total_start,3) )
    
# x, y = test_data[0][0], test_data[0][1]
# print(ort_sess.get_modelmeta())
def str_to_dtype(code):
    mapping = {
        "torch.float32": np.float32,
        "torch.float16": np.float16,
        # "torch.bfloat16": np.bfloat16,
        "torch.int64": np.int64,
        "torch.int32": np.int32,
        }
    if code not in mapping:
        raise ValueError(f"Unsupported dtype code: {code}")
    return mapping[code]

if __name__=="__main__":
    procs=[]
    iters = int(sys.argv[1])
    path = sys.argv[2]
    rank = int(sys.argv[3])
    cores= int(sys.argv[4])
    if len(path)==0:
        exit
    f=open(f"{path}/stages.dict")
    line=f.readlines()[0].strip()
    shape_dict = literal_eval(line)
    numpy_shape= shape_dict[rank]
    #network, sync technically goes here? but skip for now

    s_ts = datetime.now()
    # data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    for core in range(cores):
        p = mp.Process(target=silly_test, args=(core, path, rank, iters, shape_dict, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    
    e_ts = datetime.now()
    print(f"{e_ts} Sync done -> model run start", flush=True)
    print(f"echo 'race4fun' | sudo -S sh -c 'journalctl --since "+ f'"{s_ts}" --until "{e_ts}" -k -o short-iso '+"'| grep -i -E 'throttle|throttled|thermal|cpufreq|under-voltage|voltage'")

# Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')
