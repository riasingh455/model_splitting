import onnxruntime as ort
import sys
# import onnxruntime.quantization #import QuantFormat, QuantType, quantize_dynamic
import numpy as np
import multiprocessing as mp
from ast import literal_eval
from datetime import datetime
import time
import zmq
from functools import reduce
import gc
#from pathlib import Path

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

def silly_test(core, path, rank, iters, shape_dict, x):
    # input_name = session.get_inputs()[0].name
    # print(shape_dict[rank])
    
    # sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # x=[np.empty(shape_dict[rank][0], dtype=np.float32)]
    #network stuff technically here somewhere?
    #shapeshift to match input layer of model slice
    ort_sess = ort.InferenceSession(f'{path}/exe_split_{rank}.pte_quant.onnx', sess_options=sess_opt, providers=providers)

    if len(shape_dict[rank])>2:
        x=shape_shifter(shape_dict[rank][2:], x[0])


    
    #fake_barrier(self_host, other_hosts, extra=core)
    # fake_barrier(f_ts)

    ts = datetime.now()
    print(f"{ts} Sync done -> model run start", flush=True)
    #print sync message -> do network sync here though
    total_start = time.perf_counter()
    # print([i.name for i in ort_sess.get_inputs()])
    input_map = {i.name: x[ind] for ind, i in enumerate(ort_sess.get_inputs())}
    outputs=[]
    comp_times=[]
    for _ in range(iters):
        per_ts=time.perf_counter()
        outputs.append(ort_sess.run(None, input_map))
        comp_times.append(time.perf_counter()-per_ts)

        # print(core, outputs[0])
    total_end = time.perf_counter()
    print(core, round(total_end-total_start,3), comp_times )
    return np.stack(outputs)
    #fake_barrier_rmv(self_host, core)
    
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

def fake_barrier(f_ts):
    while int(datetime.now().timestamp())<f_ts:
        pass

    #dname=f"/home/animesh/{self_host}_{extra}"
    #Path.mkdir(Path(dname), exist_ok=False)
    #check_list=[0]
    #while sum(check_list)!=len(other_hosts):
    #    check_list = [(1 if Path.is_dir(Path(f"/home/animesh/{h}_{extra}")) else 0) for h in other_hosts]
    #Path.rmdir(Path(dname))

def fake_barrier_rmv(self_host, extra=''):
    pass
    #dname=f"/home/animesh/{self_host}_{extra}"
    #Path.rmdir(Path(dname))

if __name__=="__main__":
    # procs=[]
    mp.set_start_method("fork")
    
    #iters equivalent to batch size -> 
    #so network needs to transfer full final data, with all iters data, not just one processes's iter data!
    #batch num requires batch num network calls, not just one network call! 
    iters,batch_num = int(sys.argv[1].split(",")[0]),int(sys.argv[1].split(",")[1])
    path = sys.argv[2]
    rank, world = int(sys.argv[3].split(",")[0]), int(sys.argv[3].split(",")[1])
    #equivalent to always having minimum batch size!
    cores= int(sys.argv[4])
    port = int(sys.argv[5])+rank
    # f_ts = int(sys.argv[5])
    #f_ts = datetime.now() + timedelta(seconds=t_delay)
    #f_ts = (f_ts.timestamp()*1000)
    other_hosts=[i for i in sys.argv[6].split(",")[:-1]] #convention, ordered by ranks
    self_host=other_hosts[rank]
    

    if len(path)==0:
        exit
    f=open(f"{path}/stages.dict")
    line=f.readlines()[0].strip()
    shape_dict = literal_eval(line)
    numpy_shape= shape_dict[rank]
    #broadcast sync with everyone else -> good to have for testing?
    #or force waiting during recv/sends? -> should be easier? 
    
    # if main node, start push here! 
    # if non-main node, start pull here!
    # do not multi-process zmq -> let it be in main process, but transfer more data based on iters/batch num/batch size/cores!
    
    context = zmq.Context()
    
    

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 1
    sess_opt.inter_op_num_threads=1
    sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ['CPUExecutionProvider']
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    #sync messages between all procs before starting
    if rank!=0:
        sync_sender = context.socket(zmq.PUSH)
        sync_recver = context.socket(zmq.PULL)

        sync_sender.bind(f"tcp://*:{port}")
        sync_recver.connect(f"tcp://{other_hosts[0]}:{port-rank}")

        sync_sender.send(b'1')
        if rank == world-1:
            sync_sender.close()

        raw = sync_recver.recv()
        sync_recver.close()
        del sync_recver
    else:
        sync_sender = context.socket(zmq.PUSH)
        sync_sender.bind(f"tcp://*:{port}")

        for ind in range(1,len(other_hosts[1:])+1):
            sync_recver = context.socket(zmq.PULL)
            sync_recver.connect(f"tcp://{other_hosts[ind]}:{port+ind}")
            raw = sync_recver.recv()
            sync_recver.close()
        
        for ind in range(1,len(other_hosts[1:])+1):
            sync_sender.send(b'1')
        # sync_sender.close()
        
        print("ALL SYNC")
    receiver = context.socket(zmq.PULL)
    if rank!=0:
        receiver.connect(f"tcp://{other_hosts[rank-1]}:{port-1}")
    else:
        receiver.close()
        del receiver
    if rank==world-1:
        sync_sender.close()
        del sync_sender 
    gc.collect()
    s_ts = datetime.now()
    net_times=[]
    for _ in range(batch_num):
        recv_st = time.perf_counter()
        np_array=[]
        recv_size = 0
        send_size = 0
        if rank!=0:
            # for _ in range(batch_num):
            raw_buffer = receiver.recv()
            np_arr = np.frombuffer(raw_buffer, dtype=str_to_dtype(numpy_shape[1]))
            recv_size = np_arr.nbytes
            #filter data by iters and or cores
            flattened_length = np.prod(numpy_shape[0])
            np_arr = np_arr[:flattened_length]
            np_array.append(np_arr.reshape(numpy_shape[0]))
        recv_et = time.perf_counter()
        
        
        if rank==0:
            np_array.append(np.empty(shape_dict[rank][0], dtype=np.float32))
        # data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        outputs=[]
        with mp.Pool(processes=cores) as p:
            for i in  p.starmap(silly_test, [(core, path, rank, iters, shape_dict, np_array,) for core in range(cores)]):
                outputs.append(i)
            del np_array
            # del np_arr

            np_output=np.stack(outputs)
        # for core in range(cores):
        #     p = mp.Process(target=silly_test, args=(ort_sess, core, path, rank, iters, shape_dict, [np_array], ))
        #     p.start()
        #     procs.append(p)

        # for p in procs:
        #     p.join()
        
        send_st = time.perf_counter()
        if rank!=world-1:
            send_size = np_output.nbytes
            sync_sender.send(np_output)
            del np_output
        send_et = time.perf_counter()

        net_times.append([round(recv_et-recv_st,3), round(send_et-send_st,3), recv_size, send_size ])
        gc.collect()
    e_ts = datetime.now()
    if rank!=world-1:
        sync_sender.close()
        del sync_sender
    if rank!=0:
        receiver.close()
        del receiver
    gc.collect()
    print(f"{e_ts} Sync done -> model run start", flush=True)
    print(f"net_times: {net_times}")
    #fake_barrier_rmv(self_host)

    print(f"echo 'race4fun' | sudo -S sh -c 'journalctl --since "+ f'"{s_ts}" --until "{e_ts}" -k -o short-iso '+"'| grep -i -E 'throttle|throttled|thermal|cpufreq|under-voltage|voltage'")

# Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')
