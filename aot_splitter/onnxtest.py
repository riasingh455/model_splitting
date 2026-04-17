import onnxruntime as ort
# import onnxruntime.quantization #import QuantFormat, QuantType, quantize_dynamic
import numpy as np
import multiprocessing as mp

def silly_test(core):
    # input_name = session.get_inputs()[0].name
    ort_sess = ort.InferenceSession('test_quant.onnx')
    x=np.empty((1,3,224,224), dtype=np.float32)
    outputs = ort_sess.run(None, {'input':x })
    print(core, outputs[0])
# x, y = test_data[0][0], test_data[0][1]
# print(ort_sess.get_modelmeta())
if __name__=="__main__":
    procs=[]
    # data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    for core in range(4):
        p = mp.Process(target=silly_test, args=(core, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

# Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')
