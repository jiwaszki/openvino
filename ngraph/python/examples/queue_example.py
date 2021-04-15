"""
Third example.

User wants to run the inference on set of pictures and store the
results of the inference (e.g. in a database)
The Inference Queue allows him to run inference as parallel jobs.
"""

import numpy as np
import time

from openvino.inference_engine import IECore
# from openvino.inference_engine import TensorDesc
# from openvino.inference_engine import Blob
from openvino.inference_engine import StatusCode
from openvino.inference_engine import InferQueue

from helpers import get_images


def get_reference(executable_network, images):
    """Get reference outputs using synchronous API."""
    return [executable_network.infer({'data': img}) for img in images]


# Read images from a folder
images = get_images()

# Read and Load of network
ie = IECore()
ie_network = ie.read_network(
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.xml',
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.bin')
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})


# Create InferQueue with specific number of jobs/InferRequests
infer_queue = InferQueue(network=executable_network, jobs=6)

executable_network.infer()

ref_start_time = time.time()
for img in images:
    res = executable_network.infer({'data': img})
ref_end_time = time.time()
ref_time = (ref_end_time - ref_start_time) * 1000

results = [0] * len(images)
times = np.zeros((len(images)))


def get_results(request, userdata):
    """User-defined callback function."""
    end_time = time.time()
    print('Finished picture', userdata)
    global results
    global times
    results[userdata['index']] = request.output_blobs['fc_out'].buffer.copy()
    times[userdata['index']] = (end_time - userdata['start_time']) * 1000


# Set callbacks on each job/InferRequest
infer_queue.set_infer_callback(get_results)

print('Starting InferQueue...')
start_queue_time = time.time()
for i in range(len(images)):
    # If advanced user would like to have control:
    # tensor_desc = TensorDesc('FP32',
    #                          [1, 3, images[i].shape[2], images[i].shape[3]],
    #                          'NCHW')
    # img_blob = Blob(tensor_desc, images[i])
    start_request_time = time.time()
    infer_queue.async_infer(inputs={'data': images[i]},
                            userdata={'index': i,
                                      'start_time': start_request_time})
    print('Started picture ', i)

# Wait for all jobs/InferRequests to finish!
statuses = infer_queue.wait_all()
end_queue_time = time.time()
queue_time = (end_queue_time - start_queue_time) * 1000

if np.all(np.array(statuses) == StatusCode.OK):
    print('Reference execution time:', ref_time)
    print('Finished InferQueue! Execution time:', queue_time)
    print('Times for each image: ', times)
    reference = get_reference(executable_network, images)
    for i in range(len(results)):
        for key in executable_network.output_info:
            # print(results[i])
            # print(reference[i][key].buffer)
            assert np.allclose(results[i],
                               reference[i][key].buffer)
else:
    raise RuntimeError('InferQueue failed to finish!')
