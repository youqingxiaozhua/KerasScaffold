import socket
import os

gpu_list = {
    'liye-SYS-7048GR-TR-BO004': {
        'k80': '0, 1',
        'titan v': '2'
    }
}


def set_visible_gpu(name):
    print('set gpu to', name)
    if name is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        return
    hostname = socket.gethostname()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[hostname][name]
