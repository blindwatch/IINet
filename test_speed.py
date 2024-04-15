import options
import os
import random
import time
import numpy as np
import torch
import progressbar

from modules.disp_model import DispModel
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

height = 512
width = 960

if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    device = torch.device('cuda:0')

    left = torch.randn((1, 3, height, width), device=device)
    right = torch.randn((1, 3, height, width), device=device)

    inputs = {
        'left': left,
        'right': right,
    }
    # build model
    model = DispModel(opts).to(device)
    model.eval()

    #from thop import profile
    #Flops, params = profile(model, inputs=(inputs, False)) # macs
    #print('Flops: % .4fG'%(Flops / 1000000000))
    #print('params参数量: % .4fM'% (params / 1000000))
    
    #from pynvml import *
    #nvmlInit()
    #handle = nvmlDeviceGetHandleByIndex(0)
    #while True:
    #    out = model(inputs, False)
    #    info = nvmlDeviceGetMemoryInfo(handle)
    #    print("Memory Used: ", round(info.used / 1024**2))
    #    del out
    #    torch.cuda.empty_cache()
    #nvmlShutdown()

    #frozenlist = ['matching_model','cost_volume_net', 'depth_decoder']
    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)
    #    if name.split('.')[0] in ['cost_volume_net', 'depth_decoder']:
    #        param.requires_grad = False

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=10),
            with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
        for i in range(50):
            outputs = model(inputs, False)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_stacks("torch_cpu_stack.json", metric="self_cpu_time_total")
    prof.export_stacks("torch_cuda_stack.json", metric="self_cuda_time_total")


