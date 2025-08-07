import os
import subprocess

import torch
import torch.distributed as dist


# 用于设置 PyTorch 分布式训练环境
def setup_distributed(backend="gloo", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count() # 计算可用的 GPU 数量

    if "SLURM_JOB_ID" in os.environ:   # 如果在 SLURM 管理的集群上运行
        rank = int(os.environ["SLURM_PROCID"])  # SLURM_PROCID 环境变量表示当前进程的 ID
        world_size = int(os.environ["SLURM_NTASKS"]) # SLURM_NTASKS 环境变量表示总的进程数
        node_list = os.environ["SLURM_NODELIST"] # SLURM_NODELIST 环境变量包含分配给作业的节点列表
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1") # 使用 scontrol 命令获取主节点的主机名
        # specify master port
        if port is not None:  # 指定主节点的端口
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:  # 设置主节点的地址
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)   # 设置总的进程数
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)  # 设置本地进程在 GPU 上的 rank
        os.environ["RANK"] = str(rank)  # 设置当前进程的 rank
    else:
        rank = int(os.environ["RANK"])  # 从环境变量获取当前进程的 rank
        world_size = int(os.environ["WORLD_SIZE"])  # 从环境变量获取总的进程数

    torch.cuda.set_device(rank % num_gpus)  # 根据 rank 设置当前进程使用的 GPU

    dist.init_process_group(  # 初始化分布式进程组
        backend=backend,  # 指定分布式通信后端
        world_size=world_size,  # 总的进程数
        rank=rank, # 当前进程的 rank
    )
    return rank, world_size
