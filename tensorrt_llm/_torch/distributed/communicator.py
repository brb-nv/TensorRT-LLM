import os
from abc import ABC, abstractmethod
from typing import Optional

import copy
import numpy as np
import torch
import torch.distributed as dist

from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_broadcast,
                                 mpi_comm, mpi_isend, mpi_isend_object,
                                 mpi_recv, mpi_recv_object, mpi_send,
                                 mpi_send_object)
from tensorrt_llm.mapping import Mapping


class Distributed(ABC):

    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size

    @property
    def has_tp(self):
        return self.mapping.has_tp()

    @property
    def has_pp(self):
        return self.mapping.has_pp()

    @property
    def cp_size(self):
        return self.mapping.cp_size

    @property
    def pp_size(self):
        return self.mapping.pp_size

    @property
    def tp_size(self):
        return self.mapping.tp_size

    @property
    def cp_rank(self):
        return self.mapping.cp_rank

    @property
    def tp_rank(self):
        return self.mapping.tp_rank

    @property
    def pp_rank(self):
        return self.mapping.pp_rank

    @property
    def is_last_pp_rank(self):
        return self.mapping.is_last_pp_rank()

    @property
    def is_second_last_pp_rank(self):
        return self.mapping.is_second_last_pp_rank()

    @property
    def is_first_pp_rank(self):
        return self.mapping.is_first_pp_rank()

    @property
    def next_pp_rank(self):
        return self.mapping.next_pp_rank()

    @property
    def prev_pp_rank(self):
        return self.mapping.prev_pp_rank()

    @property
    def has_cp_ulysses(self):
        return self.mapping.has_cp_ulysses()

    @property
    def cp_config(self):
        return self.mapping.cp_config

    @abstractmethod
    def broadcast(self, obj, root=0):
        pass

    @abstractmethod
    def allgather(self, obj, root=0):
        pass


class MPIDist(Distributed):

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self.create_cp_comm()
        # Repurpose CP ranks to TP for Helix so that the right comms are created.
        mapping_with_helix = None
        if self.mapping.cp_size > 1:
            print(f"[MPIDist::__init__] Repurposing CP ranks to TP for Helix.")
            # TODO: More principled thing to do would be to update mapping to account for
            # repurposing of CP ranks to TP.
            mapping_with_helix = copy.deepcopy(self.mapping)
            mapping_without_helix = Mapping(
                world_size=self.mapping.world_size,
                rank=self.mapping.rank,
                gpus_per_node=self.mapping.gpus_per_node,
                cp_size=1,
                cp_config={},
                tp_size=self.mapping.tp_size * self.mapping.cp_size,
                pp_size=self.mapping.pp_size,
                moe_ep_size=self.mapping.moe_ep_size,
                auto_parallel=False,
                enable_attention_dp=self.mapping.enable_attention_dp)
            self.mapping = mapping_without_helix
        self.create_tp_comm()
        self.create_pp_comm()

        # Restore the original mapping.
        if mapping_with_helix is not None:
            print(f"[MPIDist::__init__] Restoring original mapping.")
            self.mapping = mapping_with_helix

    def broadcast(self, obj, root=0):
        return mpi_broadcast(obj, root)

    def allgather(self, obj):
        return mpi_allgather(obj)

    def barrier(self):
        mpi_barrier()

    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        return mpi_isend(buf, dest, tag)

    def send(self, buf: np.ndarray, dest, tag=0):
        # blocking send numpy buffer
        mpi_send(buf, dest, tag)

    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        return mpi_recv(buf, src, tag)

    def send_object(self, obj, dest, tag=0):
        mpi_send_object(obj, dest, tag)

    def isend_object(self, obj, dest, tag=0):
        return mpi_isend_object(obj, dest, tag)

    def recv_object(self, src, tag=0):
        return mpi_recv_object(src, tag)

    def create_tp_comm(self):
        print(f"[MPIDist::create_tp_comm] rank: {self.mapping.rank}, tp_rank: {self.mapping.tp_rank}, tp_group: {self.mapping.tp_group}")
        new_group = mpi_comm().group.Incl(self.mapping.tp_group)
        self.tp_comm = mpi_comm().Create_group(new_group)

    def create_pp_comm(self):
        print(f"[MPIDist::create_pp_comm] rank: {self.mapping.rank}, pp_rank: {self.mapping.pp_rank}, pp_group: {self.mapping.pp_group}")
        new_group = mpi_comm().group.Incl(self.mapping.pp_group)
        self.pp_comm = mpi_comm().Create_group(new_group)

    def tp_allgather(self, obj):
        return self.tp_comm.allgather(obj)

    def tp_gather(self, obj):
        return self.tp_comm.gather(obj)

    def tp_broadcast(self, obj, root=0):
        return self.tp_comm.bcast(obj, root)

    def pp_allgather(self, obj):
        return self.pp_comm.allgather(obj)

    def pp_gather(self, obj):
        return self.pp_comm.gather(obj)

    def pp_broadcast(self, obj, root=0):
        return self.pp_comm.bcast(obj, root)

    def create_cp_comm(self):
        print(f"[MPIDist::create_cp_comm] rank: {self.mapping.rank}, cp_rank: {self.mapping.cp_rank}, cp_group: {self.mapping.cp_group}")
        new_group = mpi_comm().group.Incl(self.mapping.cp_group)
        self.cp_comm = mpi_comm().Create_group(new_group)

    def cp_allgather(self, obj):
        return self.cp_comm.allgather(obj)


class TorchDist(Distributed):

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        if not dist.is_initialized():
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            # TODO: fix the constant default port
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=mapping.world_size,
                                    rank=mapping.rank)
        self.device_tp_group = dist.new_group(mapping.tp_group, backend="nccl")
        self.cpu_tp_group = dist.new_group(mapping.tp_group, backend="gloo")
        self.device_cp_group = dist.new_group(mapping.cp_group, backend="nccl")
        self.cpu_cp_group = dist.new_group(mapping.cp_group, backend="gloo")

    def broadcast_tp(self, obj, root=0):
        if root not in self.mapping.tp_group:
            return obj
        elif self.rank == root:
            torch.distributed.broadcast_object_list([obj],
                                                    src=root,
                                                    group=self.cpu_tp_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=root,
                                                    group=self.cpu_tp_group)
            return recv[0]

    def broadcast_cp(self, obj, root=0):
        if root not in self.mapping.cp_group:
            return obj
        elif self.rank == root:
            torch.distributed.broadcast_object_list([obj],
                                                    src=root,
                                                    group=self.cpu_cp_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=root,
                                                    group=self.cpu_cp_group)
            return recv[0]

    def broadcast(self, obj, root=0):
        assert not (self.mapping.has_cp_ulysses() and self.mapping.has_tp()
                    ), 'unsupported mix cp ulysses and tp now'
        if self.mapping.has_cp_ulysses():
            self.broadcast_cp(obj, root)
        elif self.mapping.has_tp():
            self.broadcast_tp(obj, root)
        else:
            pass


class PPComm:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
            self.mapping.world_size,
            self.mapping.rank,
        )

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()
        self.nccl_comm.send(tensor, dest)

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()
        self.nccl_comm.recv(tensor, src)


_pp_comm = None


def init_pp_comm(mapping):
    """Initialize PPComm once at startup"""
    global _pp_comm
    _pp_comm = PPComm(mapping)


def pp_recv(tensor):
    """Receive tensors from previous pp rank."""
    _pp_comm.recv(tensor)


def pp_send(tensor):
    """Send tensors to next pp rank."""
    _pp_comm.send(tensor)
