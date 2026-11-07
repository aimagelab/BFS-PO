"""Local pipes and authenticated TCP connections for TRL's vLLM DP workers."""

import os
import time
from contextlib import contextmanager
from multiprocessing import get_context
from multiprocessing.connection import Client, Listener, wait


def local_worker_count(args):
    local_size = args.data_parallel_size_local
    if local_size is None:
        local_size = args.data_parallel_size
    if local_size <= 0 or args.data_parallel_size <= 0 or args.data_parallel_size % local_size:
        raise ValueError("data_parallel_size must be a positive multiple of data_parallel_size_local")
    num_nodes = args.data_parallel_size // local_size
    if not 0 <= args.data_parallel_node_rank < num_nodes:
        raise ValueError("data_parallel_node_rank must identify one of the inference nodes")
    if num_nodes > 1:
        if not args.data_parallel_master_host or not args.data_parallel_master_port or not args.worker_port:
            raise ValueError(
                "Multi-node DP requires data_parallel_master_host, data_parallel_master_port and worker_port"
            )
        if not os.environ.get("TRL_VLLM_WORKER_AUTHKEY"):
            raise ValueError("Multi-node DP requires a shared TRL_VLLM_WORKER_AUTHKEY environment variable")
    return local_size


def weight_sync_host(args, requested_host):
    # The client sends 0.0.0.0, which only works when all server workers are local.
    if args.data_parallel_master_host and requested_host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return args.data_parallel_master_host
    return requested_host


def _remote_worker(args, rank, master_port, worker):
    deadline = time.monotonic() + args.worker_connection_timeout
    address = (args.data_parallel_master_host, args.worker_port)
    authkey = os.environ["TRL_VLLM_WORKER_AUTHKEY"].encode()
    while True:
        try:
            connection = Client(address, authkey=authkey)
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out connecting vLLM worker {rank} to {address}") from None
            time.sleep(1)
    try:
        connection.send(rank)
        worker(args, rank, master_port, connection)
    finally:
        connection.close()


@contextmanager
def worker_connections(args, master_port, worker):
    """Yield rank-ordered connections on node 0; other nodes supervise their workers."""
    local_size = local_worker_count(args)
    first_rank = args.data_parallel_node_rank * local_size
    context = get_context("spawn")
    processes = []
    connections = {}
    listener = None
    try:
        if first_rank == 0 and local_size < args.data_parallel_size:
            listener = Listener(
                (args.data_parallel_master_host, args.worker_port),
                backlog=args.data_parallel_size,
                authkey=os.environ["TRL_VLLM_WORKER_AUTHKEY"].encode(),
            )
        for rank in range(first_rank, first_rank + local_size):
            if first_rank == 0:
                parent_connection, child_connection = context.Pipe()
                connections[rank] = parent_connection
                process = context.Process(target=worker, args=(args, rank, master_port, child_connection))
            else:
                process = context.Process(target=_remote_worker, args=(args, rank, master_port, worker))
            process.start()
            processes.append(process)
            if first_rank == 0:
                child_connection.close()

        if first_rank != 0:
            # Fail promptly if any local worker fails, so Slurm can terminate the whole server step.
            pending = {process.sentinel: process for process in processes}
            while pending:
                for sentinel in wait(list(pending)):
                    process = pending.pop(sentinel)
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"vLLM worker exited with status {process.exitcode}")
            yield None
            return

        while len(connections) < args.data_parallel_size:
            connection = listener.accept()
            rank = connection.recv()
            if not isinstance(rank, int) or not local_size <= rank < args.data_parallel_size or rank in connections:
                connection.close()
                raise ValueError(f"Unexpected or duplicate remote vLLM worker rank: {rank!r}")
            connections[rank] = connection
        if listener is not None:
            listener.close()
            listener = None
        yield [connections[rank] for rank in range(args.data_parallel_size)]
    finally:
        if listener is not None:
            listener.close()
        for connection in connections.values():
            try:
                connection.send({"type": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
