import os
import psutil
import gc
import logging
import time

logger = logging.getLogger(__name__)

def get_memory_usage_mb() -> float:
    """Returns the current process resident set size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def log_memory(label: str) -> None:
    """Prints memory usage with a label."""
    mem_mb = get_memory_usage_mb()
    msg = f"[{label}] Memory Usage: {mem_mb:.2f} MB"
    print(msg)
    logger.info(msg)

def gc_collect() -> None:
    """Forces garbage collection and logs freed memory."""
    mem_before = get_memory_usage_mb()
    gc.collect()
    mem_after = get_memory_usage_mb()
    msg = f"[GC] Freed {mem_before - mem_after:.2f} MB. Current: {mem_after:.2f} MB"
    print(msg)
    logger.info(msg)

class MemoryTracker:
    """Context manager that logs peak memory during a block."""
    def __init__(self, name: str):
        self.name = name
        self.start_mem = 0.0
        self.start_time = 0.0
        
    def __enter__(self):
        self.start_mem = get_memory_usage_mb()
        self.start_time = time.time()
        log_memory(f"Start: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_mem = get_memory_usage_mb()
        end_time = time.time()
        elapsed = end_time - self.start_time
        msg = f"[End: {self.name}] Time: {elapsed:.2f}s | Delta Mem: {end_mem - self.start_mem:.2f} MB | Final Mem: {end_mem:.2f} MB"
        print(msg)
        logger.info(msg)
