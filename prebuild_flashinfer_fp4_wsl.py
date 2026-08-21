import os
import threading
from pathlib import Path

import psutil

SAMPLE_INTERVAL_SECONDS = 1.0


def read_psi_totals() -> dict[str, int]:
    values = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        fields = line.split()
        values[fields[0]] = int(fields[-1].split("=", 1)[1])
    return values


def main() -> None:
    for variable in (
        "FLASHINFER_JIT_DEBUG",
        "FLASHINFER_JIT_LINEINFO",
        "FLASHINFER_JIT_VERBOSE",
    ):
        os.environ.pop(variable, None)
    os.environ.setdefault("MAX_JOBS", "1")

    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4

    stop_event = threading.Event()
    available_samples: list[int] = []
    swap_samples: list[int] = []

    def monitor() -> None:
        while not stop_event.wait(SAMPLE_INTERVAL_SECONDS):
            available_samples.append(psutil.virtual_memory().available)
            swap_samples.append(psutil.swap_memory().used)

    available_samples.append(psutil.virtual_memory().available)
    swap_samples.append(psutil.swap_memory().used)
    psi_start = read_psi_totals()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    spec = gen_gemm_sm120_module_cutlass_fp4()
    print(
        f"building={spec.name} mode=release jobs={os.environ['MAX_JOBS']} "
        f"sources={len(spec.sources)} "
        f"output={spec.jit_library_path}",
        flush=True,
    )
    try:
        spec.build(verbose=True, need_lock=True)
    finally:
        stop_event.set()
        monitor_thread.join()
        available_samples.append(psutil.virtual_memory().available)
        swap_samples.append(psutil.swap_memory().used)

    psi_end = read_psi_totals()
    module = spec.load()
    print(f"compiled={spec.is_compiled} module_type={type(module).__name__}")
    print(f"library={spec.jit_library_path}")
    print(f"library_bytes={spec.jit_library_path.stat().st_size}")
    print(
        "resources "
        f"min_available_gib={min(available_samples) / 1024**3:.3f} "
        f"max_swap_used_gib={max(swap_samples) / 1024**3:.3f} "
        f"psi_some_seconds={(psi_end['some'] - psi_start['some']) / 1_000_000:.3f} "
        f"psi_full_seconds={(psi_end['full'] - psi_start['full']) / 1_000_000:.3f}"
    )


if __name__ == "__main__":
    main()