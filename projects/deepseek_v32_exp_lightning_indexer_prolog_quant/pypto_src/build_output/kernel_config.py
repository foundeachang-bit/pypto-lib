"""
Lightning Indexer Prolog Quant — kernel and orchestration config for simpler run_example.

Use with simpler's run_example.py (tensormap_and_ringbuffer runtime):
  -k <path-to-this-dir's-parent>  (kernels_dir = build_output, so parent is pypto_src)
  -g <path-to-golden.py>

Kernels are built from PyPTO as .pto. For a2a3 you must have compiled .o alongside each .pto
(e.g. ptoas or your build outputs .o). For a2a3sim you need .so alongside each .pto.
"""

from pathlib import Path

_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_ROOT / "orchestration" / "LightningIndexerPrologQuant.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

# func_id must match orchestration: 0=matmul, 1=mul, 2=matmul_nn, 3=symmetric_quant, 4=layernorm, 5=softmax
KERNELS = [
    {"func_id": 0, "name": "incore_matmul", "source": str(_ROOT / "kernels" / "aic" / "incore_matmul.pto"), "core_type": "aic"},
    {"func_id": 1, "name": "incore_mul", "source": str(_ROOT / "kernels" / "aiv" / "incore_mul.pto"), "core_type": "aiv"},
    {"func_id": 2, "name": "incore_matmul_nn", "source": str(_ROOT / "kernels" / "aic" / "incore_matmul_nn.pto"), "core_type": "aic"},
    {"func_id": 3, "name": "incore_symmetric_quant", "source": str(_ROOT / "kernels" / "aiv" / "incore_symmetric_quant.pto"), "core_type": "aiv"},
    {"func_id": 4, "name": "incore_layernorm", "source": str(_ROOT / "kernels" / "aiv" / "incore_layernorm.pto"), "core_type": "aiv"},
    {"func_id": 5, "name": "incore_softmax", "source": str(_ROOT / "kernels" / "aiv" / "incore_softmax.pto"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
