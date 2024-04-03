

# Original file: ./hf_Whisper__23_inference_63.3/hf_Whisper__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/bw/cbwdevkz3fxu5a7niadsvxr2gykm4mtyxf7u6ejom4x4xgbpsjc7.py
# Source Nodes: [any_1, isinf], Original ATen: [aten.any, aten.isinf]
# any_1 => any_1
# isinf => isinf
triton_per_fused_any_isinf_8 = async_compile.triton('triton_per_fused_any_isinf_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_isinf_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_any_isinf_8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 36000
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 160
    x1 = (xindex // 160)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((1500*((r2 + (128*x0) + (20480*x1)) % 384)) + (576000*((r2 + (128*x0) + (20480*x1)) // 576000)) + (((r2 + (128*x0) + (20480*x1)) // 384) % 1500)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = libdevice.isinf(tmp0).to(tl.int1)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = triton_helpers.any(tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')
