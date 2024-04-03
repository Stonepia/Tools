

# Original file: ./hf_distil_whisper__41_inference_81.21/hf_distil_whisper__41_inference_81.21.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/xx/cxx4u72jyvfjowfpqkwukmnmmzojizvidh3phn6ozqzbqqqsfeop.py
# Source Nodes: [any_1, isinf], Original ATen: [aten.any, aten.isinf]
# any_1 => any_1
# isinf => isinf
triton_per_fused_any_isinf_12 = async_compile.triton('triton_per_fused_any_isinf_12', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_isinf_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_any_isinf_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12032
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = r2 + (128*x0)
    tmp1 = tl.full([1, 1], 8171, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r2 + (128*x0) + (8171*x1)
    tmp4 = tl.full([1, 1], 1536000, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((1500*((r2 + (128*x0) + (8171*x1)) % 1024)) + (((r2 + (128*x0) + (8171*x1)) // 1024) % 1500)), rmask & tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = libdevice.isinf(tmp7).to(tl.int1)
    tmp9 = tl.where(tmp6, tmp8, 0)
    tmp10 = tl.where(tmp2, tmp9, 0)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.any(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')
