

# Original file: ./rexnet_100___60.0/rexnet_100___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/vt/cvtivvfzqonx2ahssfv5fvnabh5x2q5dfbn672pqwon7zirvcbri.py
# Source Nodes: [mean_10], Original ATen: [aten.mean]
# mean_10 => mean_10
triton_per_fused_mean_62 = async_compile.triton('triton_per_fused_mean_62', '''
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
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_62', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_mean_62(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 115968
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 906
    x1 = (xindex // 906)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r2) + (44394*x1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 49.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''')