

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/yc/cycrvb3brp3ez2dj53zqnhemchi7ndkuzrpbfh3jz2rlylbdcojr.py
# Source Nodes: [exp, logsumexp, sub], Original ATen: [aten.exp, aten.logsumexp, aten.sub]
# exp => exp_1
# logsumexp => abs_1, add_2, amax, eq, exp, full_default_1, log, sub_1, sum_1, where
# sub => sub_2
triton_per_fused_exp_logsumexp_sub_7 = async_compile.triton('triton_per_fused_exp_logsumexp_sub_7', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_exp_logsumexp_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_exp_logsumexp_sub_7(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tl.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp0 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.log(tmp15)
    tmp17 = tmp16 + tmp9
    tmp18 = tmp0 - tmp17
    tmp19 = tl.exp(tmp18)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp19, rmask)
''')
