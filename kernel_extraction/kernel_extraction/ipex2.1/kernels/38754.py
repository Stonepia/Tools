

# Original file: ./ElectraForCausalLM__0_forward_169.0/ElectraForCausalLM__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/il/cilnhrxvgt5ko3icahwn4reirxzwyb333h3kttsxqdvpjuk6tcxn.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
# cross_entropy => full_default_2, ne, neg, sum_15, where_1
triton_per_fused_nll_loss_forward_18 = async_compile.triton('triton_per_fused_nll_loss_forward_18', '''
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
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_nll_loss_forward_18(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp1 = tl.full([1, 1], 8176, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (1 + (512*(((r2 + (128*x0) + (8176*x1)) // 511) % 32)) + ((r2 + (128*x0)) % 511)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1, 1], -100, tl.int64)
    tmp5 = tmp3 != tmp4
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tl.where(tmp5, tmp3, tmp6)
    tmp8 = tl.where(tmp7 < 0, tmp7 + 30522, tmp7)
    # tl.device_assert(((0 <= tmp8) & (tmp8 < 30522)) | ~tmp2, "index out of bounds: 0 <= tmp8 < 30522")
    tmp9 = tl.load(in_ptr1 + (tmp8 + (30522*r2) + (3906816*x0) + (249547872*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp10 = -tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, 0)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')
