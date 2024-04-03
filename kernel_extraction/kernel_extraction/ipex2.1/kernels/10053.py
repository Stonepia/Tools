

# Original file: ./GPT2ForSequenceClassification__0_forward_133.0/GPT2ForSequenceClassification__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/m6/cm6mltesdbopsms46uztt7jpv2g3e3garzio3yg42dhcaoypx6uw.py
# Source Nodes: [full, full_1, l__mod___transformer_h_0_attn_attn_dropout, softmax, truediv, where], Original ATen: [aten._softmax, aten.div, aten.full, aten.native_dropout, aten.where]
# full => full_default
# full_1 => full_default_1
# l__mod___transformer_h_0_attn_attn_dropout => gt_1, mul_4, mul_5
# softmax => amax, div_1, exp, sub_1, sum_1
# truediv => div
# where => where
triton_per_fused__softmax_div_full_native_dropout_where_4 = async_compile.triton('triton_per_fused__softmax_div_full_native_dropout_where_4', '''
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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_full_native_dropout_where_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_div_full_native_dropout_where_4(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask, other=0.0)
    tmp2 = 8.0
    tmp3 = tmp1 / tmp2
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp0, tmp3, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp8, 0))
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.load(in_ptr2 + load_seed_offset)
    tmp17 = r2 + (1024*x3)
    tmp18 = tl.rand(tmp16, (tmp17).to(tl.uint32))
    tmp19 = 0.1
    tmp20 = tmp18 > tmp19
    tmp21 = tmp11 / tmp15
    tmp22 = tmp20.to(tl.float32)
    tmp23 = tmp22 * tmp21
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp20, rmask)
    tl.store(out_ptr4 + (r2 + (1024*x3)), tmp21, rmask)
    tl.store(out_ptr5 + (r2 + (1024*x3)), tmp25, rmask)
''')
