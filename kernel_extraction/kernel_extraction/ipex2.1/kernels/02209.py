

# Original file: ./M2M100ForConditionalGeneration__66_forward_205.16/M2M100ForConditionalGeneration__66_forward_205.16_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/mv/cmv6yvt5niklsmg6f6dmdimncowtf5qqa76tctyhayuxdhp575nu.py
# Source Nodes: [dropout, softmax], Original ATen: [aten._softmax, aten.native_dropout]
# dropout => gt, mul_3, mul_4
# softmax => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_native_dropout_3 = async_compile.triton('triton_per_fused__softmax_native_dropout_3', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_native_dropout_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_native_dropout_3(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr2 + load_seed_offset)
    tmp14 = r2 + (128*x3)
    tmp15 = tl.rand(tmp13, (tmp14).to(tl.uint32))
    tmp16 = 0.1
    tmp17 = tmp15 > tmp16
    tmp18 = tmp8 / tmp12
    tmp19 = tmp17.to(tl.float32)
    tmp20 = tmp19 * tmp18
    tmp21 = 1.1111111111111112
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp17, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp18, rmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp22, rmask)
''')
