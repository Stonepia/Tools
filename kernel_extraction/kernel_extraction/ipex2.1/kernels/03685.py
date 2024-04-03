

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/fh/cfhibob6z2ffw67fblzhymphghwr5chnp6kbma7bpqn5od5m7m7t.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_5 = async_compile.triton('triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_5', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask)
    tmp1 = 0.04419417382415922
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp9 * tmp16
    tmp18 = -0.5
    tmp19 = tmp15 * tmp18
    tmp20 = tmp16 * tmp16
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 * tmp21
    tmp23 = 512.0
    tmp24 = tmp22 / tmp23
    tmp25 = 2.0
    tmp26 = tmp10 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp17 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp5
    tmp32 = tmp28 * tmp31
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask)
''')
