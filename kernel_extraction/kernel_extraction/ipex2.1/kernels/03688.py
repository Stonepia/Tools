

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/nr/cnrqte4c2hvxqeh7piqnvq6sbt7ohcqjz7tp23vhhqfnf2zibvh5.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.pow, aten.sum]

triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_8 = async_compile.triton('triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_8', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_mul_native_dropout_backward_pow_sum_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp2 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = -0.5
    tmp14 = tmp8 * tmp13
    tmp15 = tmp10 * tmp10
    tmp16 = tmp15 * tmp10
    tmp17 = tmp14 * tmp16
    tmp18 = 512.0
    tmp19 = tmp17 / tmp18
    tmp20 = 2.0
    tmp21 = tmp3 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp12 + tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 1.1111111111111112
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 * tmp27
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp23, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask)
''')
