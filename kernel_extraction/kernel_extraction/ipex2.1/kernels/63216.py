

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/3w/c3wotaxy2rm2souwahmyubhnfjebl4bv5fohtvquuosuu6pfr57u.py
# Source Nodes: [add_4, l__self___encoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_4 => add_6
# l__self___encoder_dropout => mul_1, mul_2
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_48 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_48', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*i1', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_48', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp6 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp6 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = -0.5
    tmp26 = tmp20 * tmp25
    tmp27 = tmp22 * tmp22
    tmp28 = tmp27 * tmp22
    tmp29 = tmp26 * tmp28
    tmp30 = 512.0
    tmp31 = tmp29 / tmp30
    tmp32 = 2.0
    tmp33 = tmp15 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp24 + tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp11
    tmp40 = tmp36 * tmp39
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp35, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp40, rmask)
''')
