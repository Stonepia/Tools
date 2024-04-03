

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/gq/cgqnuomagfzjy7ja3c5effobepevvpieqjpcit2yominmsa6vuzk.py
# Source Nodes: [add_4, add_6, l__self___encoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_4 => add_6
# add_6 => add_8
# l__self___encoder_dropout => mul_1, mul_2
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_42 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_42', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: '*fp32', 10: '*i1', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp12 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = 1.1111111111111112
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp29 = tmp9 * tmp28
    tmp30 = tmp27 + tmp29
    tmp31 = -0.5
    tmp32 = tmp26 * tmp31
    tmp33 = tmp28 * tmp28
    tmp34 = tmp33 * tmp28
    tmp35 = tmp32 * tmp34
    tmp36 = 512.0
    tmp37 = tmp35 / tmp36
    tmp38 = 2.0
    tmp39 = tmp21 * tmp38
    tmp40 = tmp37 * tmp39
    tmp41 = tmp30 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp14
    tmp46 = tmp42 * tmp45
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp41, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp46, rmask)
''')
