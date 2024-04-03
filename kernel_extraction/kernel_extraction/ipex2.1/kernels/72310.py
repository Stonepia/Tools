

# Original file: ./DebertaForQuestionAnswering__0_backward_135.1/DebertaForQuestionAnswering__0_backward_135.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/2e/c2ewkoe2lahjvmay23lrp6bqrwrir6xqqj6qslah4yj2as7oqqus.py
# Source Nodes: [float_24, sub_46, trampoline_autograd_apply, truediv_35], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# float_24 => convert_element_type_201
# sub_46 => sub_94
# trampoline_autograd_apply => full_default_1
# truediv_35 => div_47
triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_13 = async_compile.triton('triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_13', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = -tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 / tmp11
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp5 / tmp11
    tmp20 = -tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 2.0
    tmp26 = tmp11 * tmp25
    tmp27 = tmp18 / tmp26
    tmp28 = 768.0
    tmp29 = tmp27 / tmp28
    tmp30 = tmp10 * tmp25
    tmp31 = tmp29 * tmp30
    tmp32 = -tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp19 + tmp31
    tmp38 = tmp24 + tmp36
    tmp39 = tmp38 / tmp28
    tmp40 = tmp37 + tmp39
    tmp42 = tmp40.to(tl.float32)
    tmp43 = 0.0
    tmp44 = tl.where(tmp41, tmp43, tmp42)
    tmp45 = 1.1111111111111112
    tmp46 = tmp44 * tmp45
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp40, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp46, rmask)
''')
