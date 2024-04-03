

# Original file: ./DebertaForQuestionAnswering__0_backward_135.1/DebertaForQuestionAnswering__0_backward_135.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/7y/c7yipvt3lvxoqeun3l3b37npvumbluvfvowcnyeel6tangdsbbp4.py
# Source Nodes: [float_23, sub_44, trampoline_autograd_apply, truediv_33], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# float_23 => convert_element_type_190
# sub_44 => sub_89
# trampoline_autograd_apply => full_default_1
# truediv_33 => div_44
triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_24 = async_compile.triton('triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_24', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = -tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 / tmp12
    tmp14 = tmp13 / tmp12
    tmp15 = tmp7 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp6 / tmp12
    tmp21 = -tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = 2.0
    tmp27 = tmp12 * tmp26
    tmp28 = tmp19 / tmp27
    tmp29 = 768.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp11 * tmp26
    tmp32 = tmp30 * tmp31
    tmp33 = -tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp20 + tmp32
    tmp39 = tmp25 + tmp37
    tmp40 = tmp39 / tmp29
    tmp41 = tmp38 + tmp40
    tmp43 = tmp41.to(tl.float32)
    tmp44 = 0.0
    tmp45 = tl.where(tmp42, tmp44, tmp43)
    tmp46 = 1.1111111111111112
    tmp47 = tmp45 * tmp46
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp41, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp47, rmask)
''')
