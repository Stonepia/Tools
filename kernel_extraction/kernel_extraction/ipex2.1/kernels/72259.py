

# Original file: ./DebertaForQuestionAnswering__0_backward_135.1/DebertaForQuestionAnswering__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/wf/cwfvhnjjenzwyl2rrmz35mhrs7mvt6fjtgfu3wqescc3zxpopkjv.py
# Source Nodes: [float_1, iadd, sub, trampoline_autograd_apply, truediv], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# float_1 => convert_element_type
# iadd => add
# sub => sub
# trampoline_autograd_apply => full_default_1
# truediv => div
triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_27 = async_compile.triton('triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_27', '''
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
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp0, tmp5, tmp4)
    tmp7 = 1.1111111111111112
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = -tmp11
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20 / tmp19
    tmp22 = tmp12 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp11 / tmp19
    tmp28 = 2.0
    tmp29 = tmp19 * tmp28
    tmp30 = tmp26 / tmp29
    tmp31 = 768.0
    tmp32 = tmp30 / tmp31
    tmp33 = tmp18 * tmp28
    tmp34 = tmp32 * tmp33
    tmp35 = -tmp27
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = -tmp34
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp46 = tl.where(tmp45 < 0, tmp45 + 50265, tmp45)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tmp45 == tmp47
    tmp49 = tmp27 + tmp34
    tmp50 = tmp39 + tmp44
    tmp51 = tmp50 / tmp31
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.where(tmp48, tmp5, tmp54)
    tl.store(out_ptr1 + (r2 + (768*x3)), tmp27, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp34, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r2 + (768*tmp46), [RBLOCK])), tmp55, rmask)
    tl.store(out_ptr3 + (x3), tmp39, None)
    tl.store(out_ptr4 + (x3), tmp44, None)
''')
