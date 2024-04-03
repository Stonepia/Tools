

# Original file: ./DebertaForQuestionAnswering__0_forward_133.0/DebertaForQuestionAnswering__0_forward_133.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/sb/csbwyg2xxvpquqswsacjfcmgbfeabl3kgippehp3ubvqo7fnr4wi.py
# Source Nodes: [add_6, add_7, add_8, add_9, l__self___deberta_encoder_layer_1_attention_self_in_proj, mean_4, mean_5, mul_4, mul_5, pow_3, sqrt_3, sub_4, trampoline_autograd_apply_1, trampoline_autograd_apply_4, truediv_2, truediv_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
# add_6 => add_7
# add_7 => add_9
# add_8 => add_10
# add_9 => add_11
# l__self___deberta_encoder_layer_1_attention_self_in_proj => convert_element_type_25, view_18
# mean_4 => mean_4
# mean_5 => mean_5
# mul_4 => mul_7
# mul_5 => mul_12
# pow_3 => pow_3
# sqrt_3 => sqrt_3
# sub_4 => sub_9
# trampoline_autograd_apply_1 => full_default_5
# trampoline_autograd_apply_4 => convert_element_type_23, convert_element_type_24, lt_3, mul_11, sub_8, where_5
# truediv_2 => div_3
# truediv_3 => div_4
triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14 = async_compile.triton('triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
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
    tmp9 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.9
    tmp4 = tmp2 < tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.0
    tmp7 = tmp6 - tmp5
    tmp8 = (tmp7 != 0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp18 = tmp16 / tmp17
    tmp19 = tmp15 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = 768.0
    tmp28 = tmp26 / tmp27
    tmp29 = tmp22 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp34 / tmp27
    tmp36 = 1e-07
    tmp37 = tmp35 + tmp36
    tmp38 = tl.sqrt(tmp37)
    tmp40 = tmp29 / tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp29, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp38, None)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp44, rmask)
''')
