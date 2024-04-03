

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/4i/c4i2qppvvf72rfk5egymkcgzsb45qc2q4nigh7w2zaxcuxve4fxp.py
# Source Nodes: [add_1, l__self___deberta_encoder_layer_0_attention_output_layer_norm, l__self___deberta_encoder_layer_0_output_layer_norm, l__self___deberta_encoder_layer_1_attention_self_query_proj, trampoline_autograd_apply_1, trampoline_autograd_apply_4], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add_1 => add_7
# l__self___deberta_encoder_layer_0_attention_output_layer_norm => add_5, mul_9
# l__self___deberta_encoder_layer_0_output_layer_norm => add_8, add_9, mul_14, mul_15, rsqrt_2, sub_7, var_mean_2
# l__self___deberta_encoder_layer_1_attention_self_query_proj => convert_element_type_31, view_22
# trampoline_autograd_apply_1 => full_default_5
# trampoline_autograd_apply_4 => convert_element_type_29, convert_element_type_30, lt_3, mul_13, sub_6, where_5
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: '*fp16', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (1536*x0)
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
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp14 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp20, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp27_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp25 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight,
        )
        tmp27_mean = tl.where(rmask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(rmask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(rmask & xmask, tmp27_weight_next, tmp27_weight)
    tmp27_tmp, tmp28_tmp, tmp29_tmp = triton_helpers.welford(
        tmp27_mean, tmp27_m2, tmp27_weight, 1
    )
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    tmp29 = tmp29_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp30 - tmp22
        tmp32 = 1536.0
        tmp33 = tmp28 / tmp32
        tmp34 = 1e-07
        tmp35 = tmp33 + tmp34
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp31 * tmp36
        tmp39 = tmp37 * tmp38
        tmp41 = tmp39 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp37, rmask & xmask)
        tl.store(out_ptr6 + (r1 + (1536*x0)), tmp42, rmask & xmask)
    tmp43 = 1536.0
    tmp44 = tmp28 / tmp43
    tmp45 = 1e-07
    tmp46 = tmp44 + tmp45
    tmp47 = libdevice.rsqrt(tmp46)
    tmp48 = tmp47 / tmp43
    tl.store(out_ptr7 + (x0), tmp48, xmask)
''')
