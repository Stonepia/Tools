

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ph/cph6a2dov7l6q3njxgymksh7heu7qqfxx3pjq4ukk27nropbyurz.py
# Source Nodes: [add, l__self___deberta_embeddings_layer_norm, l__self___deberta_encoder_layer_0_attention_output_layer_norm, l__self___deberta_encoder_layer_0_intermediate_dense, mul, trampoline_autograd_apply, trampoline_autograd_apply_1, trampoline_autograd_apply_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add => add_3
# l__self___deberta_embeddings_layer_norm => mul_1
# l__self___deberta_encoder_layer_0_attention_output_layer_norm => add_4, add_5, mul_8, mul_9, rsqrt_1, sub_5, var_mean_1
# l__self___deberta_encoder_layer_0_intermediate_dense => convert_element_type_22, view_18
# mul => add_2
# trampoline_autograd_apply => full_default_1, mul_3, where
# trampoline_autograd_apply_1 => full_default_5
# trampoline_autograd_apply_3 => convert_element_type_20, convert_element_type_21, lt_2, mul_7, sub_4, where_4
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_7 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_7', '''
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr2, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp25_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp16 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tl.where(tmp15, tmp10, tmp20)
        tmp22 = tmp21 * tmp12
        tmp23 = tmp14 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight,
        )
        tmp25_mean = tl.where(rmask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(rmask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(rmask & xmask, tmp25_weight_next, tmp25_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp23, rmask & xmask)
    tmp25_tmp, tmp26_tmp, tmp27_tmp = triton_helpers.welford(
        tmp25_mean, tmp25_m2, tmp25_weight, 1
    )
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    tmp30_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp30_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp30_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp30_mean_next, tmp30_m2_next, tmp30_weight_next = triton_helpers.welford_reduce(
            tmp29, tmp30_mean, tmp30_m2, tmp30_weight,
        )
        tmp30_mean = tl.where(rmask & xmask, tmp30_mean_next, tmp30_mean)
        tmp30_m2 = tl.where(rmask & xmask, tmp30_m2_next, tmp30_m2)
        tmp30_weight = tl.where(rmask & xmask, tmp30_weight_next, tmp30_weight)
    tmp30_tmp, tmp31_tmp, tmp32_tmp = triton_helpers.welford(
        tmp30_mean, tmp30_m2, tmp30_weight, 1
    )
    tmp30 = tmp30_tmp[:, None]
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp33 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp41 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tmp33 - tmp25
        tmp35 = 1536.0
        tmp36 = tmp31 / tmp35
        tmp37 = 1e-07
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp34 * tmp39
        tmp42 = tmp40 * tmp41
        tmp44 = tmp42 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp40, rmask & xmask)
        tl.store(out_ptr6 + (r1 + (1536*x0)), tmp45, rmask & xmask)
    tmp46 = 1536.0
    tmp47 = tmp31 / tmp46
    tmp48 = 1e-07
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.rsqrt(tmp49)
    tmp51 = tmp50 / tmp46
    tl.store(out_ptr7 + (x0), tmp51, xmask)
''')
