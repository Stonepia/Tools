

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/qa/cqapplmh7vf2pzdzte5yiprsbxcgrm5iuj64tp2i37pyvm67r2ak.py
# Source Nodes: [add, l__mod___deberta_embeddings_layer_norm, l__mod___deberta_encoder_layer_0_attention_output_layer_norm, l__mod___deberta_encoder_layer_0_intermediate_dense, mul, trampoline_autograd_apply, trampoline_autograd_apply_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add => add_3
# l__mod___deberta_embeddings_layer_norm => mul_1
# l__mod___deberta_encoder_layer_0_attention_output_layer_norm => add_4, add_5, mul_8, mul_9, rsqrt_1, sub_5, var_mean_1
# l__mod___deberta_encoder_layer_0_intermediate_dense => view_18
# mul => add_2
# trampoline_autograd_apply => full_default_1, mul_3, where
# trampoline_autograd_apply_3 => convert_element_type_5, convert_element_type_6, lt_2, mul_7, sub_4, where_4
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_5 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_5', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
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
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tl.where(tmp14, tmp10, tmp19)
        tmp21 = tmp20 * tmp12
        tmp22 = tmp13 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(in_out_ptr0 + (r1 + (1536*x0)), tmp22, rmask & xmask)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp29_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp27 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp29_mean_next, tmp29_m2_next, tmp29_weight_next = triton_helpers.welford_reduce(
            tmp28, tmp29_mean, tmp29_m2, tmp29_weight,
        )
        tmp29_mean = tl.where(rmask & xmask, tmp29_mean_next, tmp29_mean)
        tmp29_m2 = tl.where(rmask & xmask, tmp29_m2_next, tmp29_m2)
        tmp29_weight = tl.where(rmask & xmask, tmp29_weight_next, tmp29_weight)
    tmp29_tmp, tmp30_tmp, tmp31_tmp = triton_helpers.welford(
        tmp29_mean, tmp29_m2, tmp29_weight, 1
    )
    tmp29 = tmp29_tmp[:, None]
    tmp30 = tmp30_tmp[:, None]
    tmp31 = tmp31_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32 - tmp24
        tmp34 = 1536.0
        tmp35 = tmp30 / tmp34
        tmp36 = 1e-07
        tmp37 = tmp35 + tmp36
        tmp38 = libdevice.rsqrt(tmp37)
        tmp39 = tmp33 * tmp38
        tmp41 = tmp39 * tmp40
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp39, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp43, rmask & xmask)
    tmp44 = 1536.0
    tmp45 = tmp30 / tmp44
    tmp46 = 1e-07
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp48 / tmp44
    tl.store(out_ptr6 + (x0), tmp49, xmask)
''')
