

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/2g/c2g6b4nk4j6krr4e7hugxe2qaiya7zxutcixy7s3mnag3icsbrjh.py
# Source Nodes: [add, l__mod___deberta_encoder_layer_0_attention_output_layer_norm, l__mod___deberta_encoder_layer_0_intermediate_dense, trampoline_autograd_apply, trampoline_autograd_apply_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add => add_3
# l__mod___deberta_encoder_layer_0_attention_output_layer_norm => add_4, add_5, mul_8, mul_9, rsqrt_1, sub_5, var_mean_1
# l__mod___deberta_encoder_layer_0_intermediate_dense => view_18
# trampoline_autograd_apply => full_default_1
# trampoline_autograd_apply_3 => convert_element_type_5, convert_element_type_6, lt_2, mul_7, sub_4, where_4
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_6 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_6', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp15 = tmp13 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp29_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(out_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = 0.0
        tmp23 = tl.where(tmp20, tmp22, tmp21)
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
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
        tmp32 = tl.load(out_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp33 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp47 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = 0.0
        tmp35 = tl.where(tmp32, tmp34, tmp33)
        tmp36 = 1.1111111111111112
        tmp37 = tmp35 * tmp36
        tmp39 = tmp37 + tmp38
        tmp40 = tmp39 - tmp17
        tmp41 = 1536.0
        tmp42 = tmp30 / tmp41
        tmp43 = 1e-07
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp40 * tmp45
        tmp48 = tmp46 * tmp47
        tmp50 = tmp48 + tmp49
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp46, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp50, rmask & xmask)
    tmp51 = 1536.0
    tmp52 = tmp30 / tmp51
    tmp53 = 1e-07
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.rsqrt(tmp54)
    tmp56 = tmp55 / tmp51
    tl.store(out_ptr6 + (x0), tmp56, xmask)
''')
