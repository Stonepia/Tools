

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/rb/crba2indncbg6ibxk2gsdkybicvja5kxrgthg2jqhoil3if7e225.py
# Source Nodes: [add_1, l__mod___deberta_encoder_layer_0_attention_output_layer_norm, l__mod___deberta_encoder_layer_0_output_layer_norm, l__mod___deberta_encoder_layer_1_attention_self_query_proj, trampoline_autograd_apply, trampoline_autograd_apply_4], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add_1 => add_7
# l__mod___deberta_encoder_layer_0_attention_output_layer_norm => add_5, mul_9
# l__mod___deberta_encoder_layer_0_output_layer_norm => add_8, add_9, mul_14, mul_15, rsqrt_2, sub_7, var_mean_2
# l__mod___deberta_encoder_layer_1_attention_self_query_proj => view_22
# trampoline_autograd_apply => full_default_1
# trampoline_autograd_apply_4 => convert_element_type_7, convert_element_type_8, lt_3, mul_13, sub_6, where_5
triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp16 = tmp14 * tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight,
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
        tl.store(in_out_ptr0 + (r1 + (1536*x0)), tmp19, rmask & xmask)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp26_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp24 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp26_mean_next, tmp26_m2_next, tmp26_weight_next = triton_helpers.welford_reduce(
            tmp25, tmp26_mean, tmp26_m2, tmp26_weight,
        )
        tmp26_mean = tl.where(rmask & xmask, tmp26_mean_next, tmp26_mean)
        tmp26_m2 = tl.where(rmask & xmask, tmp26_m2_next, tmp26_m2)
        tmp26_weight = tl.where(rmask & xmask, tmp26_weight_next, tmp26_weight)
    tmp26_tmp, tmp27_tmp, tmp28_tmp = triton_helpers.welford(
        tmp26_mean, tmp26_m2, tmp26_weight, 1
    )
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp29 - tmp21
        tmp31 = 1536.0
        tmp32 = tmp27 / tmp31
        tmp33 = 1e-07
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp38 = tmp36 * tmp37
        tmp40 = tmp38 + tmp39
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp36, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp40, rmask & xmask)
    tmp41 = 1536.0
    tmp42 = tmp27 / tmp41
    tmp43 = 1e-07
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp45 / tmp41
    tl.store(out_ptr6 + (x0), tmp46, xmask)
''')
