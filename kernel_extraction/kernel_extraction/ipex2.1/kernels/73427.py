

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/6s/c6sc3flpevp3atq2yd34sfgx3qrn3l7leettybu2njsohcw7vnjf.py
# Source Nodes: [add, l__self___deberta_encoder_layer_0_attention_output_layer_norm, l__self___deberta_encoder_layer_0_intermediate_dense, trampoline_autograd_apply_1, trampoline_autograd_apply_3], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# add => add_3
# l__self___deberta_encoder_layer_0_attention_output_layer_norm => add_4, add_5, mul_8, mul_9, rsqrt_1, sub_5, var_mean_1
# l__self___deberta_encoder_layer_0_intermediate_dense => convert_element_type_22, view_18
# trampoline_autograd_apply_1 => full_default_5
# trampoline_autograd_apply_3 => convert_element_type_20, convert_element_type_21, lt_2, mul_7, sub_4, where_4
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
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
        tl.store(out_ptr1 + (r1 + (1536*x0)), tmp8, rmask & xmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp31_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(out_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = 0.0
        tmp24 = tl.where(tmp21, tmp23, tmp22)
        tmp25 = 1.1111111111111112
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp27 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight,
        )
        tmp31_mean = tl.where(rmask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(rmask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(rmask & xmask, tmp31_weight_next, tmp31_weight)
    tmp31_tmp, tmp32_tmp, tmp33_tmp = triton_helpers.welford(
        tmp31_mean, tmp31_m2, tmp31_weight, 1
    )
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    tmp33 = tmp33_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp34 = tl.load(out_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp35 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp50 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp52 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = 0.0
        tmp37 = tl.where(tmp34, tmp36, tmp35)
        tmp38 = 1.1111111111111112
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 + tmp41
        tmp43 = tmp42 - tmp18
        tmp44 = 1536.0
        tmp45 = tmp32 / tmp44
        tmp46 = 1e-07
        tmp47 = tmp45 + tmp46
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp51 = tmp49 * tmp50
        tmp53 = tmp51 + tmp52
        tmp54 = tmp53.to(tl.float32)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp49, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp54, rmask & xmask)
    tmp55 = 1536.0
    tmp56 = tmp32 / tmp55
    tmp57 = 1e-07
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.rsqrt(tmp58)
    tmp60 = tmp59 / tmp55
    tl.store(out_ptr6 + (x0), tmp60, xmask)
''')
