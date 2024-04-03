

# Original file: ./GPT2ForSequenceClassification__0_forward_133.0/GPT2ForSequenceClassification__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/mw/cmwhaqumfsbvqvwruhhgdvd5ry477fz2lualp3d2voc6bteu7iqx.py
# Source Nodes: [add, l__mod___transformer_drop, l__mod___transformer_h_0_ln_1, l__mod___transformer_wpe, l__mod___transformer_wte], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# l__mod___transformer_drop => gt, mul, mul_1
# l__mod___transformer_h_0_ln_1 => add_1, add_2, mul_2, mul_3, rsqrt, sub, var_mean
# l__mod___transformer_wpe => embedding_1
# l__mod___transformer_wte => embedding
triton_red_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_red_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_1', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    x2 = xindex % 1024
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr3 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (768*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tl.where(tmp6 < 0, tmp6 + 50257, tmp6)
        # tl.device_assert((0 <= tmp7) & (tmp7 < 50257), "index out of bounds: 0 <= tmp7 < 50257")
        tmp8 = tl.load(in_ptr2 + (r1 + (768*tmp7)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = 1.1111111111111112
        tmp13 = tmp11 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask, tmp15_weight_next, tmp15_weight)
        tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp28_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp28_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp28_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(out_ptr1 + (r1 + (768*x0)), rmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr3 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.where(tmp6 < 0, tmp6 + 50257, tmp6)
        # tl.device_assert((0 <= tmp20) & (tmp20 < 50257), "index out of bounds: 0 <= tmp20 < 50257")
        tmp21 = tl.load(in_ptr2 + (r1 + (768*tmp20)), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = 1.1111111111111112
        tmp26 = tmp24 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp28_mean_next, tmp28_m2_next, tmp28_weight_next = triton_helpers.welford_reduce(
            tmp27, tmp28_mean, tmp28_m2, tmp28_weight,
        )
        tmp28_mean = tl.where(rmask, tmp28_mean_next, tmp28_mean)
        tmp28_m2 = tl.where(rmask, tmp28_m2_next, tmp28_m2)
        tmp28_weight = tl.where(rmask, tmp28_weight_next, tmp28_weight)
    tmp28_tmp, tmp29_tmp, tmp30_tmp = triton_helpers.welford(
        tmp28_mean, tmp28_m2, tmp28_weight, 1
    )
    tmp28 = tmp28_tmp[:, None]
    tmp29 = tmp29_tmp[:, None]
    tmp30 = tmp30_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp31 = tl.load(out_ptr1 + (r1 + (768*x0)), rmask, eviction_policy='evict_first')
        tmp35 = tl.load(in_ptr3 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp47 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.where(tmp6 < 0, tmp6 + 50257, tmp6)
        # tl.device_assert((0 <= tmp33) & (tmp33 < 50257), "index out of bounds: 0 <= tmp33 < 50257")
        tmp34 = tl.load(in_ptr2 + (r1 + (768*tmp33)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = tmp34 + tmp35
        tmp37 = tmp32 * tmp36
        tmp38 = 1.1111111111111112
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 - tmp15
        tmp41 = 768.0
        tmp42 = tmp29 / tmp41
        tmp43 = 1e-05
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp40 * tmp45
        tmp48 = tmp46 * tmp47
        tmp50 = tmp48 + tmp49
        tl.store(out_ptr4 + (r1 + (768*x0)), tmp46, rmask)
        tl.store(out_ptr5 + (r1 + (768*x0)), tmp50, rmask)
    tmp51 = 768.0
    tmp52 = tmp29 / tmp51
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.rsqrt(tmp54)
    tmp56 = tmp55 / tmp51
    tl.store(out_ptr6 + (x0), tmp56, None)
''')
