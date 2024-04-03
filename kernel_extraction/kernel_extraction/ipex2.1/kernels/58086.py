

# Original file: ./hf_Bart___60.0/hf_Bart___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/kj/ckjsmffxrypwybgvkjrioqusk2ggiwtcsfkruwj7u5bmcqttvkgs.py
# Source Nodes: [add, add_12, add_16, dropout_22, embedding_1, l__mod___model_model_decoder_embed_tokens, l__mod___model_model_decoder_layernorm_embedding, l__mod___model_model_encoder_layers_5_self_attn_layer_norm, mul_7], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# add_12 => add_39
# add_16 => add_48
# dropout_22 => clone_46
# embedding_1 => embedding_3
# l__mod___model_model_decoder_embed_tokens => embedding_2
# l__mod___model_model_decoder_layernorm_embedding => add_49, add_50, mul_52, mul_53, rsqrt_13, sub_19, var_mean_13
# l__mod___model_model_encoder_layers_5_self_attn_layer_norm => add_40, add_41, mul_44, mul_45, rsqrt_11, sub_17, var_mean_11
# mul_7 => mul_51
triton_red_fused_add_clone_embedding_mul_native_layer_norm_7 = async_compile.triton('triton_red_fused_add_clone_embedding_mul_native_layer_norm_7', '''
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_mul_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_embedding_mul_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tmp8 = tl.where(tmp7 < 0, tmp7 + 50265, tmp7)
        # tl.device_assert(((0 <= tmp8) & (tmp8 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp8 < 50265")
        tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = 1.0
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp4
        tmp22 = 768.0
        tmp23 = tmp5 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp32 = tl.where(tmp7 < 0, tmp7 + 50265, tmp7)
        # tl.device_assert(((0 <= tmp32) & (tmp32 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp32 < 50265")
        tmp33 = tl.load(in_ptr3 + (r1 + (768*tmp32)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = 1.0
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tmp38 = tmp37 - tmp15
        tmp39 = tmp16 / tmp22
        tmp40 = tmp39 + tmp24
        tmp41 = libdevice.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp42 * tmp28
        tmp44 = tmp43 + tmp30
        tl.store(out_ptr4 + (r1 + (768*x0)), tmp31, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (768*x0)), tmp44, rmask & xmask)
''')
