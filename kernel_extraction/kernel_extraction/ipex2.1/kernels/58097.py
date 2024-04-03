

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/zy/czye7wrikwxgfm3hz3feiqt5qvhy6aht3h6r6au5sl5te3u3ztyl.py
# Source Nodes: [add, add_12, add_16, dropout_22, embedding_1, l__self___model_model_decoder_embed_tokens, l__self___model_model_decoder_layernorm_embedding, l__self___model_model_decoder_layers_0_self_attn_q_proj, l__self___model_model_encoder_layers_5_fc1, l__self___model_model_encoder_layers_5_self_attn_layer_norm, mul_7], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# add_12 => add_39
# add_16 => add_48
# dropout_22 => clone_46
# embedding_1 => embedding_3
# l__self___model_model_decoder_embed_tokens => embedding_2
# l__self___model_model_decoder_layernorm_embedding => add_49, add_50, mul_52, mul_53, rsqrt_13, sub_19, var_mean_13
# l__self___model_model_decoder_layers_0_self_attn_q_proj => convert_element_type_120
# l__self___model_model_encoder_layers_5_fc1 => convert_element_type_113
# l__self___model_model_encoder_layers_5_self_attn_layer_norm => add_40, add_41, mul_44, mul_45, rsqrt_11, sub_17, var_mean_11
# mul_7 => mul_51
triton_red_fused__to_copy_add_clone_embedding_mul_native_layer_norm_7 = async_compile.triton('triton_red_fused__to_copy_add_clone_embedding_mul_native_layer_norm_7', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp32', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_embedding_mul_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_embedding_mul_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
        tmp9 = tl.where(tmp8 < 0, tmp8 + 50265, tmp8)
        # tl.device_assert(((0 <= tmp9) & (tmp9 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp9 < 50265")
        tmp10 = tl.load(in_ptr3 + (r1 + (768*tmp9)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = 1.0
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight,
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 + tmp21
        tmp23 = tmp22 - tmp5
        tmp24 = 768.0
        tmp25 = tmp6 / tmp24
        tmp26 = 1e-05
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tl.where(tmp8 < 0, tmp8 + 50265, tmp8)
        # tl.device_assert(((0 <= tmp35) & (tmp35 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp35 < 50265")
        tmp36 = tl.load(in_ptr3 + (r1 + (768*tmp35)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = 1.0
        tmp38 = tmp36 * tmp37
        tmp40 = tmp38 + tmp39
        tmp41 = tmp40 - tmp16
        tmp42 = tmp17 / tmp24
        tmp43 = tmp42 + tmp26
        tmp44 = libdevice.rsqrt(tmp43)
        tmp45 = tmp41 * tmp44
        tmp46 = tmp45 * tmp30
        tmp47 = tmp46 + tmp32
        tmp48 = tmp47.to(tl.float32)
        tl.store(out_ptr4 + (r1 + (768*x0)), tmp33, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (768*x0)), tmp34, rmask & xmask)
        tl.store(out_ptr6 + (r1 + (768*x0)), tmp47, rmask & xmask)
        tl.store(out_ptr7 + (r1 + (768*x0)), tmp48, rmask & xmask)
''')
