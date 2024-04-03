

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/id/cid27z4yotj2itaulnvexp3robfetz7eg7rr5ibgbconi44prmeo.py
# Source Nodes: [add, add_12, add_16, dropout_22, embedding_1, l__mod___model_model_decoder_embed_tokens, l__mod___model_model_decoder_layernorm_embedding, l__mod___model_model_encoder_layers_5_self_attn_layer_norm, mul_7], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mul, aten.native_layer_norm]
# add => add
# add_12 => add_39
# add_16 => add_48
# dropout_22 => clone_46
# embedding_1 => embedding_3
# l__mod___model_model_decoder_embed_tokens => embedding_2
# l__mod___model_model_decoder_layernorm_embedding => add_49, add_50, convert_element_type_51, convert_element_type_52, mul_52, mul_53, rsqrt_13, sub_19, var_mean_13
# l__mod___model_model_encoder_layers_5_self_attn_layer_norm => add_40, add_41, convert_element_type_44, convert_element_type_45, mul_44, mul_45, rsqrt_11, sub_17, var_mean_11
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_mul_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
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
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
        tmp9 = tl.where(tmp8 < 0, tmp8 + 50265, tmp8)
        # tl.device_assert(((0 <= tmp9) & (tmp9 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp9 < 50265")
        tmp10 = tl.load(in_ptr3 + (r1 + (768*tmp9)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = 1.0
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr4 + (1536 + r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp5
        tmp25 = 768.0
        tmp26 = tmp6 / tmp25
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tl.where(tmp8 < 0, tmp8 + 50265, tmp8)
        # tl.device_assert(((0 <= tmp38) & (tmp38 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp38 < 50265")
        tmp39 = tl.load(in_ptr3 + (r1 + (768*tmp38)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = 1.0
        tmp41 = tmp39 * tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp44 - tmp17
        tmp46 = tmp18 / tmp25
        tmp47 = tmp46 + tmp27
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp45 * tmp48
        tmp50 = tmp49 * tmp32
        tmp51 = tmp50 + tmp35
        tmp52 = tmp51.to(tl.float32)
        tl.store(out_ptr4 + (r1 + (768*x0)), tmp37, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (768*x0)), tmp52, rmask & xmask)
''')
