

# Original file: ./LayoutLMForMaskedLM__0_forward_205.0/LayoutLMForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/cp/ccp3h5oc23kdwj37sdkfkxuhnpq3swwcn2qdpslpmhdc2da4r6sc.py
# Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, l__mod___layoutlm_embeddings_dropout, l__mod___layoutlm_embeddings_h_position_embeddings, l__mod___layoutlm_embeddings_layer_norm, l__mod___layoutlm_embeddings_position_embeddings, l__mod___layoutlm_embeddings_token_type_embeddings, l__mod___layoutlm_embeddings_w_position_embeddings, l__mod___layoutlm_embeddings_word_embeddings, l__mod___layoutlm_embeddings_x_position_embeddings, l__mod___layoutlm_embeddings_x_position_embeddings_1, l__mod___layoutlm_embeddings_y_position_embeddings, l__mod___layoutlm_embeddings_y_position_embeddings_1, l__mod___layoutlm_encoder_layer_0_attention_self_query], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# add_7 => add_7
# l__mod___layoutlm_embeddings_dropout => gt, mul_3, mul_4
# l__mod___layoutlm_embeddings_h_position_embeddings => embedding_6
# l__mod___layoutlm_embeddings_layer_norm => add_8, add_9, mul_1, mul_2, rsqrt, sub_3, var_mean
# l__mod___layoutlm_embeddings_position_embeddings => embedding_1
# l__mod___layoutlm_embeddings_token_type_embeddings => embedding_8
# l__mod___layoutlm_embeddings_w_position_embeddings => embedding_7
# l__mod___layoutlm_embeddings_word_embeddings => embedding
# l__mod___layoutlm_embeddings_x_position_embeddings => embedding_2
# l__mod___layoutlm_embeddings_x_position_embeddings_1 => embedding_4
# l__mod___layoutlm_embeddings_y_position_embeddings => embedding_3
# l__mod___layoutlm_embeddings_y_position_embeddings_1 => embedding_5
# l__mod___layoutlm_encoder_layer_0_attention_self_query => view
triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*i1', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr10 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr11 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30522), "index out of bounds: 0 <= tmp1 < 30522")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 512, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 512), "index out of bounds: 0 <= tmp4 < 512")
    tmp5 = tl.load(in_ptr3 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 + tmp7
    tmp12 = tmp11 + tmp9
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tl.load(in_ptr9 + load_seed_offset)
    tmp36 = r2 + (768*x3)
    tmp37 = tl.rand(tmp35, (tmp36).to(tl.uint32))
    tmp38 = 0.1
    tmp39 = tmp37 > tmp38
    tmp40 = tmp18 - tmp28
    tmp41 = 768.0
    tmp42 = tmp34 / tmp41
    tmp43 = 1e-12
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp40 * tmp45
    tmp47 = tmp39.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tmp47 * tmp51
    tmp53 = 1.1111111111111112
    tmp54 = tmp52 * tmp53
    tmp55 = tmp45 / tmp41
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp39, rmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp46, rmask)
    tl.store(out_ptr6 + (r2 + (768*x3)), tmp54, rmask)
    tl.store(out_ptr7 + (x3), tmp55, None)
''')
