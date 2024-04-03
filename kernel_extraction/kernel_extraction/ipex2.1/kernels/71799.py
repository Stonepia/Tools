

# Original file: ./LayoutLMForSequenceClassification__0_forward_133.0/LayoutLMForSequenceClassification__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/sm/csmu5m6gfxls2aooqi757quw7xyput5o3noel3jnnuicbmbky6mb.py
# Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, l__mod___layoutlm_embeddings_dropout, l__mod___layoutlm_embeddings_h_position_embeddings, l__mod___layoutlm_embeddings_layer_norm, l__mod___layoutlm_embeddings_position_embeddings, l__mod___layoutlm_embeddings_token_type_embeddings, l__mod___layoutlm_embeddings_w_position_embeddings, l__mod___layoutlm_embeddings_word_embeddings, l__mod___layoutlm_embeddings_x_position_embeddings, l__mod___layoutlm_embeddings_x_position_embeddings_1, l__mod___layoutlm_embeddings_y_position_embeddings, l__mod___layoutlm_embeddings_y_position_embeddings_1], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm]
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
# l__mod___layoutlm_embeddings_layer_norm => add_8, add_9, convert_element_type_1, convert_element_type_2, mul_1, mul_2, rsqrt, sub_3, var_mean
# l__mod___layoutlm_embeddings_position_embeddings => embedding_1
# l__mod___layoutlm_embeddings_token_type_embeddings => embedding_8
# l__mod___layoutlm_embeddings_w_position_embeddings => embedding_7
# l__mod___layoutlm_embeddings_word_embeddings => embedding
# l__mod___layoutlm_embeddings_x_position_embeddings => embedding_2
# l__mod___layoutlm_embeddings_x_position_embeddings_1 => embedding_4
# l__mod___layoutlm_embeddings_y_position_embeddings => embedding_3
# l__mod___layoutlm_embeddings_y_position_embeddings_1 => embedding_5
triton_per_fused_add_embedding_native_dropout_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*i64', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*i64', 12: '*bf16', 13: '*bf16', 14: '*fp32', 15: '*i1', 16: '*bf16', 17: 'i32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr10 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp53 = tl.load(in_ptr11 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30522), "index out of bounds: 0 <= tmp1 < 30522")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 512, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 512), "index out of bounds: 0 <= tmp4 < 512")
    tmp5 = tl.load(in_ptr3 + (r2 + (768*tmp4)), rmask, other=0.0).to(tl.float32)
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 + tmp7
    tmp12 = tmp11 + tmp9
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 768, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = 768.0
    tmp37 = tmp35 / tmp36
    tmp38 = 1e-12
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tl.load(in_ptr9 + load_seed_offset)
    tmp42 = r2 + (768*x3)
    tmp43 = tl.rand(tmp41, (tmp42).to(tl.uint32))
    tmp44 = tmp43.to(tl.float32)
    tmp45 = 0.1
    tmp46 = tmp44 > tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp19 - tmp29
    tmp49 = tmp48 * tmp40
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 * tmp51
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp52 + tmp54
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp47 * tmp56
    tmp58 = 1.1111111111111112
    tmp59 = tmp57 * tmp58
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp18, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp40, None)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp46, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp59, rmask)
    tl.store(out_ptr0 + (x3), tmp29, None)
''')
