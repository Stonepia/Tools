

# Original file: ./BertForQuestionAnswering__0_forward_169.0/BertForQuestionAnswering__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/lh/clhwvogpbqghv2f7hhtva3apzmqzwml2koai5sr6mkhpqplgxuqf.py
# Source Nodes: [add, iadd, l__self___bert_embeddings_dropout, l__self___bert_embeddings_layer_norm, l__self___bert_embeddings_position_embeddings, l__self___bert_embeddings_token_type_embeddings, l__self___bert_embeddings_word_embeddings, l__self___bert_encoder_layer_0_attention_self_query], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add => add
# iadd => add_1
# l__self___bert_embeddings_dropout => gt, mul_3, mul_4
# l__self___bert_embeddings_layer_norm => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# l__self___bert_embeddings_position_embeddings => embedding_2
# l__self___bert_embeddings_token_type_embeddings => embedding_1
# l__self___bert_embeddings_word_embeddings => embedding
# l__self___bert_encoder_layer_0_attention_self_query => convert_element_type, view
triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*bf16', 12: '*fp32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30522), "index out of bounds: 0 <= tmp1 < 30522")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 2, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 2), "index out of bounds: 0 <= tmp4 < 2")
    tmp5 = tl.load(in_ptr3 + (r2 + (768*tmp4)), rmask, other=0.0)
    tmp6 = tmp2 + tmp5
    tmp8 = tl.where(tmp7 < 0, tmp7 + 512, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 512), "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr5 + (r2 + (768*tmp8)), rmask, other=0.0)
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.load(in_ptr6 + load_seed_offset)
    tmp28 = r2 + (768*x3)
    tmp29 = tl.rand(tmp27, (tmp28).to(tl.uint32))
    tmp30 = 0.1
    tmp31 = tmp29 > tmp30
    tmp32 = tmp10 - tmp20
    tmp33 = 768.0
    tmp34 = tmp26 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp31.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp39 * tmp43
    tmp45 = 1.1111111111111112
    tmp46 = tmp44 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp37 / tmp33
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp31, rmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp38, rmask)
    tl.store(out_ptr6 + (r2 + (768*x3)), tmp47, rmask)
    tl.store(out_ptr7 + (x3), tmp48, None)
''')
