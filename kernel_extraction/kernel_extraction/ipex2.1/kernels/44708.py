

# Original file: ./MegatronBertForCausalLM__0_forward_349.0/MegatronBertForCausalLM__0_forward_349.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/zi/czi6lrwt6mg6kotkboxdcvcbmcyifqwtucwtdufdrzy3c6wmknwt.py
# Source Nodes: [add, iadd, l__mod___bert_embeddings_dropout, l__mod___bert_embeddings_position_embeddings, l__mod___bert_embeddings_token_type_embeddings, l__mod___bert_embeddings_word_embeddings, l__mod___bert_encoder_layer_0_attention_ln, l__mod___bert_encoder_layer_0_attention_self_query], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add => add
# iadd => add_1
# l__mod___bert_embeddings_dropout => gt, mul_1, mul_2
# l__mod___bert_embeddings_position_embeddings => embedding_2
# l__mod___bert_embeddings_token_type_embeddings => embedding_1
# l__mod___bert_embeddings_word_embeddings => embedding
# l__mod___bert_encoder_layer_0_attention_ln => add_2, add_3, mul_3, mul_4, rsqrt, sub_1, var_mean
# l__mod___bert_encoder_layer_0_attention_self_query => view
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr2, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp6 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (1024*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.where(tmp6 < 0, tmp6 + 29056, tmp6)
    # tl.device_assert((0 <= tmp7) & (tmp7 < 29056), "index out of bounds: 0 <= tmp7 < 29056")
    tmp8 = tl.load(in_ptr2 + (r1 + (1024*tmp7)), rmask, other=0.0)
    tmp10 = tmp8 + tmp9
    tmp12 = tl.where(tmp11 < 0, tmp11 + 512, tmp11)
    # tl.device_assert((0 <= tmp12) & (tmp12 < 512), "index out of bounds: 0 <= tmp12 < 512")
    tmp13 = tl.load(in_ptr5 + (r1 + (1024*tmp12)), rmask, other=0.0)
    tmp14 = tmp10 + tmp13
    tmp15 = tmp5 * tmp14
    tmp16 = 1.1111111111111112
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tl.full([1], 1024, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 / tmp26
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp17 - tmp27
    tmp35 = 1024.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-12
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tmp39 / tmp35
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp4, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp17, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp40, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp44, rmask)
    tl.store(out_ptr7 + (x0), tmp45, None)
''')
