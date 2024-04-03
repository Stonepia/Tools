

# Original file: ./MegatronBertForQuestionAnswering__0_forward_205.0/MegatronBertForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/do/cdob5fcd3tyu7sul4m3m7wkyuagmjfxodxx36dseelmuo5igoijj.py
# Source Nodes: [add, iadd, l__mod___bert_embeddings_dropout, l__mod___bert_embeddings_position_embeddings, l__mod___bert_embeddings_token_type_embeddings, l__mod___bert_embeddings_word_embeddings, l__mod___bert_encoder_layer_0_attention_ln, l__mod___bert_encoder_layer_0_attention_self_query], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.view]
# add => add
# iadd => add_1
# l__mod___bert_embeddings_dropout => gt, mul_1, mul_2
# l__mod___bert_embeddings_position_embeddings => embedding_2
# l__mod___bert_embeddings_token_type_embeddings => embedding_1
# l__mod___bert_embeddings_word_embeddings => embedding
# l__mod___bert_encoder_layer_0_attention_ln => add_2, add_3, convert_element_type_1, convert_element_type_2, mul_3, mul_4, rsqrt, sub_1, var_mean
# l__mod___bert_encoder_layer_0_attention_self_query => view
triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_1 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_1', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*i64', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*i1', 10: '*fp16', 11: '*fp32', 12: '*fp16', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr2, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
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
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (1024*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tl.where(tmp7 < 0, tmp7 + 29056, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 29056), "index out of bounds: 0 <= tmp8 < 29056")
    tmp9 = tl.load(in_ptr2 + (r1 + (1024*tmp8)), rmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp13 = tl.where(tmp12 < 0, tmp12 + 512, tmp12)
    # tl.device_assert((0 <= tmp13) & (tmp13 < 512), "index out of bounds: 0 <= tmp13 < 512")
    tmp14 = tl.load(in_ptr5 + (r1 + (1024*tmp13)), rmask, other=0.0).to(tl.float32)
    tmp15 = tmp11 + tmp14
    tmp16 = tmp6 * tmp15
    tmp17 = 1.1111111111111112
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 1024, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = 1024.0
    tmp37 = tmp35 / tmp36
    tmp38 = 1e-12
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp19 - tmp29
    tmp42 = tmp41 * tmp40
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 * tmp44
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 + tmp47
    tmp49 = tmp48.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp5, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp18, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp40, None)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp49, rmask)
    tl.store(out_ptr3 + (x0), tmp29, None)
''')
