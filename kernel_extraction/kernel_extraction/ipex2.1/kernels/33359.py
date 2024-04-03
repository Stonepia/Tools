

# Original file: ./RobertaForCausalLM__0_forward_133.0/RobertaForCausalLM__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ry/cryktcet7w3ukqra2bpdzb2khpxp7samk3fltnrua5tdmc6ri6fx.py
# Source Nodes: [add, add_1, add_2, iadd, int_1, l__self___roberta_embeddings_dropout, l__self___roberta_embeddings_layer_norm, l__self___roberta_embeddings_position_embeddings, l__self___roberta_embeddings_token_type_embeddings, l__self___roberta_embeddings_word_embeddings, l__self___roberta_encoder_layer_0_attention_self_query, long, mul_1, ne, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne, aten.view]
# add => add
# add_1 => add_1
# add_2 => add_2
# iadd => add_3
# int_1 => convert_element_type
# l__self___roberta_embeddings_dropout => gt, mul_4, mul_5
# l__self___roberta_embeddings_layer_norm => add_4, add_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# l__self___roberta_embeddings_position_embeddings => embedding_2
# l__self___roberta_embeddings_token_type_embeddings => embedding_1
# l__self___roberta_embeddings_word_embeddings => embedding
# l__self___roberta_encoder_layer_0_attention_self_query => convert_element_type_3, view
# long => convert_element_type_2
# mul_1 => mul_1
# ne => ne
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_view_1 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_view_1', '''
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
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp16', 12: '*fp32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
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
    x0 = xindex
    r3 = rindex
    x1 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr7 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tl.where(tmp4 < 0, tmp4 + 50265, tmp4)
    # tl.device_assert((0 <= tmp11) & (tmp11 < 50265), "index out of bounds: 0 <= tmp11 < 50265")
    tmp12 = tl.load(in_ptr1 + (r3 + (768*tmp11)), rmask, other=0.0)
    tmp14 = tl.where(tmp13 < 0, tmp13 + 2, tmp13)
    # tl.device_assert((0 <= tmp14) & (tmp14 < 2), "index out of bounds: 0 <= tmp14 < 2")
    tmp15 = tl.load(in_ptr3 + (r3 + (768*tmp14)), rmask, other=0.0)
    tmp16 = tmp12 + tmp15
    tmp17 = tl.where(tmp10 < 0, tmp10 + 512, tmp10)
    # tl.device_assert((0 <= tmp17) & (tmp17 < 512), "index out of bounds: 0 <= tmp17 < 512")
    tmp18 = tl.load(in_ptr4 + (r3 + (768*tmp17)), rmask, other=0.0)
    tmp19 = tmp16 + tmp18
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
    tmp36 = tl.load(in_ptr5 + load_seed_offset)
    tmp37 = r3 + (768*x0)
    tmp38 = tl.rand(tmp36, (tmp37).to(tl.uint32))
    tmp39 = 0.1
    tmp40 = tmp38 > tmp39
    tmp41 = tmp19 - tmp29
    tmp42 = 768.0
    tmp43 = tmp35 / tmp42
    tmp44 = 1e-12
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.rsqrt(tmp45)
    tmp47 = tmp41 * tmp46
    tmp48 = tmp40.to(tl.float32)
    tmp50 = tmp47 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tmp48 * tmp52
    tmp54 = 1.1111111111111112
    tmp55 = tmp53 * tmp54
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp46 / tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr4 + (r3 + (768*x0)), tmp40, rmask)
    tl.store(out_ptr5 + (r3 + (768*x0)), tmp47, rmask)
    tl.store(out_ptr6 + (r3 + (768*x0)), tmp56, rmask)
    tl.store(out_ptr7 + (x0), tmp57, None)
''')
