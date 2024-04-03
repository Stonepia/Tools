

# Original file: ./RobertaForQuestionAnswering__0_forward_133.0/RobertaForQuestionAnswering__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/cb/ccbqlax6tgu3yclcpm65gr2juugxv2fondvgsi7rd45fi37w633j.py
# Source Nodes: [add, add_1, add_2, iadd, int_1, l__mod___roberta_embeddings_dropout, l__mod___roberta_embeddings_layer_norm, l__mod___roberta_embeddings_position_embeddings, l__mod___roberta_embeddings_token_type_embeddings, l__mod___roberta_embeddings_word_embeddings, long, mul_1, ne, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_dropout, aten.native_layer_norm, aten.ne]
# add => add
# add_1 => add_1
# add_2 => add_2
# iadd => add_3
# int_1 => convert_element_type_1
# l__mod___roberta_embeddings_dropout => gt, mul_4, mul_5
# l__mod___roberta_embeddings_layer_norm => add_4, add_5, convert_element_type_4, convert_element_type_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# l__mod___roberta_embeddings_position_embeddings => embedding_2
# l__mod___roberta_embeddings_token_type_embeddings => embedding_1
# l__mod___roberta_embeddings_word_embeddings => embedding
# long => convert_element_type_3
# mul_1 => mul_1
# ne => ne
# type_as => convert_element_type_2
triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_1 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_1', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp16', 4: '*i64', 5: '*fp16', 6: '*fp16', 7: '*i64', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*i1', 13: '*fp16', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
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
    tmp51 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp54 = tl.load(in_ptr7 + (r3), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp12 = tl.load(in_ptr1 + (r3 + (768*tmp11)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp13 < 0, tmp13 + 2, tmp13)
    # tl.device_assert((0 <= tmp14) & (tmp14 < 2), "index out of bounds: 0 <= tmp14 < 2")
    tmp15 = tl.load(in_ptr3 + (r3 + (768*tmp14)), rmask, other=0.0).to(tl.float32)
    tmp16 = tmp12 + tmp15
    tmp17 = tl.where(tmp10 < 0, tmp10 + 512, tmp10)
    # tl.device_assert((0 <= tmp17) & (tmp17 < 512), "index out of bounds: 0 <= tmp17 < 512")
    tmp18 = tl.load(in_ptr4 + (r3 + (768*tmp17)), rmask, other=0.0).to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = 768.0
    tmp38 = tmp36 / tmp37
    tmp39 = 1e-12
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tl.load(in_ptr5 + load_seed_offset)
    tmp43 = r3 + (768*x0)
    tmp44 = tl.rand(tmp42, (tmp43).to(tl.uint32))
    tmp45 = tmp44.to(tl.float32)
    tmp46 = 0.1
    tmp47 = tmp45 > tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp20 - tmp30
    tmp50 = tmp49 * tmp41
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp50 * tmp52
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp53 + tmp55
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp48 * tmp57
    tmp59 = 1.1111111111111112
    tmp60 = tmp58 * tmp59
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr0 + (r3 + (768*x0)), tmp19, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, None)
    tl.store(out_ptr3 + (r3 + (768*x0)), tmp47, rmask)
    tl.store(out_ptr4 + (r3 + (768*x0)), tmp60, rmask)
    tl.store(out_ptr1 + (x0), tmp30, None)
''')
