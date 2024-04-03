

# Original file: ./ElectraForQuestionAnswering__0_forward_133.0/ElectraForQuestionAnswering__0_forward_133.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/3e/c3ejyhmdjq66xpju7a53hlhdxccoa3daho5fsvlkhk23hbrqdfim.py
# Source Nodes: [add, iadd, l__mod___electra_embeddings_dropout, l__mod___electra_embeddings_layer_norm, l__mod___electra_embeddings_position_embeddings, l__mod___electra_embeddings_project, l__mod___electra_embeddings_token_type_embeddings, l__mod___electra_embeddings_word_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.view]
# add => add
# iadd => add_1
# l__mod___electra_embeddings_dropout => gt, mul_3, mul_4
# l__mod___electra_embeddings_layer_norm => add_2, add_3, convert_element_type_1, convert_element_type_2, mul_1, mul_2, rsqrt, sub_1, var_mean
# l__mod___electra_embeddings_position_embeddings => embedding_2
# l__mod___electra_embeddings_project => view
# l__mod___electra_embeddings_token_type_embeddings => embedding_1
# l__mod___electra_embeddings_word_embeddings => embedding
triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_0 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_0', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*i64', 4: '*fp16', 5: '*i64', 6: '*fp16', 7: '*i64', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*i1', 13: '*fp16', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30522), "index out of bounds: 0 <= tmp1 < 30522")
    tmp2 = tl.load(in_ptr1 + (r2 + (128*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 2, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 2), "index out of bounds: 0 <= tmp4 < 2")
    tmp5 = tl.load(in_ptr3 + (r2 + (128*tmp4)), rmask, other=0.0).to(tl.float32)
    tmp6 = tmp2 + tmp5
    tmp8 = tl.where(tmp7 < 0, tmp7 + 512, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 512), "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr5 + (r2 + (128*tmp8)), rmask, other=0.0).to(tl.float32)
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = 128.0
    tmp29 = tmp27 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tl.load(in_ptr6 + load_seed_offset)
    tmp34 = r2 + (128*x3)
    tmp35 = tl.rand(tmp33, (tmp34).to(tl.uint32))
    tmp36 = tmp35.to(tl.float32)
    tmp37 = 0.1
    tmp38 = tmp36 > tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp11 - tmp21
    tmp41 = tmp40 * tmp32
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp39 * tmp48
    tmp50 = 1.1111111111111112
    tmp51 = tmp49 * tmp50
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp38, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp51, rmask)
    tl.store(out_ptr1 + (x3), tmp21, None)
''')
