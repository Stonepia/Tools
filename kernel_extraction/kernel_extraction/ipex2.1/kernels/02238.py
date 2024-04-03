

# Original file: ./AlbertForQuestionAnswering__0_forward_133.0/AlbertForQuestionAnswering__0_forward_133.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ra/crahvrlfssgagdakcnzscoynpyt7z2afsdgxvyepc4orig3vwntt.py
# Source Nodes: [add, iadd, l__self___albert_embeddings_layer_norm, l__self___albert_embeddings_position_embeddings, l__self___albert_embeddings_token_type_embeddings, l__self___albert_embeddings_word_embeddings, l__self___albert_encoder_embedding_hidden_mapping_in], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add => add
# iadd => add_1
# l__self___albert_embeddings_layer_norm => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# l__self___albert_embeddings_position_embeddings => embedding_2
# l__self___albert_embeddings_token_type_embeddings => embedding_1
# l__self___albert_embeddings_word_embeddings => embedding
# l__self___albert_encoder_embedding_hidden_mapping_in => convert_element_type, view
triton_per_fused__to_copy_add_embedding_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', '''
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_native_layer_norm_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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
    tmp34 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30000, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30000), "index out of bounds: 0 <= tmp1 < 30000")
    tmp2 = tl.load(in_ptr1 + (r2 + (128*tmp1)), rmask, other=0.0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 2, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 2), "index out of bounds: 0 <= tmp4 < 2")
    tmp5 = tl.load(in_ptr3 + (r2 + (128*tmp4)), rmask, other=0.0)
    tmp6 = tmp2 + tmp5
    tmp8 = tl.where(tmp7 < 0, tmp7 + 512, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 512), "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr5 + (r2 + (128*tmp8)), rmask, other=0.0)
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = 128.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp32 / tmp28
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp33, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp38, rmask)
    tl.store(out_ptr5 + (x3), tmp39, None)
''')
