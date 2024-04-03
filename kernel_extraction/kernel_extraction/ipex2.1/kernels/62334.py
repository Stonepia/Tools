

# Original file: ./GoogleFnet__0_forward_61.0/GoogleFnet__0_forward_61.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/rz/crzuki7x57aqfhpabk3qytqrmj7w4f2bfzipuiley4unvt5be5yx.py
# Source Nodes: [add, iadd, l__mod___fnet_embeddings_layer_norm, l__mod___fnet_embeddings_position_embeddings, l__mod___fnet_embeddings_projection, l__mod___fnet_embeddings_token_type_embeddings, l__mod___fnet_embeddings_word_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add => add
# iadd => add_1
# l__mod___fnet_embeddings_layer_norm => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
# l__mod___fnet_embeddings_position_embeddings => embedding_2
# l__mod___fnet_embeddings_projection => view
# l__mod___fnet_embeddings_token_type_embeddings => embedding_1
# l__mod___fnet_embeddings_word_embeddings => embedding
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp34 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32000, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 32000), "index out of bounds: 0 <= tmp1 < 32000")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 4, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
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
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp33, rmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp37, rmask)
    tl.store(out_ptr5 + (x3), tmp38, None)
''')
