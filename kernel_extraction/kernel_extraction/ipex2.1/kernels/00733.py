

# Original file: ./YituTechConvBert__0_backward_171.1/YituTechConvBert__0_backward_171.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/la/cla6nc5s56sczalwavga6ddkooftitq5rk2rpl2h4jokbl7d2le6.py
# Source Nodes: [l__mod___convbert_embeddings_layer_norm], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___convbert_embeddings_layer_norm => convert_element_type_1
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_35 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_35', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*i1', 7: '*bf16', 8: '*bf16', 9: '*fp32', 10: '*fp32', 11: '*i64', 12: '*i64', 13: '*fp32', 14: '*bf16', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr4', 'out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (393216*x1)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr6 + (r2 + (768*x3)), rmask)
    tmp17 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr8 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 - tmp26
    tmp29 = tmp27 * tmp28
    tmp30 = tmp19 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 768.0
    tmp36 = tmp28 / tmp35
    tmp37 = tmp19 * tmp35
    tmp38 = tmp37 - tmp23
    tmp39 = tmp29 * tmp34
    tmp40 = tmp38 - tmp39
    tmp41 = tmp36 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp44 = tl.where(tmp43 < 0, tmp43 + 2, tmp43)
    tmp45 = tl.full([1], -1, tl.int64)
    tmp46 = tmp43 == tmp45
    tmp47 = tmp42.to(tl.float32)
    tmp48 = 0.0
    tmp49 = tl.where(tmp46, tmp48, tmp47)
    tmp51 = tl.where(tmp50 < 0, tmp50 + 30522, tmp50)
    tmp52 = tl.full([1], 0, tl.int64)
    tmp53 = tmp50 == tmp52
    tmp54 = tl.where(tmp53, tmp48, tmp47)
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp16, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp42, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r2 + (768*tmp44), [RBLOCK])), tmp49, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r2 + (768*tmp51), [RBLOCK])), tmp54, rmask)
''')
