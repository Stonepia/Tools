

# Original file: ./ElectraForQuestionAnswering__0_backward_135.1/ElectraForQuestionAnswering__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/6b/c6blz3svm3vsodfqsilyennyylqg4x2q3c4od6xjtjm2owwb5t6j.py
# Source Nodes: [l__mod___electra_embeddings_layer_norm], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___electra_embeddings_layer_norm => convert_element_type_1
triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', '''
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
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*i64', 8: '*bf16', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = 128.0
    tmp26 = tmp18 / tmp25
    tmp27 = tmp9 * tmp25
    tmp28 = tmp27 - tmp13
    tmp29 = tmp19 * tmp24
    tmp30 = tmp28 - tmp29
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tl.where(tmp33 < 0, tmp33 + 2, tmp33)
    tmp35 = tl.full([1, 1], -1, tl.int64)
    tmp36 = tmp33 == tmp35
    tmp37 = tmp32.to(tl.float32)
    tmp38 = 0.0
    tmp39 = tl.where(tmp36, tmp38, tmp37)
    tmp41 = tl.where(tmp40 < 0, tmp40 + 30522, tmp40)
    tmp42 = tl.full([1, 1], 0, tl.int64)
    tmp43 = tmp40 == tmp42
    tmp44 = tl.where(tmp43, tmp38, tmp37)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp32, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (128*tmp34), [XBLOCK, RBLOCK])), tmp39, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (128*tmp41), [XBLOCK, RBLOCK])), tmp44, rmask)
''')
