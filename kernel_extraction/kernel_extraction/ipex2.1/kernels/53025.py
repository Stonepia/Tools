

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ah/cahs37toie6efsfvd7wh6nfexwnl53pw4fk2ytr2mlmera33ir24.py
# Source Nodes: [l__mod___albert_embeddings_layer_norm], Original ATen: [aten.embedding_dense_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___albert_embeddings_layer_norm => convert_element_type_1
triton_per_fused_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_per_fused_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_24', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*bf16', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 128.0
    tmp21 = tmp13 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp14 * tmp19
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tl.where(tmp28 < 0, tmp28 + 2, tmp28)
    tmp30 = tl.full([1, 1], -1, tl.int64)
    tmp31 = tmp28 == tmp30
    tmp32 = tmp27.to(tl.float32)
    tmp33 = 0.0
    tmp34 = tl.where(tmp31, tmp33, tmp32)
    tmp36 = tl.where(tmp35 < 0, tmp35 + 30000, tmp35)
    tmp37 = tl.full([1, 1], 0, tl.int64)
    tmp38 = tmp35 == tmp37
    tmp39 = tl.where(tmp38, tmp33, tmp32)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (128*tmp29), [XBLOCK, RBLOCK])), tmp34, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (128*tmp36), [XBLOCK, RBLOCK])), tmp39, rmask)
''')
