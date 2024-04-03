

# Original file: ./RobertaForQuestionAnswering__0_backward_135.1/RobertaForQuestionAnswering__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/n3/cn3saqfem5sk6efdbfqils6oytqtim63lzl7nnmg7xu624ahhtqt.py
# Source Nodes: [l__mod___roberta_embeddings_layer_norm], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___roberta_embeddings_layer_norm => convert_element_type_4
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*i1', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*i64', 11: '*i64', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr4', 'out_ptr5', 'out_ptr6'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
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
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask)
    tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp15 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = 768.0
    tmp32 = tmp24 / tmp31
    tmp33 = tmp15 * tmp31
    tmp34 = tmp33 - tmp19
    tmp35 = tmp25 * tmp30
    tmp36 = tmp34 - tmp35
    tmp37 = tmp32 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tl.where(tmp40 < 0, tmp40 + 512, tmp40)
    tmp42 = tl.full([1], 0, tl.int64)
    tmp43 = tmp40 == tmp42
    tmp44 = 0.0
    tmp45 = tl.where(tmp43, tmp44, tmp39)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 2, tmp46)
    tmp48 = tl.full([1], -1, tl.int64)
    tmp49 = tmp46 == tmp48
    tmp50 = tl.where(tmp49, tmp44, tmp39)
    tmp52 = tl.where(tmp51 < 0, tmp51 + 50265, tmp51)
    tmp53 = tmp51 == tmp42
    tmp54 = tl.where(tmp53, tmp44, tmp39)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp41), [RBLOCK])), tmp45, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r1 + (768*tmp47), [RBLOCK])), tmp50, rmask)
    tl.atomic_add(out_ptr6 + (tl.broadcast_to(r1 + (768*tmp52), [RBLOCK])), tmp54, rmask)
''')
