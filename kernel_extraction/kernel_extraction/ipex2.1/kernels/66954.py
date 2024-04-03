

# Original file: ./LayoutLMForMaskedLM__0_backward_135.1/LayoutLMForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/6q/c6qkxo5sh3omsho3kwzrpijg3xs4gn7qrgg2iin4bjlbnfevqrc2.py
# Source Nodes: [l__mod___layoutlm_embeddings_layer_norm], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___layoutlm_embeddings_layer_norm => convert_element_type_1
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i64', 10: '*i64', 11: '*i64', 12: '*i64', 13: '*i64', 14: '*i64', 15: '*fp32', 16: '*fp16', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: 'i32', 25: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr10', 'out_ptr4', 'out_ptr5', 'out_ptr6', 'out_ptr7', 'out_ptr8', 'out_ptr9'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask)
    tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr10 + (4*x0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (4*x0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (4*x0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr13 + (4*x0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
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
    tmp40 = tl.where(tmp39 < 0, tmp39 + 2, tmp39)
    tmp41 = tmp38.to(tl.float32)
    tmp42 = tl.full([1], False, tl.int1)
    tmp43 = 0.0
    tmp44 = tl.where(tmp42, tmp43, tmp41)
    tmp45 = tl.where(tmp39 < 0, tmp39 + 1024, tmp39)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 1024, tmp46)
    tmp49 = tl.where(tmp48 < 0, tmp48 + 1024, tmp48)
    tmp51 = tl.where(tmp50 < 0, tmp50 + 1024, tmp50)
    tmp53 = tl.where(tmp52 < 0, tmp52 + 1024, tmp52)
    tmp55 = tl.where(tmp54 < 0, tmp54 + 30522, tmp54)
    tmp56 = tl.full([1], 0, tl.int64)
    tmp57 = tmp54 == tmp56
    tmp58 = tl.where(tmp57, tmp43, tmp41)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp38, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp40), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r1 + (768*tmp45), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr6 + (tl.broadcast_to(r1 + (768*tmp47), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr7 + (tl.broadcast_to(r1 + (768*tmp49), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr8 + (tl.broadcast_to(r1 + (768*tmp51), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr9 + (tl.broadcast_to(r1 + (768*tmp53), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr10 + (tl.broadcast_to(r1 + (768*tmp55), [RBLOCK])), tmp58, rmask)
''')
