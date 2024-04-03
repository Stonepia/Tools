

# Original file: ./DistilBertForMaskedLM__0_backward_99.1/DistilBertForMaskedLM__0_backward_99.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/2g/c2gsecihwihgfzm375it4wosisgeme2nrmjbyz5xjpc4s3bm5trx.py
# Source Nodes: [add, l__mod___distilbert_embeddings_layer_norm], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# l__mod___distilbert_embeddings_layer_norm => convert_element_type
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp16', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16384
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
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask)
    tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr7 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
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
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = 768.0
    tmp34 = tmp26 / tmp33
    tmp35 = tmp15 * tmp33
    tmp36 = tmp35 - tmp19
    tmp37 = tmp27 * tmp32
    tmp38 = tmp36 - tmp37
    tmp39 = tmp34 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp42 = tl.where(tmp41 < 0, tmp41 + 30522, tmp41)
    tmp43 = tl.full([1], 0, tl.int64)
    tmp44 = tmp41 == tmp43
    tmp45 = tmp40.to(tl.float32)
    tmp46 = 0.0
    tmp47 = tl.where(tmp44, tmp46, tmp45)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp40, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp42), [RBLOCK])), tmp47, rmask)
''')
