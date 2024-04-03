

# Original file: ./AllenaiLongformerBase__0_backward_144.6/AllenaiLongformerBase__0_backward_144.6.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/rv/crvdmz5o5b2rt2u4hokksfezu7su6lwzfsbj3jqiljuhqk6jpopf.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_3 = async_compile.triton('triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_3', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4', 'out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 768.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tl.where(tmp25 < 0, tmp25 + 1, tmp25)
    tmp27 = tl.full([1], False, tl.int1)
    tmp28 = 0.0
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tmp31 = tl.where(tmp30 < 0, tmp30 + 4098, tmp30)
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp30 == tmp32
    tmp34 = tl.where(tmp33, tmp28, tmp24)
    tmp36 = tl.where(tmp35 < 0, tmp35 + 50265, tmp35)
    tmp37 = tmp35 == tmp32
    tmp38 = tl.where(tmp37, tmp28, tmp24)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1, [RBLOCK])), tmp29, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp31), [RBLOCK])), tmp34, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r1 + (768*tmp36), [RBLOCK])), tmp38, rmask)
''')
