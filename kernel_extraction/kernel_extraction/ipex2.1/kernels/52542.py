

# Original file: ./DistilBertForQuestionAnswering__0_backward_99.1/DistilBertForQuestionAnswering__0_backward_99.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ky/ckyyu3ihqrwfdupqpqmdqtapgbi372tuxfhf5uzhfzsz5btuymkb.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused__to_copy_native_dropout_backward_native_layer_norm_backward_6 = async_compile.triton('triton_per_fused__to_copy_native_dropout_backward_native_layer_norm_backward_6', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_dropout_backward_native_layer_norm_backward_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_native_dropout_backward_native_layer_norm_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32768
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
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp4
    tmp30 = tmp26 * tmp29
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp30, rmask)
''')
