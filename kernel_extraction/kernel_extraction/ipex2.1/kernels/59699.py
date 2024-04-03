

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/4c/c4clhdqrkyuytgjhqv2ficgnp7ggxu7ogwp3wbdchx4yfxfol476.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.add, aten.clone, aten.copy, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_18 = async_compile.triton('triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_18', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask)
    tmp20 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp3
    tmp19 = tmp15 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp20 * tmp22
    tmp24 = tmp21 - tmp23
    tmp25 = tmp14 + tmp24
    tmp26 = tmp25 + tmp13
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp13, rmask)
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp26, rmask)
''')
