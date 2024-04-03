

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/4z/c4zhqxxc7oxdbkzjchzcjc4cz6xdwuwcptwdvqvjivleppgpbqrz.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.add, aten.clone, aten.copy, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_32 = async_compile.triton('triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_32', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_add_clone_copy_native_dropout_backward_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask)
    tmp19 = tl.load(in_ptr4 + (r1 + (128*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (r1 + (128*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr7 + (r1 + (128*x0)), rmask)
    tmp29 = tl.load(in_ptr8 + (r1 + (128*x0)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp3
    tmp18 = tmp14 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp19 * tmp21
    tmp23 = tmp20 - tmp22
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp24 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tmp23 + tmp33
    tmp35 = tmp34 + tmp13
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp13, rmask)
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp35, rmask)
''')
