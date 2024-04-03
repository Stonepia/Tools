

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/sh/cshywoj2igkhyg2waflosa6esyu63u5ywrgxfzqifimr7tnqrw2p.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.add, aten.clone, aten.copy, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_15 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_15', '''
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
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask)
    tmp22 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp23 - tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp15.to(tl.float32)
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp15, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp30, rmask)
''')