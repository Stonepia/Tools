

# Original file: ./LayoutLMForMaskedLM__0_forward_205.0/LayoutLMForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/uc/cucea7an73m2kjx7bu5s25pgrjvyul6a6wxvnk2hdwue7upfcixy.py
# Source Nodes: [], Original ATen: []

triton_per_fused_4 = async_compile.triton('triton_per_fused_4', '''
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
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_4(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel):
    xnumel = 98304
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.load(in_ptr1 + load_seed_offset)
    tmp12 = r1 + (512*x0)
    tmp13 = tl.rand(tmp11, (tmp12).to(tl.uint32))
    tmp14 = 0.1
    tmp15 = tmp13 > tmp14
    tmp16 = tmp6 / tmp10
    tmp17 = tmp15.to(tl.float32)
    tmp18 = tmp17 * tmp16
    tmp19 = 1.1111111111111112
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp15, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp16, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp20, rmask)
''')
