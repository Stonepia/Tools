

# Original file: ./hf_Bart___60.0/hf_Bart___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/oo/coor5bahanz3am3pdi7aagtf3vjf766b4bmsziqran6t5y7j33mt.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => amax_6, convert_element_type_53, convert_element_type_54, div_6, exp_6, sub_20, sum_7
triton_per_fused__softmax_8 = async_compile.triton('triton_per_fused__softmax_8', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_8(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.3895313892515355e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp0 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, float("-inf"))
    tmp13 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp12, 0))
    tmp14 = tmp9 - tmp13
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp15 / tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp21, rmask)
''')
