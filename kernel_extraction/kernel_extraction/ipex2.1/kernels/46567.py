

# Original file: ./moondream___60.0/moondream___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tc/ctc4mjwpfluhexfgidithimnthu6wjmlwdczszpiwtvyncpxmprq.py
# Source Nodes: [add_3, softmax, to_5, truediv], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div]
# add_3 => add_5
# softmax => amax, div_1, exp, sub_1, sum_1
# to_5 => convert_element_type_5
# truediv => div
triton_per_fused__softmax__to_copy_add_div_6 = async_compile.triton('triton_per_fused__softmax__to_copy_add_div_6', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_div_6(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = r2
    tmp4 = 1 + x0
    tmp5 = tmp3 < tmp4
    tmp6 = 0.0
    tmp7 = -65504.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp2 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, float("-inf"))
    tmp15 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp14, 0))
    tmp16 = tmp11 - tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tmp17 / tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp23, rmask)
''')
