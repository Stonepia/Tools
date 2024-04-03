

# Original file: ./moondream___60.0/moondream___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/nd/cndnvhsdpgj6noq6dfkn7vvcyqlfxaosv3qazctlk2poenfnx4r7.py
# Source Nodes: [add_3, softmax, to_5, truediv], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div]
# add_3 => add_5
# softmax => amax, div_1, exp, sub_1, sum_1
# to_5 => convert_element_type_15
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
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
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = r2
    tmp5 = 1 + x0
    tmp6 = tmp4 < tmp5
    tmp7 = 0.0
    tmp8 = -3.4028234663852886e+38
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp3 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp13, 0))
    tmp15 = tmp10 - tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp16 / tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp22, rmask)
''')
