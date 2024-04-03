

# Original file: ./fastNLP_Bert__21_inference_61.1/fastNLP_Bert__21_inference_61.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gv/cgv5hc3ugew7vpgp4zs37kc3zopvkw3uyq46kl5v2l36l7uadsjx.py
# Source Nodes: [add_2, mul, softmax, sub, to, truediv], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
# add_2 => add_4
# mul => mul
# softmax => amax, div_1, exp, sub_2, sum_1
# sub => sub
# to => convert_element_type
# truediv => div
triton_per_fused__softmax__to_copy_add_div_mul_rsub_2 = async_compile.triton('triton_per_fused__softmax__to_copy_add_div_mul_rsub_2', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_mul_rsub_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_div_mul_rsub_2(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 5700
    XBLOCK: tl.constexpr = 1
    rnumel = 475
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (475*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = -10000.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, float("-inf"))
    tmp13 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp12, 0))
    tmp14 = tmp9 - tmp13
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp15 / tmp19
    tl.store(out_ptr2 + (r1 + (475*x0)), tmp20, rmask & xmask)
''')
