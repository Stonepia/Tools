

# Original file: ./fastNLP_Bert__21_inference_61.1/fastNLP_Bert__21_inference_61.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/wc/cwcobtggmw2hcurpihracbzjmn4fqggywbns3lmw7ztddv4c5blj.py
# Source Nodes: [add_2, matmul_1, mul, softmax, sub, to, truediv], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.rsub]
# add_2 => add_4
# matmul_1 => convert_element_type_10
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
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_mul_rsub_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (475*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.0
    tmp7 = tmp6 - tmp5
    tmp8 = -10000.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp13, 0))
    tmp15 = tmp10 - tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp16 / tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (475*x0)), tmp22, rmask & xmask)
''')
