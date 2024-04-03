

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/35/c35tokvtztcvkocxi5juw4nv6kb27nylxyekd5xn7b4ra5cg6rvk.py
# Source Nodes: [mul_16, neg_3, truediv_11], Original ATen: [aten.div, aten.mul, aten.neg]
# mul_16 => mul_19
# neg_3 => neg_3
# truediv_11 => div_8
triton_poi_fused_div_mul_neg_3 = async_compile.triton('triton_poi_fused_div_mul_neg_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_neg_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_div_mul_neg_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (410 + x0 + (204*x1)), xmask)
    tmp8 = tl.load(in_ptr1 + (1 + (26*x2)), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (26*x2), xmask)
    tmp25 = tl.load(in_ptr3 + (25 + (26*x2)), xmask).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (26*x2), xmask).to(tl.float32)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp1 >= tmp2
    tmp6 = tmp4 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp1 == tmp2
    tmp11 = tmp4 & tmp10
    tmp12 = tmp11 == 0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp9 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 / tmp16
    tmp18 = -tmp17
    tmp19 = tmp3 >= tmp2
    tmp20 = tmp4 & tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = tl.full([1], 25, tl.int32)
    tmp24 = tmp22 == tmp23
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tmp21 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp18 * tmp29
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''')
