

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/dp/cdpuliulemfn33uevvbtc6d3axhybcvpzfryqidbhglaiq3gfjmj.py
# Source Nodes: [add_4, add_5, mul_3, mul_6, setitem_4, truediv_5, truediv_6], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.reciprocal]
# add_4 => add_4
# add_5 => add_5
# mul_3 => mul_4
# mul_6 => mul_8
# setitem_4 => copy_4
# truediv_5 => div_4
# truediv_6 => mul_7, reciprocal_1
triton_poi_fused_add_copy_div_mul_reciprocal_1 = async_compile.triton('triton_poi_fused_add_copy_div_mul_reciprocal_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_reciprocal_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_reciprocal_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (25)).to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (25)).to(tl.float32)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr3 + (10685 + (26*x0) + (5304*x1)), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr4 + (32055 + (78*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp0 = tl.full([1], 24, tl.int64)
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp5 = 1 / tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tl.load(in_ptr1 + (10684 + (26*x0) + (5304*x1)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (10685 + (26*x0) + (5304*x1)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp2, tmp14, 0.0)
    tmp16 = 0.0
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp20 = tmp19 * tmp9
    tmp21 = tmp17 / tmp20
    tmp22 = tmp21 + tmp6
    tmp23 = tmp22.to(tl.float32)
    tmp25 = 1 / tmp24
    tmp26 = 0.7
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = triton_helpers.maximum(tmp16, tmp30)
    tmp32 = tl.sqrt(tmp31)
    tmp33 = tmp28 * tmp32
    tmp34 = tmp23 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp35, xmask)
''')
