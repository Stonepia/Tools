

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/rk/crkzvl6ncha3s27ilpnd7b567exsynx4xk2tressrvut3x6raiyr.py
# Source Nodes: [add_1, add_2, add_3, mul_4, truediv_3, truediv_4], Original ATen: [aten.add, aten.div, aten.mul]
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# mul_4 => mul_5
# truediv_3 => div_2
# truediv_4 => div_3
triton_poi_fused_add_div_mul_0 = async_compile.triton('triton_poi_fused_add_div_mul_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24) % 200
    x2 = (xindex // 4800)
    x4 = xindex
    tmp31 = tl.load(in_ptr2 + (1 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp35 = tl.load(in_ptr3 + (31983 + (3*x0) + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp41 = tl.load(in_ptr4 + (10661 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (2 + x0), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = 1 / tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp5
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tl.load(in_ptr1 + (10661 + x0 + (26*x1) + (5304*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (10662 + x0 + (26*x1) + (5304*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp2, tmp13, 0.0)
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = x0
    tmp18 = tmp17 < tmp1
    tmp19 = tl.load(in_ptr0 + (1 + x0), tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = 1 / tmp19
    tmp21 = tmp20 * tmp5
    tmp22 = tmp21 * tmp5
    tmp23 = tmp22 * tmp8
    tmp24 = tl.load(in_ptr1 + (10660 + x0 + (26*x1) + (5304*x2)), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (10661 + x0 + (26*x1) + (5304*x2)), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = tl.where(tmp18, tmp27, 0.0)
    tmp29 = tl.where(tmp18, tmp28, tmp15)
    tmp30 = tmp16 + tmp29
    tmp32 = tmp30 / tmp31
    tmp33 = tmp32 + tmp5
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = triton_helpers.maximum(tmp15, tmp36)
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 0.7
    tmp40 = tmp38 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 / tmp42
    tmp44 = tmp34 + tmp43
    tl.store(out_ptr0 + (x4), tmp44, xmask)
''')
