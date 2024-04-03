

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2v/c2vayu57kbt5l3ilrkphslxrxyx3fq27jiu5wlcfo76diwt3tcij.py
# Source Nodes: [add, add_1, cat_17, mul_3, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.cat, aten.mul]
# add => add_36
# add_1 => add_37
# cat_17 => cat_8
# mul_3 => mul_78
# mul_4 => mul_79
# mul_5 => mul_80
# mul_6 => mul_81
triton_poi_fused_add_cat_mul_18 = async_compile.triton('triton_poi_fused_add_cat_mul_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_mul_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1486848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 247808)
    x0 = xindex % 2
    x4 = (xindex // 2)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (x0 + (4*x4)), None)
    tmp22 = tl.load(in_ptr1 + (2 + x0 + (4*x4)), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 7, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 7), "index out of bounds: 0 <= tmp1 < 7")
    tmp2 = tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 3.5
    tmp5 = tmp3 < tmp4
    tmp6 = 0.125
    tmp7 = tmp3 * tmp6
    tmp8 = tmp7 + tmp6
    tmp9 = 6 + ((-1)*tmp1)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp6
    tmp12 = 0.875
    tmp13 = tmp12 - tmp11
    tmp14 = tl.where(tmp5, tmp8, tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp16 * tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = -tmp16
    tmp21 = tmp20 * tmp14
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 + tmp23
    tmp25 = tmp21 * tmp18
    tmp26 = tmp14 * tmp14
    tmp27 = tmp26 * tmp22
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr0 + (x3), tmp24, None)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x0 + (20*x4)), tmp18, None)
    tl.store(out_ptr3 + (x0 + (20*x4)), tmp22, None)
    tl.store(out_ptr4 + (x0 + (20*x4)), tmp28, None)
    tl.store(out_ptr5 + (x0 + (20*x4)), tmp24, None)
''')
