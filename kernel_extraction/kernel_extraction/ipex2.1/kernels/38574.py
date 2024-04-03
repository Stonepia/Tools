

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/da/cdaxnjav3m2qtw2yzbg3fugsouxtqko5ebmcvgt3m6mykga2qyx3.py
# Source Nodes: [add, add_1, mul_3, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
# add => add_36
# add_1 => add_37
# mul_3 => mul_78
# mul_4 => mul_79
# mul_5 => mul_80
# mul_6 => mul_81
triton_poi_fused_add_mul_29 = async_compile.triton('triton_poi_fused_add_mul_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576, 2], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_29(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 743424
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 123904)
    x2 = xindex
    y3 = yindex
    y0 = yindex % 123904
    tmp0 = tl.load(in_ptr0 + (y1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (x2 + (4*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr1 + (2 + x2 + (4*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.0
    tmp21 = tmp19 > tmp20
    tmp22 = 0.1
    tmp23 = tmp19 * tmp22
    tmp24 = tl.where(tmp21, tmp19, tmp23)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp17 * tmp26
    tmp28 = -tmp16
    tmp29 = tmp28 * tmp14
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 > tmp20
    tmp33 = tmp31 * tmp22
    tmp34 = tl.where(tmp32, tmp31, tmp33)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp29 * tmp36
    tmp38 = tmp27 + tmp37
    tmp39 = tmp29 * tmp26
    tmp40 = tmp14 * tmp14
    tmp41 = tmp40 * tmp36
    tmp42 = tmp39 + tmp41
    tl.store(out_ptr0 + (x2 + (2*y3)), tmp27, xmask)
    tl.store(out_ptr1 + (x2 + (2*y3)), tmp37, xmask)
    tl.store(out_ptr2 + (y0 + (123904*x2) + (2478080*y1)), tmp38, xmask)
    tl.store(out_ptr3 + (x2 + (2*y3)), tmp39, xmask)
    tl.store(out_ptr4 + (x2 + (2*y3)), tmp41, xmask)
    tl.store(out_ptr5 + (y0 + (123904*x2) + (2478080*y1)), tmp42, xmask)
''')
