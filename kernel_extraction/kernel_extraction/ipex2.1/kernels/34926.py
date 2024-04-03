

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/zk/czkqf64b3h7q4jch25clgxf7wge7twr6456ifddlua2rfrjyjpue.py
# Source Nodes: [mul_183, mul_184, truediv_41], Original ATen: [aten.div, aten.mul]
# mul_183 => mul_183
# mul_184 => mul_184
# truediv_41 => div_41
triton_poi_fused_div_mul_49 = async_compile.triton('triton_poi_fused_div_mul_49', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_div_mul_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x4 = (xindex // 25)
    x5 = xindex
    tmp26 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr5 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tl.load(in_ptr2 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp5, tmp10, 0.0)
    tmp12 = 2 + x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp5
    tmp17 = tl.load(in_ptr3 + (x0 + (26*x4)), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp19 = 0.0
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp5, tmp20, 0.0)
    tmp22 = tl.where(tmp5, tmp21, tmp19)
    tmp23 = tl.where(tmp5, tmp11, tmp22)
    tmp24 = tl.where(tmp5, tmp9, tmp23)
    tmp25 = tl.where(tmp5, tmp7, tmp24)
    tmp27 = 4.0
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp25 / tmp30
    tl.store(out_ptr0 + (x5), tmp31, xmask)
''')
