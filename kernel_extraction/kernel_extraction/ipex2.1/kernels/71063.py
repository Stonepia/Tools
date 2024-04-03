

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/r7/cr7lcst6mahdxzqy6ml555muyvalr5tb6cxhqa2wo6p3vlkyytyp.py
# Source Nodes: [add_8, gelu_39, mul_76, mul_77, mul_78, mul_79, mul__47, mul__48], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_8 => add_90
# gelu_39 => add_91, erf_39, mul_330, mul_331, mul_332
# mul_76 => mul_326
# mul_77 => mul_327
# mul_78 => mul_329
# mul_79 => mul_334
# mul__47 => mul_328
# mul__48 => mul_333
triton_poi_fused_add_gelu_mul_24 = async_compile.triton('triton_poi_fused_add_gelu_mul_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_24(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28311552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 221184)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = 0.7071067811865476
    tmp14 = tmp10 * tmp13
    tmp15 = libdevice.erf(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp12 * tmp17
    tmp19 = 1.7015043497085571
    tmp20 = tmp18 * tmp19
    tmp21 = 0.8980265101338745
    tmp22 = tmp20 * tmp21
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')
