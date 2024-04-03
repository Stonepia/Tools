

# Original file: ./dm_nfnet_f0___60.0/dm_nfnet_f0___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/hu/chuqqmrqniutjah5rseki4t2jho4m2ueuz5v33jtjoc62q3pke4l.py
# Source Nodes: [add_8, gelu_39, mul_76, mul_77, mul_78, mul_79, mul__47, mul__48], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_8 => add_90
# gelu_39 => add_91, convert_element_type_164, convert_element_type_165, erf_39, mul_330, mul_331, mul_332
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

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_24(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 393216)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = -3.046875
    tmp6 = tmp4 * tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = 0.7071067811865476
    tmp15 = tmp11 * tmp14
    tmp16 = libdevice.erf(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = tmp13 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.7015043497085571
    tmp22 = tmp20 * tmp21
    tmp23 = 0.8980265101338745
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, None)
''')
