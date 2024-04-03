

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/pl/cpl4degm3dpthcrlbwx2kc3dc4ckbufg6c4vwddvzng4l7d64cky.py
# Source Nodes: [add_4, add_5, getattr_getattr_l__mod___stages___2_____3___act1, mul_44, mul_45, mul_46, mul_52, mul_53, mul_54, mul_55], Original ATen: [aten.add, aten.mul, aten.silu]
# add_4 => add_31
# add_5 => add_36
# getattr_getattr_l__mod___stages___2_____3___act1 => convert_element_type_116, convert_element_type_117, mul_144, sigmoid_33
# mul_44 => mul_121
# mul_45 => mul_122
# mul_46 => mul_123
# mul_52 => mul_141
# mul_53 => mul_142
# mul_54 => mul_143
# mul_55 => mul_145
triton_poi_fused_add_mul_silu_21 = async_compile.triton('triton_poi_fused_add_mul_silu_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63700992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp3
    tmp11 = tmp10 * tmp5
    tmp13 = tmp11 + tmp12
    tmp14 = tmp6 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 0.9449111825230679
    tmp20 = tmp18 * tmp19
    tl.store(in_out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr0 + (x3), tmp20, None)
''')
