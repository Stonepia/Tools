

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ti/ctibokfgh4kkm6uqjkquk2f7h2fma6jg5rqbyjqr47hzx5xo2z72.py
# Source Nodes: [gelu_16, mul__19, pad_3], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_16 => add_39, convert_element_type_85, convert_element_type_86, erf_16, mul_140, mul_141, mul_142
# mul__19 => mul_143
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_gelu_mul_14 = async_compile.triton('triton_poi_fused_constant_pad_nd_gelu_mul_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_gelu_mul_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19200) % 25
    x1 = (xindex // 768) % 25
    x3 = (xindex // 480000)
    x4 = xindex % 19200
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (18432*x2) + (442368*x3)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = 0.7071067811865476
    tmp11 = tmp7 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp9 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.where(tmp5, tmp18, 0.0)
    tl.store(out_ptr0 + (x5), tmp19, None)
''')
