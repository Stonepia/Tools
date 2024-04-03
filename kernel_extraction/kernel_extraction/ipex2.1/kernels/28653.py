

# Original file: ./dm_nfnet_f0___60.0/dm_nfnet_f0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/2b/c2blwj3lzjzn3ewfuaokmov32gjpsijflas5bjybcvgrudhejc7z.py
# Source Nodes: [gelu_40, mul__49, pad_4], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_40 => add_94, convert_element_type_207, convert_element_type_208, erf_40, mul_341, mul_342, mul_343
# mul__49 => mul_344
# pad_4 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_gelu_mul_25 = async_compile.triton('triton_poi_fused_constant_pad_nd_gelu_mul_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_gelu_mul_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28409856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 13056) % 17
    x1 = (xindex // 768) % 17
    x3 = (xindex // 221952)
    x4 = xindex % 13056
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (12288*x2) + (196608*x3)), tmp5, other=0.0).to(tl.float32)
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
