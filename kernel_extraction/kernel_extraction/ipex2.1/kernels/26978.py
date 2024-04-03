

# Original file: ./tf_efficientnet_b0___60.0/tf_efficientnet_b0___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/6c/c6chj275p424g5lb2gxeopstjji3niq4gnr637dznrgf3lfmsjbq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___bn1_act, pad_2], Original ATen: [aten.constant_pad_nd, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___bn1_act => convert_element_type_49, mul_42, sigmoid_12
# pad_2 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_silu_20 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64161792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8496) % 59
    x1 = (xindex // 144) % 59
    x3 = (xindex // 501264)
    x4 = xindex % 8496
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8208) + x4 + (8064*x2) + (451584*x3)), tmp10, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.where(tmp10, tmp14, 0.0)
    tl.store(out_ptr0 + (x6), tmp15, None)
''')
