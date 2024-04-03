

# Original file: ./mobilenetv3_large_100___60.0/mobilenetv3_large_100___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/rh/crhe3326ujgj3d3btdaavjsojv33dooj7ojl6gucng4tn6j7h6ux.py
# Source Nodes: [getattr_getattr_l__self___blocks___6_____0___bn1_act], Original ATen: [aten.hardswish]
# getattr_getattr_l__self___blocks___6_____0___bn1_act => add_129, clamp_max_27, clamp_min_27, convert_element_type_272, div_27, mul_165
triton_poi_fused_hardswish_25 = async_compile.triton('triton_poi_fused_hardswish_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072, 64], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardswish_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 122880
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (960*x2) + (47040*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp9, xmask)
''')
