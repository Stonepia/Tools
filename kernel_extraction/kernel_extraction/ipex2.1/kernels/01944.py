

# Original file: ./mobilenet_v3_large___60.0/mobilenet_v3_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/dq/cdqtugqypjqcknrenoqdkdpp2qvwaoiimxlecys5gub2ybemrzkl.py
# Source Nodes: [getattr_getattr_l__mod___features___13___block___2___scale_activation, mul_5], Original ATen: [aten.hardsigmoid, aten.mul]
# getattr_getattr_l__mod___features___13___block___2___scale_activation => add_104, clamp_max_20, clamp_min_20, convert_element_type_154, convert_element_type_155, div_20
# mul_5 => mul_134
triton_poi_fused_hardsigmoid_mul_18 = async_compile.triton('triton_poi_fused_hardsigmoid_mul_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_mul_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_mul_18(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 21504
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 3.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp11, xmask)
''')
