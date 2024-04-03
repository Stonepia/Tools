

# Original file: ./drq___60.0/drq___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/m5/cm5a2gqvx6g3t4c7ig2dgh6fp7irusgoc6udru4g63546yezoapb.py
# Source Nodes: [sub0_0, truediv], Original ATen: [aten._to_copy, aten.div]
# sub0_0 => convert_element_type
# truediv => div
triton_poi_fused__to_copy_div_0 = async_compile.triton('triton_poi_fused__to_copy_div_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32, 8192], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_div_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_div_0(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 27
    xnumel = 7056
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + (7056*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 255.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x1 + (7056*y0)), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (27*x1)), tmp3, xmask & ymask)
''')
