

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/jo/cjoas53fvg34esj6k5my3sazvgecd5g7noubqg75ssiq2y6b7kzg.py
# Source Nodes: [leaky_relu_18], Original ATen: [aten.leaky_relu]
# leaky_relu_18 => convert_element_type_107, convert_element_type_108, gt_18, mul_58, where_18
triton_poi_fused_leaky_relu_23 = async_compile.triton('triton_poi_fused_leaky_relu_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512, 32768], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_leaky_relu_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 30976
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (1982464*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp4 = 0.1
    tmp5 = tmp1 * tmp4
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (30976*y0) + (3964928*y1)), tmp7, xmask & ymask)
''')
