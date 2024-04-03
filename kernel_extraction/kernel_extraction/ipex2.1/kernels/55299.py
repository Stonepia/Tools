

# Original file: ./YituTechConvBert__0_forward_241.0/YituTechConvBert__0_forward_241.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/7f/c7fkqmi2vm2ntpwgfegqjzsfl5u4k57xncbz5hpwgwl4egdbay44.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128, 262144], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr1, load_seed_offset, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 6
    y3 = (yindex // 6)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x1 + (262144*y0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr1 + (y2 + (6*x1) + (1572864*y3)), tmp4, ymask)
''')
