

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ee/ceem2mip5cgjfbpxozixnjwetnv6mtefsrrgdgccnba6in24ujw3.py
# Source Nodes: [cat_85], Original ATen: [aten.cat]
# cat_85 => cat_6
triton_poi_fused_cat_24 = async_compile.triton('triton_poi_fused_cat_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (480*x1)), tmp0, xmask)
''')