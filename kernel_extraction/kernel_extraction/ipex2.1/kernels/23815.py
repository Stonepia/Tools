

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/44/c44rmszybxq3tubppig7fyplaeo7on6uzhn7ybk73agxdu57cnwz.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_18
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12861440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (301440*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
