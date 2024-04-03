

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ks/cksuybt5iwuxrufuvtws4sokz72fjnl7k57zin5oujgiayednrjm.py
# Source Nodes: [cat_307], Original ATen: [aten.cat]
# cat_307 => cat_6
triton_poi_fused_cat_14 = async_compile.triton('triton_poi_fused_cat_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2949120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x3 = (xindex // 245760)
    x4 = (xindex // 64) % 3840
    x5 = xindex % 4096
    x6 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (98304 + x0 + (64*x3) + (768*x4)), None).to(tl.float32)
    tl.store(out_ptr0 + (x5 + (12288*x6)), tmp0, None)
''')
