

# Original file: ./res2net101_26w_4s___60.0/res2net101_26w_4s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/2v/c2v5n47vdvzxlyyla4qtbizduet7c44gh2kg2d4ztzkpqrx6gwat.py
# Source Nodes: [cat_61], Original ATen: [aten.cat]
# cat_61 => cat_4
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp0 = tl.load(in_ptr0 + (156 + x0 + (208*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (208*x1)), tmp0, None)
''')