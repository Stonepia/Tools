

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/fi/cfidod4fq7yeml4d4s3qzcmogx4jix2jlaxvgubgenp2q74dxdf6.py
# Source Nodes: [cat_21], Original ATen: [aten.cat]
# cat_21 => cat_6
triton_poi_fused_cat_51 = async_compile.triton('triton_poi_fused_cat_51', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (50432*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (50432*x1)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (256 + x0 + (50432*x1)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0 + (50432*x1)), tmp4, None)
''')
