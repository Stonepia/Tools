

# Original file: ./res2net50_14w_8s___60.0/res2net50_14w_8s___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/c5/cc5z6mx7nhzf7zu3ylaspuqohdscgjkiokesqqafx4nom3ktuuql.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28)
    tmp0 = tl.load(in_ptr0 + (196 + x0 + (224*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (224*x1)), tmp0, None)
''')