

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/m2/cm2fxwpx6qaukal5dvx2wjs4zr6yzi46yep6kw46rkyc5ult6icu.py
# Source Nodes: [scaled_dot_product_attention_2], Original ATen: [aten.clone, aten.mul]
# scaled_dot_product_attention_2 => clone_20, mul_104
triton_poi_fused_clone_mul_27 = async_compile.triton('triton_poi_fused_clone_mul_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 64
    x2 = (xindex // 3072) % 4
    x3 = (xindex // 12288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (576*x1) + (36864*x3)), None).to(tl.float32)
    tmp1 = 0.3799178428257963
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')
