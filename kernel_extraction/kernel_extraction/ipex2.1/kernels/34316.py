

# Original file: ./pit_b_224___60.0/pit_b_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/um/cumqzazf7njyklam73pvsb2ryb34uynnl6l43henqj6zqufn6zkh.py
# Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten.clone]
# scaled_dot_product_attention_9 => clone_66
triton_poi_fused_clone_35 = async_compile.triton('triton_poi_fused_clone_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4259840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 65
    x2 = (xindex // 4160) % 16
    x3 = (xindex // 66560)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (64*x2) + (3072*x1) + (199680*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
