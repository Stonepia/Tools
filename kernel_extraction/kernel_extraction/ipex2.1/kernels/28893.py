

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/7c/c7c3xu2onr2rf7f5bz5j3c3rwrggwtkt2szp735cwdbzqbvwtmsh.py
# Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.clone, aten.mul]
# scaled_dot_product_attention => clone_2, mul_65
triton_poi_fused_clone_mul_9 = async_compile.triton('triton_poi_fused_clone_mul_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9437184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 36
    x1 = (xindex // 36) % 256
    x2 = (xindex // 9216) % 4
    x3 = (xindex // 36864)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (36*x2) + (432*x1) + (110592*x3)), None).to(tl.float32)
    tmp1 = 0.408248290463863
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')
