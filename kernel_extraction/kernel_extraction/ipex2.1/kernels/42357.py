

# Original file: ./DALLE2_pytorch__28_inference_68.8/DALLE2_pytorch__28_inference_68.8.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2l/c2lbvr7ksqh5bzossofypogtyccvpo4uwexhgxyaovk6n37ffrgg.py
# Source Nodes: [mul_2, silu], Original ATen: [aten.mul, aten.silu]
# mul_2 => mul_3
# silu => mul_2, sigmoid
triton_poi_fused_mul_silu_1 = async_compile.triton('triton_poi_fused_mul_silu_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_silu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1064960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*x1)), None)
    tmp1 = tl.load(in_ptr0 + (2048 + x0 + (4096*x1)), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')
