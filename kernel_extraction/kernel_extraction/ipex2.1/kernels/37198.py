

# Original file: ./DALLE2_pytorch__37_inference_77.17/DALLE2_pytorch__37_inference_77.17.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2k/c2kgviaubtsc3oqsumwkdvy5i7j3imed5ulgwsbcrwuiap3jdyce.py
# Source Nodes: [add, mul, mul_1], Original ATen: [aten.add, aten.mul]
# add => add
# mul => mul
# mul_1 => mul_1
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0 + (133120*x1)), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 1, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 1)) | ~xmask, "index out of bounds: 0 <= tmp1 < 1")
    tmp3 = 1.0
    tmp4 = tmp3 * tmp2
    tmp6 = 0.0
    tmp7 = tmp6 * tmp5
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''')
