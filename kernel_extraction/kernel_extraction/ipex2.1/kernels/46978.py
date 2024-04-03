

# Original file: ./volo_d1_224___60.0/volo_d1_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/ix/cix5rwdndzvsgto7enliikrqygztc5tyd6kdumrqdp7hwe35eyhq.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_4
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21676032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = (-1) + (2*(x2 // 14)) + (x1 // 3) + (tl.where(((2*(x2 // 14)) + (x1 // 3)) >= 0, 0, 30))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*(x2 % 14)) + (x1 % 3) + (tl.where(((2*(x2 % 14)) + (x1 % 3)) >= 0, 0, 30))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5568) + x0 + (32*x3) + (192*(x1 % 3)) + (192*(tl.where(((2*(x2 % 14)) + (x1 % 3)) >= 0, 0, 30))) + (384*(x2 % 14)) + (5376*(x1 // 3)) + (5376*(tl.where(((2*(x2 // 14)) + (x1 // 3)) >= 0, 0, 30))) + (10752*(x2 // 14)) + (150528*x4)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.where(tmp10, tmp12, 0.0)
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp14, None)
''')
