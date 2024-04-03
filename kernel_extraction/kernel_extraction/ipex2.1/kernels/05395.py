

# Original file: ./llama___60.0/llama___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/yy/cyyzd6halysjvfgiykwjucqyah2gwkpp5rv2y6mtigazq3nmw6nn.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_2
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

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 540672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 512) % 33
    x3 = (xindex // 16896)
    x4 = xindex % 16896
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    tmp8 = tl.load(in_ptr1 + (x4 + (524288*x3)), None).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-512) + x4 + (16384*x3)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.store(out_ptr0 + (x0 + (64*x2) + (2112*x1) + (16896*x3)), tmp9, None)
''')
