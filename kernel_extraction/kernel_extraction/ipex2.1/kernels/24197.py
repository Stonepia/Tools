

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/zx/czxbl6fgzcwgmdzyoscbfpnn2tc7sy7ibnvb57j6xscnka2h7r2r.py
# Source Nodes: [reshape_30], Original ATen: [aten.clone]
# reshape_30 => clone_52
triton_poi_fused_clone_81 = async_compile.triton('triton_poi_fused_clone_81', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_81', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_81(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 50
    x3 = (xindex // 25600)
    x4 = xindex % 512
    x5 = (xindex // 512)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (3200*x1) + (25600*x3)), None).to(tl.float32)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (1536*x5)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x4 + (512*(((-1) + x2) % 49)) + (25088*x3)), tmp5, other=0.0).to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (x6), tmp10, None)
''')
