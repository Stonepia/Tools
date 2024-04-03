

# Original file: ./dla102___60.0/dla102___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/s4/cs4dokkhczdvzjybz5klacaivcbfwcvienth5i3ohkk4s45ckemq.py
# Source Nodes: [l__self___level2_downsample], Original ATen: [aten.max_pool2d_with_indices]
# l__self___level2_downsample => max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_1 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 56
    x2 = (xindex // 1792)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x1) + (7168*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + (64*x1) + (7168*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (3584 + x0 + (64*x1) + (7168*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (3616 + x0 + (64*x1) + (7168*x2)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')
