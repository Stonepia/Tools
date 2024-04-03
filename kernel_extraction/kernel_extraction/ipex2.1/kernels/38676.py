

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/li/clim24cvv3czmrdmwh5tmo7cw7bqkpq7ui2fwp3u6u2is55zdafg.py
# Source Nodes: [avg_pool2d_1], Original ATen: [aten.avg_pool2d]
# avg_pool2d_1 => avg_pool2d_1
triton_poi_fused_avg_pool2d_3 = async_compile.triton('triton_poi_fused_avg_pool2d_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2973696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 88
    x2 = (xindex // 5632)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (22528*x2)), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (22528*x2)), None)
    tmp3 = tl.load(in_ptr0 + (11264 + x0 + (128*x1) + (22528*x2)), None)
    tmp5 = tl.load(in_ptr0 + (11328 + x0 + (128*x1) + (22528*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')
