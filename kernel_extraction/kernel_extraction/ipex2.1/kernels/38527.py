

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/bp/cbpy5q4z6g6qq3dp57itwthb2pfdmtp6o53dk3cnrp77umtg2heo.py
# Source Nodes: [l__self___vgg16_conv_4_3_4], Original ATen: [aten.max_pool2d_with_indices]
# l__self___vgg16_conv_4_3_4 => max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_46 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11894784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 176
    x2 = (xindex // 11264)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (45056*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (45056*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (22528 + x0 + (128*x1) + (45056*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (22592 + x0 + (128*x1) + (45056*x2)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')