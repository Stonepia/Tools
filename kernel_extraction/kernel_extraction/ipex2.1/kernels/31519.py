

# Original file: ./timm_resnest___60.0/timm_resnest___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jz/cjzldokpou4mpkbpayvtcsywterwvn5oh2oukccdmg5dsmhnivmq.py
# Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___downsample_0 => avg_pool2d_3
triton_poi_fused_avg_pool2d_12 = async_compile.triton('triton_poi_fused_avg_pool2d_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 14
    x2 = (xindex // 7168)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (1024*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14848 + x0 + (1024*x1) + (28672*x2)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')
