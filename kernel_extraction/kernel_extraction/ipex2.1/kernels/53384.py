

# Original file: ./resnest101e___60.0/resnest101e___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/y4/cy4n2kuaa2ezyb34bm4nrxeepwn4sisqxl5v4jd3q2kapewoolb4.py
# Source Nodes: [getattr_l__self___layer4___0___downsample_0], Original ATen: [aten.avg_pool2d]
# getattr_l__self___layer4___0___downsample_0 => avg_pool2d_5
triton_poi_fused_avg_pool2d_19 = async_compile.triton('triton_poi_fused_avg_pool2d_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*x1) + (32768*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1024 + x0 + (2048*x1) + (32768*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + (2048*x1) + (32768*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (17408 + x0 + (2048*x1) + (32768*x2)), None).to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')
