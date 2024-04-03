

# Original file: ./pytorch_unet___60.0/pytorch_unet___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ly/cly7kl73r2rvw66ryz7dmj4pyfuzie4qtzt37s6xtpahmwmvz6mb.py
# Source Nodes: [l__mod___down3_maxpool_conv_0], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___down3_maxpool_conv_0 => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2437120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 119
    x2 = (xindex // 30464)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (122368*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (122368*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (61184 + x0 + (512*x1) + (122368*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (61440 + x0 + (512*x1) + (122368*x2)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')
