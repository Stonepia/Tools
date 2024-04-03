

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/pz/cpzzhkbx2fhmccpfj4baqp2xjanuezacq7lduhukuuj2b5ombqsu.py
# Source Nodes: [l__mod___encoder_b_encoder], Original ATen: [aten.mul]
# l__mod___encoder_b_encoder => mul_102
triton_poi_fused_mul_17 = async_compile.triton('triton_poi_fused_mul_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1261568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 77
    x2 = (xindex // 4928) % 8
    x3 = (xindex // 39424)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x3) + (1536*((x0 + (64*x2)) // 512)) + (49152*x1) + (49152*((x0 + (64*x2) + (512*x3)) // 16384))), None)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')