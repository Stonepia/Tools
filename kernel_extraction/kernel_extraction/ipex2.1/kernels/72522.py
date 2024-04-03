

# Original file: ./timm_vision_transformer_large___60.0/timm_vision_transformer_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/oi/coimeq2ay6g3bc6ru4nupnjm2f5kgtyh3qpvjfz3siqwsa4zkc7m.py
# Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.clone, aten.mul]
# scaled_dot_product_attention => clone_1, mul_2
triton_poi_fused_clone_mul_4 = async_compile.triton('triton_poi_fused_clone_mul_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11579392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 88
    x1 = (xindex // 88) % 257
    x2 = (xindex // 22616) % 16
    x3 = (xindex // 361856)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (88*x2) + (4224*x1) + (1085568*x3)), None).to(tl.float32)
    tmp1 = 0.3264971028628052
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')
