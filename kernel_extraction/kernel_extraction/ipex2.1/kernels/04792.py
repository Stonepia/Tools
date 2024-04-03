

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/q3/cq3ipuar4omcorkoux73ccuxaoospd4nokgqm62ji7naepkx53xb.py
# Source Nodes: [matmul_4, mul_2], Original ATen: [aten.clone, aten.mul]
# matmul_4 => clone_25
# mul_2 => mul_22
triton_poi_fused_clone_mul_16 = async_compile.triton('triton_poi_fused_clone_mul_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (37632*x3)), None).to(tl.float32)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')