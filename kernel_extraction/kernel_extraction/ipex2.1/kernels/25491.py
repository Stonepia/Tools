

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/bd/cbdoeffsix6vwp5z64xj64243duhiqn5alz7t2jzedqauje5dfgo.py
# Source Nodes: [mul_35], Original ATen: [aten.mul]
# mul_35 => mul_86
triton_poi_fused_mul_15 = async_compile.triton('triton_poi_fused_mul_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3840*x1)), None).to(tl.float32)
    tmp1 = 0.11180339887498948
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')