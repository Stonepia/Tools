

# Original file: ./DALLE2_pytorch__25_inference_65.5/DALLE2_pytorch__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ur/curizk4hxj6xebu7a7nulb22jju2v3uiqbrky5pwkpuxazu2mooe.py
# Source Nodes: [stack_1], Original ATen: [aten.stack]
# stack_1 => cat_2
triton_poi_fused_stack_7 = async_compile.triton('triton_poi_fused_stack_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (128*x1)), xmask)
    tl.store(out_ptr0 + (2*x2), tmp0, xmask)
''')
