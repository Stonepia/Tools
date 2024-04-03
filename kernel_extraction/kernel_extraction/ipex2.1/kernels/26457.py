

# Original file: ./vision_maskrcnn__25_inference_65.5/vision_maskrcnn__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/lc/clcanoaxgcqtdrmn55nukexfoxyj5ovxywabbkqktkz7fhdcmqni.py
# Source Nodes: [stack_2], Original ATen: [aten.stack]
# stack_2 => cat_2
triton_poi_fused_stack_16 = async_compile.triton('triton_poi_fused_stack_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*i32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_16', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 16*(x0 % 76)
    tl.store(out_ptr0 + (4*x0), tmp0, xmask)
''')
