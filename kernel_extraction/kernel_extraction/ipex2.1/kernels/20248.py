

# Original file: ./cm3leon_generate__26_inference_66.6/cm3leon_generate__26_inference_66.6_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/az/cazuyy3stapgbtwe7ygz7oulf6qjbvtgjj2jgl6vfqvqbq5uj3co.py
# Source Nodes: [cat_63], Original ATen: [aten.cat]
# cat_63 => cat_32
triton_poi_fused_cat_58 = async_compile.triton('triton_poi_fused_cat_58', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_58', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3840
    x1 = (xindex // 3840)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (3936*x1)), tmp0, None)
''')
