

# Original file: ./tacotron2__28_inference_68.8/tacotron2__28_inference_68.8_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/lh/clhwbdnoissxr3uy5dj4a77a4n3xef3iincbyt7ibtnekqxettlu.py
# Source Nodes: [masked_fill__2], Original ATen: [aten.masked_fill]
# masked_fill__2 => full_default_2, where_2
triton_poi_fused_masked_fill_1 = async_compile.triton('triton_poi_fused_masked_fill_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr1', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_masked_fill_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0 == 0
    tmp3 = 1000.0
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')
