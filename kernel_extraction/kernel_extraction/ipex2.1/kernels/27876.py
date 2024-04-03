

# Original file: ./speech_transformer__24_inference_64.4/speech_transformer__24_inference_64.4_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/4n/c4nlz3oyeq2dyb6z2bwjxvpazjgxzezsx2oyrfhq2l2ki5l6ehsa.py
# Source Nodes: [cat_25, cat_35], Original ATen: [aten.cat]
# cat_25 => cat_14
# cat_35 => cat_4
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''')
