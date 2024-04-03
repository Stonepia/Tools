

# Original file: ./DALLE2_pytorch__37_inference_77.17/DALLE2_pytorch__37_inference_77.17.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ex/cexgeycabhtsweljfdxacannltl5mbopdbm3cy3o565hwi2dmq3v.py
# Source Nodes: [gather_2, gather_3, reshape_2, reshape_3], Original ATen: [aten.gather, aten.view]
# gather_2 => gather_2
# gather_3 => gather_3
# reshape_2 => view_2
# reshape_3 => view_3
triton_poi_fused_gather_view_1 = async_compile.triton('triton_poi_fused_gather_view_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gather_view_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 1, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 1)) | ~xmask, "index out of bounds: 0 <= tmp1 < 1")
    tmp2 = 0.0
    tmp3 = -46.051700592041016
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
''')
