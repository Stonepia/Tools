

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/4x/c4xbo6zjdlnj5gvols4rzngjfyfykxfdmesgxzq7ks3jfr2y7uho.py
# Source Nodes: [getitem_4], Original ATen: [aten.slice]
# getitem_4 => slice_78
triton_poi_fused_slice_8 = async_compile.triton('triton_poi_fused_slice_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257)
    x2 = xindex
    tmp0 = (-255) + x0 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''')
