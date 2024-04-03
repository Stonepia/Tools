

# Original file: ./detectron2_maskrcnn__87_inference_127.67/detectron2_maskrcnn__87_inference_127.67.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/yl/cyl6m2cr35bpcn4lgqxtuc4tda5vhdunlo5gxo2lcl5mga2q4a43.py
# Source Nodes: [split_1], Original ATen: [aten.split_with_sizes]
# split_1 => getitem
triton_poi_fused_split_with_sizes_3 = async_compile.triton('triton_poi_fused_split_with_sizes_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_split_with_sizes_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_split_with_sizes_3(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 80, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 80)) | ~xmask, "index out of bounds: 0 <= tmp1 < 80")
    tmp2 = tl.load(in_ptr1 + (tmp1 + (80*x2)), xmask).to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')
