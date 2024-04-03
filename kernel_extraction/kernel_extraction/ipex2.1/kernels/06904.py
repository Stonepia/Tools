

# Original file: ./detectron2_maskrcnn_r_101_fpn__61_inference_101.41/detectron2_maskrcnn_r_101_fpn__61_inference_101.41_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/gs/cgs6klweezi7yuyvf4xeqywvmj36vc4wxo4zieevqalfamhfgq5u.py
# Source Nodes: [getitem_1], Original ATen: [aten.index]
# getitem_1 => index
triton_poi_fused_index_1 = async_compile.triton('triton_poi_fused_index_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4)
    x0 = xindex % 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + ks0, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < ks0)) | ~xmask, "index out of bounds: 0 <= tmp1 < ks0")
    tmp2 = tl.load(in_ptr1 + (x0 + (4*tmp1)), xmask)
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''')
