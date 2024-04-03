

# Original file: ./detectron2_maskrcnn_r_50_fpn__61_inference_101.41/detectron2_maskrcnn_r_50_fpn__61_inference_101.41_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/r5/cr5blixoduq6dn2uzjuaiyx3owaoi3a377a55fnntti5xlvawp33.py
# Source Nodes: [getitem_2], Original ATen: [aten.index]
# getitem_2 => index_1
triton_poi_fused_index_2 = async_compile.triton('triton_poi_fused_index_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 338, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 338)) | ~xmask, "index out of bounds: 0 <= tmp1 < 338")
    tmp2 = tl.load(in_ptr1 + (tmp1), xmask)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')
