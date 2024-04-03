

# Original file: ./detectron2_maskrcnn__88_inference_128.68/detectron2_maskrcnn__88_inference_128.68.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/7l/c7lwgalj3ysjtacp4qw74gkkvwqa45sabgblrmzwb5p2joldnztn.py
# Source Nodes: [imul, imul_1, setitem, setitem_1], Original ATen: [aten.copy, aten.mul, aten.slice, aten.slice_scatter]
# imul => mul, slice_4, slice_scatter, slice_scatter_1
# imul_1 => slice_18, slice_scatter_5
# setitem => copy, slice_scatter_2, slice_scatter_3
# setitem_1 => copy_1, slice_scatter_6, slice_scatter_7
triton_poi_fused_copy_mul_slice_slice_scatter_5 = async_compile.triton('triton_poi_fused_copy_mul_slice_slice_scatter_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mul_slice_slice_scatter_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_mul_slice_slice_scatter_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 164
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp9 = tl.load(in_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((-1) + x0) % 2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (1 + (2*(((-1) + x0) // 2)) + (4*x1)), tmp6 & xmask, other=0.0)
    tmp8 = tl.where(tmp6, tmp7, 0.0)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')
