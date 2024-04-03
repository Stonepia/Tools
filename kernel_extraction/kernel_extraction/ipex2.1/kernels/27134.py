

# Original file: ./cm3leon_generate__21_inference_61.1/cm3leon_generate__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/br/cbr6j3yootoaiqdumga33ro26cgvy2qcrv2ma7r7oipwyetrcfgf.py
# Source Nodes: [fill_, setitem], Original ATen: [aten.copy, aten.fill, aten.slice, aten.slice_scatter]
# fill_ => full_1
# setitem => copy, slice_3, slice_scatter, slice_scatter_1
triton_poi_fused_copy_fill_slice_slice_scatter_0 = async_compile.triton('triton_poi_fused_copy_fill_slice_slice_scatter_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_fill_slice_slice_scatter_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_fill_slice_slice_scatter_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0), tmp2 & xmask, other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0)
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')
