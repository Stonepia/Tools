

# Original file: ./llama___60.0/llama___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/4p/c4py5lpxh5wz7cpy2o7dwvby6jqtf3q2nju5aj4kwgt52raxsuuy.py
# Source Nodes: [setitem_1], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_1 => copy_1, slice_scatter_1
triton_poi_fused_copy_slice_scatter_4 = async_compile.triton('triton_poi_fused_copy_slice_scatter_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr1', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 1024
    x2 = (xindex // 524288)
    x3 = xindex % 524288
    x4 = xindex
    tmp8 = tl.load(in_ptr1 + (x4), None).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-512) + x3 + (16384*x2)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr1 + (x4), tmp9, None)
''')