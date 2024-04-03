

# Original file: ./speech_transformer__24_inference_64.4/speech_transformer__24_inference_64.4.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/2o/c2o4ihr3clavy5xuwfvhlrf2h72z3c4ghdjukqwlmtweim45oj2c.py
# Source Nodes: [setitem_4, setitem_5], Original ATen: [aten.copy, aten.select_scatter, aten.slice_scatter]
# setitem_4 => copy_4, select_scatter_4, slice_scatter_3
# setitem_5 => copy_5, select_scatter_5, slice_scatter_4
triton_poi_fused_copy_select_scatter_slice_scatter_8 = async_compile.triton('triton_poi_fused_copy_select_scatter_slice_scatter_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: '*i64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_select_scatter_slice_scatter_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_select_scatter_slice_scatter_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 220
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 22)
    x0 = xindex % 22
    x2 = xindex
    tmp14 = tl.load(in_ptr2 + (88 + x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (110 + x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 5, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 13, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr0 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0)
    tmp8 = tl.full([1], 4, tl.int32)
    tmp9 = tmp1 == tmp8
    tmp10 = tl.full([1], 10, tl.int64)
    tmp11 = tmp3 < tmp10
    tmp12 = tl.load(in_ptr1 + (x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.where(tmp11, tmp12, 0)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tl.where(tmp5, tmp7, tmp17)
    tmp19 = tmp0 == tmp8
    tmp21 = tl.where(tmp19, tmp15, tmp20)
    tmp22 = tl.where(tmp2, tmp18, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''')