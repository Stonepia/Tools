

# Original file: ./speech_transformer__24_inference_64.4/speech_transformer__24_inference_64.4_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/v6/cv6rkyvdhzg5o5ayd6st5gxm6iyvobnomi4g5t2lh5fiiy2y2wix.py
# Source Nodes: [fill_, setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.copy, aten.fill, aten.select_scatter, aten.slice_scatter]
# fill_ => full
# setitem => copy, select_scatter, slice_scatter
# setitem_1 => copy_1, select_scatter_1, slice_scatter_1
# setitem_2 => copy_2, select_scatter_2
# setitem_3 => copy_3, select_scatter_3, slice_scatter_2
triton_poi_fused_copy_fill_select_scatter_slice_scatter_5 = async_compile.triton('triton_poi_fused_copy_fill_select_scatter_slice_scatter_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: '*i64', 4: '*i64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_fill_select_scatter_slice_scatter_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_fill_select_scatter_slice_scatter_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 220
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 22)
    x0 = xindex % 22
    x2 = xindex
    tmp10 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 19, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr0 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0)
    tmp8 = tl.full([1], 2, tl.int32)
    tmp9 = tmp1 == tmp8
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp1 == tmp11
    tmp13 = tl.full([1], 16, tl.int64)
    tmp14 = tmp3 < tmp13
    tmp15 = tl.load(in_ptr2 + (x0), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0)
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tmp11 == tmp17
    tmp19 = tl.full([1], 18, tl.int64)
    tmp20 = tmp3 < tmp19
    tmp21 = tl.load(in_ptr3 + (x0), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0)
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tl.where(tmp20, tmp22, tmp23)
    tmp25 = tl.where(tmp18, tmp24, tmp23)
    tmp26 = tl.where(tmp14, tmp16, tmp25)
    tmp27 = tmp1 == tmp17
    tmp28 = tl.where(tmp27, tmp24, tmp23)
    tmp29 = tl.where(tmp12, tmp26, tmp28)
    tmp30 = tl.where(tmp9, tmp10, tmp29)
    tmp31 = tl.where(tmp5, tmp7, tmp30)
    tmp32 = tmp0 == tmp8
    tmp33 = tmp0 == tmp11
    tmp34 = tmp0 == tmp17
    tmp35 = tl.where(tmp34, tmp24, tmp23)
    tmp36 = tl.where(tmp33, tmp26, tmp35)
    tmp37 = tl.where(tmp32, tmp10, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''')
