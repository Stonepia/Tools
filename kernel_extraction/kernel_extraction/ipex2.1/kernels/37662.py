

# Original file: ./BartForConditionalGeneration___60.0/BartForConditionalGeneration___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/r2/cr2pdchxt4xy2vu23shxdy3wi4htib6pfjzwfezo4a4qszvobejw.py
# Source Nodes: [clone, eq, masked_fill_, new_zeros, setitem, setitem_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.masked_fill, aten.new_zeros, aten.select_scatter, aten.slice, aten.slice_scatter]
# clone => clone
# eq => eq
# masked_fill_ => full_default, where
# new_zeros => full
# setitem => copy, slice_5, slice_scatter, slice_scatter_1
# setitem_1 => copy_1, select_scatter, slice_10, slice_scatter_2
triton_poi_fused_clone_copy_eq_masked_fill_new_zeros_select_scatter_slice_slice_scatter_0 = async_compile.triton('triton_poi_fused_clone_copy_eq_masked_fill_new_zeros_select_scatter_slice_slice_scatter_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_eq_masked_fill_new_zeros_select_scatter_slice_slice_scatter_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_copy_eq_masked_fill_new_zeros_select_scatter_slice_slice_scatter_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tl.load(in_ptr0 + ((-1) + x2), tmp4, other=0.0)
    tmp6 = tl.where(tmp4, tmp5, 0)
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tmp11 = tl.full([1], -100, tl.int64)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.where(tmp12, tmp3, tmp10)
    tl.store(out_ptr0 + (x2), tmp13, None)
''')
