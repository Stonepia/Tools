

# Original file: ./vision_maskrcnn__53_inference_93.33/vision_maskrcnn__53_inference_93.33_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/i2/ci2sbfaiub4zq4nlm5ryk63ps55drndy2zktsqhakcyubdmcidje.py
# Source Nodes: [add, add_1, add_2, getitem_77, imul, imul_1, mul, mul_1, mul_2, mul_3, setitem, setitem_1, setitem_2, setitem_3, sub, sub_1, sub_2, sub_3, to, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.mul, aten.select, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub, aten.zeros_like]
# add => add
# add_1 => add_1
# add_2 => add_2
# getitem_77 => select_88
# imul => mul_4
# imul_1 => mul_5
# mul => mul
# mul_1 => mul_1
# mul_2 => mul_2
# mul_3 => mul_3
# setitem => copy, select_scatter, slice_9, slice_scatter
# setitem_1 => copy_1, select_scatter_1, slice_13, slice_scatter_1
# setitem_2 => copy_2, select_scatter_2, slice_17, slice_scatter_2
# setitem_3 => slice_scatter_3
# sub => sub
# sub_1 => sub_1
# sub_2 => sub_2
# sub_3 => sub_3
# to => convert_element_type
# zeros_like => full
triton_poi_fused__to_copy_add_copy_mul_select_select_scatter_slice_slice_scatter_sub_zeros_like_34 = async_compile.triton('triton_poi_fused__to_copy_add_copy_mul_select_select_scatter_slice_slice_scatter_sub_zeros_like_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_mul_select_select_scatter_slice_slice_scatter_sub_zeros_like_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_mul_select_select_scatter_slice_slice_scatter_sub_zeros_like_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''')
