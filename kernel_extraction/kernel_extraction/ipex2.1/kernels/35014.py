

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/c5/cc55ffaht6wrudcftzzzm25ccrtpqkwclhfy3ixxzsaulc3342r4.py
# Source Nodes: [iadd_7, mul_58, setitem_24, setitem_28, truediv_23], Original ATen: [aten._to_copy, aten.copy, aten.div, aten.mul, aten.slice_scatter]
# iadd_7 => convert_element_type_7, slice_scatter_122, slice_scatter_123, slice_scatter_124, slice_scatter_125, slice_scatter_126
# mul_58 => mul_58
# setitem_24 => copy_24, slice_scatter_114, slice_scatter_115, slice_scatter_116, slice_scatter_117, slice_scatter_118
# setitem_28 => copy_28, slice_scatter_135, slice_scatter_136, slice_scatter_137
# truediv_23 => div_23
triton_poi_fused__to_copy_copy_div_mul_slice_scatter_31 = async_compile.triton('triton_poi_fused__to_copy_copy_div_mul_slice_scatter_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_ptr2', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_div_mul_slice_scatter_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_copy_div_mul_slice_scatter_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x1 = (xindex // 26) % 204
    x3 = xindex % 5304
    x4 = xindex
    x0 = xindex % 26
    tmp30 = tl.load(in_out_ptr0 + (x4), xmask).to(tl.float32)
    tmp47 = tl.load(in_ptr2 + (x4), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + ((-10478) + x3 + (5226*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.where(tmp11, tmp13, 0.0)
    tmp15 = tmp5 & tmp5
    tmp16 = tmp10 & tmp15
    tmp17 = tl.load(in_out_ptr0 + (x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp19 = tl.load(in_out_ptr0 + (x4), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp10, tmp18, tmp19)
    tmp21 = tl.where(tmp15, tmp20, 0.0)
    tmp22 = tl.load(in_out_ptr0 + (x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp24 = tl.where(tmp10, tmp14, tmp23)
    tmp25 = tl.where(tmp5, tmp24, 0.0)
    tmp26 = tl.load(in_out_ptr0 + (x4), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp11, tmp26, 0.0)
    tmp28 = tl.where(tmp10, tmp27, tmp22)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp31 = tl.where(tmp5, tmp29, tmp30)
    tmp32 = tl.where(tmp5, tmp25, tmp31)
    tmp33 = tmp5 & tmp11
    tmp34 = tmp10 & tmp33
    tmp35 = tl.where(tmp34, tmp32, 0.0)
    tmp36 = tl.where(tmp10, tmp35, tmp32)
    tmp37 = tl.where(tmp33, tmp36, 0.0)
    tmp38 = tl.where(tmp5, tmp37, tmp32)
    tmp39 = tl.load(in_ptr1 + (x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = 4.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp38 / tmp41
    tmp43 = tl.where(tmp11, tmp42, 0.0)
    tmp44 = tl.load(in_ptr2 + (x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp45 = tl.where(tmp10, tmp43, tmp44)
    tmp46 = tl.where(tmp5, tmp45, 0.0)
    tmp48 = tl.where(tmp5, tmp46, tmp47)
    tl.store(out_ptr0 + (x4), tmp48, xmask)
''')
