

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/23/c23k3fyij4i6tnzplehbj5numah6pm25aaovr3anzpcyjbnp3h6l.py
# Source Nodes: [iadd_5], Original ATen: [aten._to_copy, aten.slice_scatter]
# iadd_5 => convert_element_type_5, slice_scatter_96, slice_scatter_97, slice_scatter_98, slice_scatter_99
triton_poi_fused__to_copy_slice_scatter_27 = async_compile.triton('triton_poi_fused__to_copy_slice_scatter_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_scatter_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_slice_scatter_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x4 = xindex
    tmp42 = tl.load(in_ptr1 + (10608 + x4), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp7 & tmp5
    tmp9 = tl.load(in_ptr0 + ((-26) + x0 + (25*x1) + (5025*x2)), tmp8 & xmask, other=0.0)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.where(tmp8, tmp10, 0.0)
    tmp12 = 2 + x2
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tmp12 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp16 & tmp5
    tmp18 = tmp5 & tmp17
    tmp19 = tmp7 & tmp18
    tmp20 = tl.load(in_ptr1 + (10608 + x4), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp19, tmp20, 0.0)
    tmp22 = tl.load(in_ptr1 + (10608 + x4), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp7, tmp21, tmp22)
    tmp24 = tl.where(tmp18, tmp23, 0.0)
    tmp25 = tl.load(in_ptr1 + (10608 + x4), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp26 = tl.where(tmp5, tmp24, tmp25)
    tmp27 = tl.where(tmp17, tmp26, 0.0)
    tmp28 = tl.load(in_ptr1 + (10608 + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp16, tmp27, tmp28)
    tmp30 = tl.where(tmp7, tmp11, tmp29)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp32 = tmp5 & tmp16
    tmp33 = tmp7 & tmp32
    tmp34 = tl.load(in_ptr1 + (10608 + x4), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tl.load(in_ptr1 + (10608 + x4), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp7, tmp35, tmp36)
    tmp38 = tl.where(tmp32, tmp37, 0.0)
    tmp39 = tl.load(in_ptr1 + (10608 + x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tl.where(tmp16, tmp40, 0.0)
    tmp43 = tl.where(tmp16, tmp41, tmp42)
    tmp44 = tl.where(tmp5, tmp31, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, xmask)
''')
