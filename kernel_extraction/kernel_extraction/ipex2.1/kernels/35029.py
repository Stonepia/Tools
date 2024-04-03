

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/mf/cmfhbc4ad4iesdz6oxzli2igyah4rvzrghypxxkngjciplhne2fp.py
# Source Nodes: [iadd_14], Original ATen: [aten._to_copy, aten.slice_scatter]
# iadd_14 => convert_element_type_14, slice_scatter_174, slice_scatter_175
triton_poi_fused__to_copy_slice_scatter_46 = async_compile.triton('triton_poi_fused__to_copy_slice_scatter_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_scatter_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_slice_scatter_46(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x4 = xindex
    x3 = xindex % 5304
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 25, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp9 & xmask, other=0.0)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.where(tmp9, tmp11, 0.0)
    tmp13 = 2 + x2
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp16 & tmp5
    tmp18 = tl.load(in_ptr1 + (x4), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp5 & tmp17
    tmp21 = tl.load(in_ptr2 + ((-52) + x3 + (5200*x2)), tmp20 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = 0.0
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tl.where(tmp17, tmp24, 0.0)
    tmp26 = tl.where(tmp16, tmp25, tmp23)
    tmp27 = tl.where(tmp16, tmp19, tmp26)
    tmp28 = tl.where(tmp8, tmp12, tmp27)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp30 = tl.load(in_ptr1 + (x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp16, tmp30, 0.0)
    tmp32 = tmp5 & tmp16
    tmp33 = tl.load(in_ptr2 + ((-52) + x3 + (5200*x2)), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp34 = tl.where(tmp32, tmp33, 0.0)
    tmp35 = tl.where(tmp5, tmp34, tmp23)
    tmp36 = tl.where(tmp16, tmp35, 0.0)
    tmp37 = tl.where(tmp16, tmp36, tmp23)
    tmp38 = tl.where(tmp16, tmp31, tmp37)
    tmp39 = tl.where(tmp5, tmp29, tmp38)
    tl.store(out_ptr0 + (x4), tmp39, xmask)
''')
