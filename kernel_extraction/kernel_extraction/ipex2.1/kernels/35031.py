

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/nj/cnjla4dnykrimm5o6c2syzp6rhsoiyjznybw4oiyyyu6kdwaz5va.py
# Source Nodes: [iadd_15], Original ATen: [aten._to_copy, aten.slice_scatter]
# iadd_15 => convert_element_type_15, slice_scatter_180, slice_scatter_181
triton_poi_fused__to_copy_slice_scatter_48 = async_compile.triton('triton_poi_fused__to_copy_slice_scatter_48', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_scatter_48', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_slice_scatter_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp20 = tl.load(in_ptr2 + (x4), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp17, tmp20, 0.0)
    tmp22 = tmp5 & tmp17
    tmp23 = tl.load(in_ptr3 + ((-52) + x3 + (5200*x2)), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp22, tmp23, 0.0)
    tmp25 = 0.0
    tmp26 = tl.where(tmp5, tmp24, tmp25)
    tmp27 = tl.where(tmp17, tmp26, 0.0)
    tmp28 = tl.where(tmp16, tmp27, tmp25)
    tmp29 = tl.where(tmp16, tmp21, tmp28)
    tmp30 = tl.where(tmp16, tmp19, tmp29)
    tmp31 = tl.where(tmp8, tmp12, tmp30)
    tmp32 = tl.where(tmp5, tmp31, 0.0)
    tmp33 = tl.load(in_ptr1 + (x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp34 = tl.where(tmp16, tmp33, 0.0)
    tmp35 = tl.load(in_ptr2 + (x4), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp16, tmp35, 0.0)
    tmp37 = tmp5 & tmp16
    tmp38 = tl.load(in_ptr3 + ((-52) + x3 + (5200*x2)), tmp37 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tl.where(tmp5, tmp39, tmp25)
    tmp41 = tl.where(tmp16, tmp40, 0.0)
    tmp42 = tl.where(tmp16, tmp41, tmp25)
    tmp43 = tl.where(tmp16, tmp36, tmp42)
    tmp44 = tl.where(tmp16, tmp34, tmp43)
    tmp45 = tl.where(tmp5, tmp32, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, xmask)
''')
