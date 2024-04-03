

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6r/c6rxtvsgtydfmzhffzmu4qx6io6py4crgnau6gdc22nlw7ksrtbw.py
# Source Nodes: [iadd_11], Original ATen: [aten._to_copy, aten.slice_scatter]
# iadd_11 => convert_element_type_11, slice_scatter_156, slice_scatter_157
triton_poi_fused__to_copy_slice_scatter_44 = async_compile.triton('triton_poi_fused__to_copy_slice_scatter_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_scatter_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_slice_scatter_44(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x3 = xindex % 5304
    x4 = xindex
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
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr1 + ((-52) + x3 + (5200*x2)), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = 0.0
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tl.where(tmp17, tmp22, 0.0)
    tmp24 = tl.where(tmp16, tmp23, tmp21)
    tmp25 = tl.where(tmp8, tmp12, tmp24)
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tmp27 = tmp5 & tmp16
    tmp28 = tl.load(in_ptr1 + ((-52) + x3 + (5200*x2)), tmp27 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp27, tmp28, 0.0)
    tmp30 = tl.where(tmp5, tmp29, tmp21)
    tmp31 = tl.where(tmp16, tmp30, 0.0)
    tmp32 = tl.where(tmp16, tmp31, tmp21)
    tmp33 = tl.where(tmp5, tmp26, tmp32)
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''')
