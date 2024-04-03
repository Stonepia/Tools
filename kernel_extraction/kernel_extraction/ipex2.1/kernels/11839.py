

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/pj/cpjudz4s4pmj22ay3kuq5msstw6x5y7ufjmlqhccpitjqkgksqmp.py
# Source Nodes: [mul_70, mul_71, mul_72, setitem_64, setitem_65, setitem_66, setitem_67, sub_6, sub_7, sub_8, truediv_42, truediv_43, truediv_44], Original ATen: [aten.copy, aten.div, aten.mul, aten.select_scatter, aten.sub]
# mul_70 => mul_73
# mul_71 => mul_74
# mul_72 => mul_75
# setitem_64 => copy_64, select_scatter_110
# setitem_65 => copy_65, select_scatter_111
# setitem_66 => copy_66, select_scatter_112
# setitem_67 => copy_67, select_scatter_113
# sub_6 => sub_6
# sub_7 => sub_7
# sub_8 => sub_8
# truediv_42 => div_39
# truediv_43 => div_40
# truediv_44 => div_41
triton_poi_fused_copy_div_mul_select_scatter_sub_57 = async_compile.triton('triton_poi_fused_copy_div_mul_select_scatter_sub_57', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_select_scatter_sub_57', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_select_scatter_sub_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 17, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tl.full([1], 18, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp10 = tl.full([1], 19, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp8, tmp9, tmp14)
    tmp16 = tl.where(tmp5, tmp6, tmp15)
    tmp17 = tl.where(tmp2, tmp3, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')
