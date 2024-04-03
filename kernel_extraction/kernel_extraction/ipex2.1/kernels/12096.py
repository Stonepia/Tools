

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7c/c7cwpcuw5wanjcqtk5hx273gh7xtsbl5uvsfz2ytc2ddk67uizfi.py
# Source Nodes: [mul_66, mul_67, mul_68, setitem_60, setitem_61, setitem_62, setitem_63, sub_2, sub_3, sub_4, truediv_38, truediv_39, truediv_40], Original ATen: [aten.copy, aten.div, aten.mul, aten.select_scatter, aten.sub]
# mul_66 => mul_69
# mul_67 => mul_70
# mul_68 => mul_71
# setitem_60 => copy_60, select_scatter_106
# setitem_61 => copy_61, select_scatter_107
# setitem_62 => copy_62, select_scatter_108
# setitem_63 => copy_63, select_scatter_109
# sub_2 => sub_2
# sub_3 => sub_3
# sub_4 => sub_4
# truediv_38 => div_35
# truediv_39 => div_36
# truediv_40 => div_37
triton_poi_fused_copy_div_mul_select_scatter_sub_56 = async_compile.triton('triton_poi_fused_copy_div_mul_select_scatter_sub_56', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_select_scatter_sub_56', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_select_scatter_sub_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.full([1], 20, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 21, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tl.full([1], 22, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp10 = tl.full([1], 23, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp8, tmp9, tmp14)
    tmp16 = tl.where(tmp5, tmp6, tmp15)
    tmp17 = tl.where(tmp2, tmp3, tmp16)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''')
