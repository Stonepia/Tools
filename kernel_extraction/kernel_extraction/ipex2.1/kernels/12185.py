

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ei/ceiavfkzwd3xrnrwqknm64guw6vt2po36o42feauagpzjyswcj4s.py
# Source Nodes: [mul_65, setitem_58, setitem_59, sub_1, truediv_36, truediv_37], Original ATen: [aten.copy, aten.div, aten.mul, aten.select_scatter, aten.sub]
# mul_65 => mul_68
# setitem_58 => copy_58, select_scatter_104
# setitem_59 => copy_59, select_scatter_105
# sub_1 => sub_1
# truediv_36 => div_33
# truediv_37 => div_34
triton_poi_fused_copy_div_mul_select_scatter_sub_54 = async_compile.triton('triton_poi_fused_copy_div_mul_select_scatter_sub_54', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_select_scatter_sub_54', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_select_scatter_sub_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (25 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (25 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 24, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 25, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tmp4 == tmp4
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.where(tmp7, tmp10, tmp10)
    tmp12 = tmp6 / tmp11
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.where(tmp2, tmp3, tmp14)
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''')
