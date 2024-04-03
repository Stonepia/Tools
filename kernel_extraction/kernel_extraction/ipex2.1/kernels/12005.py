

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/us/cusslau3rn3hyyms4itmdygqsmoprsfiwsxea4xl6ivenfrnwwan.py
# Source Nodes: [setitem_62, setitem_63, setitem_64, setitem_65, truediv_43], Original ATen: [aten.copy, aten.div, aten.select_scatter]
# setitem_62 => copy_62, select_scatter_108
# setitem_63 => copy_63, select_scatter_109
# setitem_64 => copy_64, select_scatter_110
# setitem_65 => copy_65, select_scatter_111
# truediv_43 => div_40
triton_poi_fused_copy_div_select_scatter_49 = async_compile.triton('triton_poi_fused_copy_div_select_scatter_49', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_select_scatter_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_select_scatter_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 18, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 19, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 20, tl.int32)
    tmp9 = tmp0 == tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 21, tl.int32)
    tmp13 = tmp0 == tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp11, tmp17)
    tmp19 = tl.where(tmp5, tmp7, tmp18)
    tmp20 = tl.where(tmp2, tmp3, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''')
