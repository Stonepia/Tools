

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6l/c6le57smyp42urkraf7yxivtmk2ecxyzsspkmuwvxmacacjfsy7v.py
# Source Nodes: [iadd_20, iadd_22, setitem_25, setitem_27], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.select_scatter]
# iadd_20 => add_29, convert_element_type_9, select_scatter_42
# iadd_22 => convert_element_type_10, select_scatter_46
# setitem_25 => copy_25, select_scatter_39
# setitem_27 => copy_27, select_scatter_43
triton_poi_fused__to_copy_add_copy_select_scatter_24 = async_compile.triton('triton_poi_fused__to_copy_add_copy_select_scatter_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_select_scatter_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_select_scatter_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (9 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (10 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 11, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full([1], 10, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tmp5 == tmp5
    tmp8 = tl.full([1], 9, tl.int32)
    tmp9 = tmp5 == tmp8
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.where(tmp7, tmp16, tmp12)
    tmp18 = tmp0 == tmp8
    tmp20 = tl.where(tmp18, tmp10, tmp19)
    tmp21 = tl.where(tmp6, tmp16, tmp20)
    tmp22 = tl.where(tmp6, tmp17, tmp21)
    tmp23 = tl.where(tmp2, tmp4, tmp22)
    tl.store(out_ptr0 + (x2), tmp23, xmask)
''')
