

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3u/c3u756dc4lkxjkbypgronlpadchx2cmgx7wy5kft7ingeqrrytmy.py
# Source Nodes: [iadd_28, mul_42, neg_29, setitem_33, setitem_35], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.mul, aten.neg, aten.select_scatter]
# iadd_28 => add_37, convert_element_type_13, select_scatter_58
# mul_42 => mul_45
# neg_29 => neg_29
# setitem_33 => copy_33, select_scatter_55
# setitem_35 => copy_35, select_scatter_59
triton_poi_fused__to_copy_add_copy_mul_neg_select_scatter_29 = async_compile.triton('triton_poi_fused__to_copy_add_copy_mul_neg_select_scatter_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_mul_neg_select_scatter_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_mul_neg_select_scatter_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp6 = tl.load(in_ptr0 + (13 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (14 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 14, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp4 = tl.full([1], 13, tl.int32)
    tmp5 = tmp1 == tmp4
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = -tmp10
    tmp12 = tmp4 == tmp4
    tmp13 = tl.where(tmp12, tmp6, tmp6)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = tmp9 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.where(tmp3, tmp17, tmp8)
    tmp19 = tmp0 == tmp4
    tmp21 = tl.where(tmp19, tmp6, tmp20)
    tmp22 = tl.where(tmp2, tmp17, tmp21)
    tmp23 = tl.where(tmp2, tmp18, tmp22)
    tl.store(out_ptr0 + (x2), tmp23, xmask)
''')
