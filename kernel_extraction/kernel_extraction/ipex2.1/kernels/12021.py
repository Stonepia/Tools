

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/on/conml2ktrczvaxuu3nbhwi72bscpyrilf4mqgcmp6rojtfgvjjz2.py
# Source Nodes: [iadd_51, mul_101, mul_98], Original ATen: [aten.add, aten.mul, aten.select_scatter, aten.slice_scatter]
# iadd_51 => add_61, select_scatter_135, slice_scatter_37, slice_scatter_38
# mul_101 => mul_104
# mul_98 => mul_101
triton_poi_fused_add_mul_select_scatter_slice_scatter_65 = async_compile.triton('triton_poi_fused_add_mul_select_scatter_slice_scatter_65', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_select_scatter_slice_scatter_65', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_select_scatter_slice_scatter_65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3182400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 78) % 204
    x0 = xindex % 3
    x3 = (xindex // 15912)
    x4 = (xindex // 3) % 5304
    x6 = xindex
    tmp40 = tl.load(in_ptr3 + (31824 + x6), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + ((-52) + x4 + (5200*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = 2 + x3
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp10 < tmp3
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp5
    tmp15 = tl.load(in_ptr1 + (x6), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tmp5 & tmp14
    tmp18 = tl.load(in_ptr2 + ((-52) + x4 + (5200*x3)), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (31824 + x6), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp8, tmp18, tmp19)
    tmp21 = tl.where(tmp17, tmp20, 0.0)
    tmp22 = tl.load(in_ptr3 + (31824 + x6), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, 0.0)
    tmp25 = tl.load(in_ptr3 + (31824 + x6), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp26 = tl.where(tmp13, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp16, tmp26)
    tmp28 = tl.where(tmp8, tmp9, tmp27)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp30 = tl.load(in_ptr1 + (x6), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp13, tmp30, 0.0)
    tmp32 = tmp5 & tmp13
    tmp33 = tl.load(in_ptr2 + ((-52) + x4 + (5200*x3)), tmp32 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (31824 + x6), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp8, tmp33, tmp34)
    tmp36 = tl.where(tmp32, tmp35, 0.0)
    tmp37 = tl.load(in_ptr3 + (31824 + x6), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.where(tmp5, tmp36, tmp37)
    tmp39 = tl.where(tmp13, tmp38, 0.0)
    tmp41 = tl.where(tmp13, tmp39, tmp40)
    tmp42 = tl.where(tmp13, tmp31, tmp41)
    tmp43 = tl.where(tmp5, tmp29, tmp42)
    tl.store(out_ptr0 + (x6), tmp43, xmask)
''')
