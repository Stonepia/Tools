

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/od/codi4hzojgsnnen3zdhx74wovooyxjqmf3inljdhqpt5yomegdkr.py
# Source Nodes: [add_4, mul_17, setitem_7], Original ATen: [aten.add, aten.copy, aten.mul, aten.select_scatter, aten.slice_scatter]
# add_4 => add_4
# mul_17 => mul_17
# setitem_7 => copy_7, select_scatter, slice_scatter_21
triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_9 = async_compile.triton('triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_mul_select_scatter_slice_scatter_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1066104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5304)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + (5304 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr0 + (10608 + (26*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = 1 + x2
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tmp14 < tmp3
    tmp18 = tmp16 & tmp17
    tmp19 = tmp18 & tmp5
    tmp20 = tmp5 & tmp19
    tmp21 = tmp6 >= tmp15
    tmp22 = tmp21 & tmp20
    tmp23 = tl.load(in_ptr0 + (5304 + x5), tmp22 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr0 + (5303 + x5), tmp22 & xmask, other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr0 + (10608 + x5), tmp22 & xmask, other=0.0)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr0 + (10607 + x5), tmp22 & xmask, other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = 0.25
    tmp31 = tmp29 * tmp30
    tmp32 = tl.where(tmp22, tmp31, 0.0)
    tmp33 = 0.0
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tl.where(tmp20, tmp34, 0.0)
    tmp36 = tl.where(tmp5, tmp35, tmp33)
    tmp37 = tl.where(tmp19, tmp36, 0.0)
    tmp38 = tl.where(tmp18, tmp37, tmp33)
    tmp39 = tl.where(tmp8, tmp13, tmp38)
    tmp40 = tl.where(tmp5, tmp39, 0.0)
    tmp41 = tmp5 & tmp18
    tmp42 = tmp21 & tmp41
    tmp43 = tl.load(in_ptr0 + (5304 + x5), tmp42 & xmask, other=0.0)
    tmp44 = tl.load(in_ptr0 + (5303 + x5), tmp42 & xmask, other=0.0)
    tmp45 = tmp43 + tmp44
    tmp46 = tl.load(in_ptr0 + (10608 + x5), tmp42 & xmask, other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tl.load(in_ptr0 + (10607 + x5), tmp42 & xmask, other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49 * tmp30
    tmp51 = tl.where(tmp42, tmp50, 0.0)
    tmp52 = tl.where(tmp21, tmp51, tmp33)
    tmp53 = tl.where(tmp41, tmp52, 0.0)
    tmp54 = tl.where(tmp5, tmp53, tmp33)
    tmp55 = tl.where(tmp18, tmp54, 0.0)
    tmp56 = tl.where(tmp18, tmp55, tmp33)
    tmp57 = tl.where(tmp5, tmp40, tmp56)
    tl.store(out_ptr0 + (x5), tmp57, xmask)
''')
