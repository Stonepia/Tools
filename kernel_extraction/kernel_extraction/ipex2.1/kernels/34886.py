

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/zk/czkquk3o6hljtif5yezwgoqkp5uhrvklpymurzxfws76jmw5jrkr.py
# Source Nodes: [setitem_8], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_8 => copy_8, slice_scatter_28
triton_poi_fused_copy_slice_scatter_9 = async_compile.triton('triton_poi_fused_copy_slice_scatter_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x3 = xindex % 5200
    x4 = (xindex // 26)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 1 + x2
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 202, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = 2 + x1
    tmp10 = tl.full([1], 2, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tmp9 < tmp5
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp8
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + ((-1) + x0), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (5356 + x3 + (5304*x2)), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr3 + ((-1) + x0 + (25*x4)), tmp15 & xmask, other=0.0)
    tmp23 = tl.abs(tmp22)
    tmp24 = -tmp23
    tmp25 = 0.001
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 / tmp25
    tmp28 = libdevice.tanh(tmp27)
    tmp29 = 1.0
    tmp30 = tmp28 + tmp29
    tmp31 = 0.5
    tmp32 = tmp30 * tmp31
    tmp33 = tmp21 * tmp32
    tmp34 = 50.0
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tmp19 * tmp35
    tmp37 = 0.0
    tmp38 = tmp37 + tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tl.where(tmp15, tmp39, 0.0)
    tmp41 = tl.where(tmp2, tmp40, tmp37)
    tmp42 = tl.where(tmp14, tmp41, 0.0)
    tmp43 = tl.where(tmp13, tmp42, tmp37)
    tmp44 = tl.where(tmp8, tmp43, 0.0)
    tmp45 = tl.where(tmp7, tmp44, tmp37)
    tmp46 = tl.where(tmp2, tmp45, 0.0)
    tmp47 = tmp13 & tmp7
    tmp48 = tmp2 & tmp47
    tmp49 = tl.load(in_ptr0 + ((-1) + x0), tmp48 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr1 + (5356 + x3 + (5304*x2)), tmp48 & xmask, other=0.0).to(tl.float32)
    tmp51 = tmp49 * tmp50
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp48 & xmask, other=0.0).to(tl.float32)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr3 + ((-1) + x0 + (25*x4)), tmp48 & xmask, other=0.0)
    tmp56 = tl.abs(tmp55)
    tmp57 = -tmp56
    tmp58 = tmp57 + tmp25
    tmp59 = tmp58 / tmp25
    tmp60 = libdevice.tanh(tmp59)
    tmp61 = tmp60 + tmp29
    tmp62 = tmp61 * tmp31
    tmp63 = tmp54 * tmp62
    tmp64 = triton_helpers.maximum(tmp34, tmp63)
    tmp65 = tmp52 * tmp64
    tmp66 = tmp37 + tmp65
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tl.where(tmp48, tmp67, 0.0)
    tmp69 = tl.where(tmp2, tmp68, tmp37)
    tmp70 = tl.where(tmp47, tmp69, 0.0)
    tmp71 = tl.where(tmp13, tmp70, tmp37)
    tmp72 = tl.where(tmp7, tmp71, 0.0)
    tmp73 = tl.where(tmp7, tmp72, tmp37)
    tmp74 = tl.where(tmp2, tmp46, tmp73)
    tl.store(out_ptr0 + (x5), tmp74, xmask)
''')
