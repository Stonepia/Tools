

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/iz/ciz46fnyxo5hy62xxmztuqcanlvpc3wqkzchqk5byhd7jmeutx2c.py
# Source Nodes: [iadd_2, setitem_10], Original ATen: [aten.copy, aten.slice_scatter]
# iadd_2 => slice_scatter_49, slice_scatter_50, slice_scatter_51, slice_scatter_52, slice_scatter_53
# setitem_10 => copy_10, slice_scatter_41, slice_scatter_42, slice_scatter_43, slice_scatter_44, slice_scatter_45
triton_poi_fused_copy_slice_scatter_17 = async_compile.triton('triton_poi_fused_copy_slice_scatter_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x1 = (xindex // 26) % 204
    x3 = xindex % 5304
    x0 = xindex % 26
    x4 = xindex
    tmp39 = tl.load(in_out_ptr0 + (x4), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + ((-5252) + x3 + (5200*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tmp5 & tmp5
    tmp15 = tmp10 & tmp14
    tmp16 = x0
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_out_ptr0 + (x4), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.load(in_out_ptr0 + (x4), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp15, tmp22, 0.0)
    tmp24 = tl.load(in_out_ptr0 + (x4), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp14, tmp25, 0.0)
    tmp27 = tl.load(in_out_ptr0 + (x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp5, tmp26, tmp27)
    tmp29 = tl.where(tmp10, tmp13, tmp28)
    tmp30 = tl.where(tmp5, tmp29, 0.0)
    tmp31 = tmp17 & tmp11
    tmp32 = tl.load(in_out_ptr0 + (x4), tmp31 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp31, tmp32, 0.0)
    tmp34 = tl.load(in_out_ptr0 + (x4), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp17, tmp33, tmp34)
    tmp36 = tl.where(tmp11, tmp35, 0.0)
    tmp37 = tl.where(tmp10, tmp36, tmp27)
    tmp38 = tl.where(tmp5, tmp37, 0.0)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tl.where(tmp5, tmp30, tmp40)
    tl.store(in_out_ptr0 + (x4), tmp41, xmask)
''')
