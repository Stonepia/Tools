

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/hd/chdtrgi33xsq7qehnpmdhsu3onmic4nbqmo5e2kmaal24ewjozox.py
# Source Nodes: [setitem_91], Original ATen: [aten.slice_scatter]
# setitem_91 => slice_scatter_40, slice_scatter_41
triton_poi_fused_slice_scatter_67 = async_compile.triton('triton_poi_fused_slice_scatter_67', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_67', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3182400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 78) % 204
    x3 = (xindex // 15912)
    x4 = xindex % 15912
    x5 = xindex
    x0 = xindex % 3
    x6 = (xindex // 3) % 5304
    tmp27 = tl.load(in_ptr4 + (31824 + x5), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-156) + x4 + (15600*x3)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 2 + x3
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tl.load(in_ptr1 + (x5), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tl.load(in_ptr2 + (x5), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp11, tmp14, 0.0)
    tmp16 = tmp5 & tmp11
    tmp17 = x0
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp17 == tmp18
    tmp20 = tl.load(in_ptr3 + ((-52) + x6 + (5200*x3)), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (31824 + x5), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.where(tmp16, tmp22, 0.0)
    tmp24 = tl.load(in_ptr4 + (31824 + x5), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp5, tmp23, tmp24)
    tmp26 = tl.where(tmp11, tmp25, 0.0)
    tmp28 = tl.where(tmp11, tmp26, tmp27)
    tmp29 = tl.where(tmp11, tmp15, tmp28)
    tmp30 = tl.where(tmp11, tmp13, tmp29)
    tmp31 = tl.where(tmp5, tmp7, tmp30)
    tl.store(out_ptr0 + (x5), tmp31, xmask)
''')
