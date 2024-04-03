

# Original file: ./detectron2_maskrcnn_r_101_fpn__74_inference_114.54/detectron2_maskrcnn_r_101_fpn__74_inference_114.54_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/q3/cq3gpo2lxmubg5vxd23auam65qqaf5edu44dqsmosyx6fihb6exf.py
# Source Nodes: [imul, imul_1, setitem], Original ATen: [aten.copy, aten.mul, aten.slice, aten.slice_scatter]
# imul => mul, slice_4, slice_scatter, slice_scatter_1
# imul_1 => mul_1, slice_15, slice_scatter_4
# setitem => copy, slice_scatter_2, slice_scatter_3
triton_poi_fused_copy_mul_slice_slice_scatter_0 = async_compile.triton('triton_poi_fused_copy_mul_slice_slice_scatter_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mul_slice_slice_scatter_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_mul_slice_slice_scatter_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x2 = xindex
    x1 = (xindex // 4)
    tmp40 = tl.load(in_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((-1) + x0) % 2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tmp1 == tmp4
    tmp8 = tmp7 & tmp6
    tmp9 = (2*(((-1) + x2) // 2)) % 2
    tmp10 = tmp9 == tmp4
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((2*(((-1) + x0) // 2)) + (4*x1)), tmp11 & xmask, other=0.0)
    tmp13 = 0.5337781484570475
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp11, tmp14, 0.0)
    tmp16 = tl.load(in_ptr0 + ((2*(((-1) + x0) // 2)) + (4*x1)), tmp8 & xmask, other=0.0)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp8, tmp17, 0.0)
    tmp19 = tmp16 * tmp13
    tmp20 = tl.where(tmp8, tmp19, 0.0)
    tmp21 = tl.load(in_ptr0 + (1 + (2*(((-1) + x0) // 2)) + (4*x1)), tmp6 & xmask, other=0.0)
    tmp22 = tl.where(tmp7, tmp20, tmp21)
    tmp23 = tl.where(tmp7, tmp18, tmp22)
    tmp24 = 0.53375
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp6, tmp25, 0.0)
    tmp27 = x0 % 2
    tmp28 = tmp27 == tmp4
    tmp29 = (2*(x0 // 2)) % 2
    tmp30 = tmp29 == tmp4
    tmp31 = tmp30 & tmp28
    tmp32 = tl.load(in_ptr0 + ((2*(x0 // 2)) + (4*x1)), tmp31 & xmask, other=0.0)
    tmp33 = tmp32 * tmp13
    tmp34 = tl.where(tmp31, tmp33, 0.0)
    tmp35 = tl.load(in_ptr0 + ((2*(x0 // 2)) + (4*x1)), tmp28 & xmask, other=0.0)
    tmp36 = tl.where(tmp30, tmp34, tmp35)
    tmp37 = tl.where(tmp28, tmp36, 0.0)
    tmp38 = tmp35 * tmp13
    tmp39 = tl.where(tmp28, tmp38, 0.0)
    tmp41 = tl.where(tmp28, tmp39, tmp40)
    tmp42 = tl.where(tmp28, tmp37, tmp41)
    tmp43 = tl.where(tmp6, tmp26, tmp42)
    tl.store(out_ptr0 + (x2), tmp43, xmask)
''')
