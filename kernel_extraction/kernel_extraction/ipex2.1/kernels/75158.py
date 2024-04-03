

# Original file: ./detectron2_maskrcnn_r_101_fpn__67_inference_107.47/detectron2_maskrcnn_r_101_fpn__67_inference_107.47_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/hc/chc7m65fpxd75e5iif77iputavncbjad2qxurpo3fcanu76aneoy.py
# Source Nodes: [add, add_1, clamp, floor, log2, mul, sqrt, sub, sub_1, sub_2, to, truediv], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.div, aten.floor, aten.log2, aten.mul, aten.sqrt, aten.sub]
# add => add
# add_1 => add_1
# clamp => clamp_max, clamp_min
# floor => floor
# log2 => log, mul_1
# mul => mul
# sqrt => sqrt
# sub => sub
# sub_1 => sub_1
# sub_2 => sub_2
# to => convert_element_type
# truediv => div
triton_poi_fused__to_copy_add_clamp_div_floor_log2_mul_sqrt_sub_0 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_div_floor_log2_mul_sqrt_sub_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_div_floor_log2_mul_sqrt_sub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clamp_div_floor_log2_mul_sqrt_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + (4*x0)), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x0), xmask)
    tmp3 = tl.load(in_ptr0 + (3 + (4*x0)), xmask)
    tmp4 = tl.load(in_ptr0 + (1 + (4*x0)), xmask)
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 224.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-08
    tmp11 = tmp9 + tmp10
    tmp12 = tl.log(tmp11)
    tmp13 = 1.4426950408889634
    tmp14 = tmp12 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = 2.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 5.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp21.to(tl.int64)
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp22 - tmp23
    tl.store(out_ptr0 + (x0), tmp24, xmask)
''')
