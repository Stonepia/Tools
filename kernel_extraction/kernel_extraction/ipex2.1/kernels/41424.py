

# Original file: ./vision_maskrcnn__33_inference_73.13/vision_maskrcnn__33_inference_73.13_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/a7/ca7bb6r4dznz6ln3qlvhooborgq274erght25lviykwsg5g5zrlh.py
# Source Nodes: [add, add_1, clamp, floor, log2, mul, sqrt, sub, sub_1, sub_2, tensor, to, truediv], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.div, aten.floor, aten.lift_fresh, aten.log2, aten.mul, aten.sqrt, aten.sub]
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
# tensor => full_default
# to => convert_element_type_1
# truediv => div
triton_poi_fused__to_copy_add_clamp_div_floor_lift_fresh_log2_mul_sqrt_sub_0 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_div_floor_lift_fresh_log2_mul_sqrt_sub_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_div_floor_lift_fresh_log2_mul_sqrt_sub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clamp_div_floor_lift_fresh_log2_mul_sqrt_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + (4*x0)), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (4*x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (3 + (4*x0)), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (1 + (4*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 224.0
    tmp13 = tmp11 / tmp12
    tmp14 = tl.log(tmp13)
    tmp15 = 1.4426950408889634
    tmp16 = tmp14 * tmp15
    tmp17 = 4.0
    tmp18 = tmp16 + tmp17
    tmp19 = 9.999999974752427e-07
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.floor(tmp20)
    tmp22 = 2.0
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = 5.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25.to(tl.int64)
    tmp27 = tl.full([1], 2, tl.int64)
    tmp28 = tmp26 - tmp27
    tl.store(out_ptr0 + (x0), tmp28, xmask)
''')
