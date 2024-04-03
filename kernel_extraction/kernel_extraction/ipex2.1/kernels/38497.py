

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ra/crawjdz6y2zoulh74utospghgqi7rfuchqfxn7ouju7i7sngkvxa.py
# Source Nodes: [interpolate_1, leaky_relu_13], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.leaky_relu, aten.mul, aten.rsub, aten.sub]
# interpolate_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_11, add_12, add_13, add_7, add_9, clamp_min_2, convert_element_type_66, convert_element_type_68, convert_element_type_72, iota_2, mul_24, mul_26, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, sub_10, sub_11, sub_6, sub_8, sub_9
# leaky_relu_13 => convert_element_type_63, gt_13, mul_23, where_13
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_16 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_16(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 1936
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 44)
    x2 = xindex % 44
    y0 = yindex % 512
    y1 = (yindex // 512)
    x4 = xindex
    y5 = yindex
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 - tmp6
    tmp10 = triton_helpers.maximum(tmp9, tmp4)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = x2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14 + tmp4
    tmp16 = tmp15 + tmp6
    tmp17 = tmp16 * tmp6
    tmp18 = tmp17 - tmp6
    tmp19 = triton_helpers.maximum(tmp18, tmp4)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.load(in_ptr0 + (y0 + (512*tmp20) + (11264*tmp11) + (247808*y1)), xmask).to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 > tmp4
    tmp24 = 0.1
    tmp25 = tmp22 * tmp24
    tmp26 = tl.where(tmp23, tmp22, tmp25)
    tmp27 = tmp11.to(tl.float32)
    tmp28 = tmp10 - tmp27
    tmp29 = tmp2 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = libdevice.ceil(tmp10)
    tmp32 = 21.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tmp33.to(tl.int32)
    tmp35 = tl.load(in_ptr0 + (y0 + (512*tmp20) + (11264*tmp34) + (247808*y1)), xmask).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 > tmp4
    tmp38 = tmp36 * tmp24
    tmp39 = tl.where(tmp37, tmp36, tmp38)
    tmp40 = tmp39 * tmp28
    tmp41 = tmp30 + tmp40
    tmp42 = libdevice.ceil(tmp19)
    tmp43 = triton_helpers.minimum(tmp42, tmp32)
    tmp44 = tmp43.to(tl.int32)
    tmp45 = tl.load(in_ptr0 + (y0 + (512*tmp44) + (11264*tmp11) + (247808*y1)), xmask).to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 > tmp4
    tmp48 = tmp46 * tmp24
    tmp49 = tl.where(tmp47, tmp46, tmp48)
    tmp50 = tmp49 * tmp29
    tmp51 = tl.load(in_ptr0 + (y0 + (512*tmp44) + (11264*tmp34) + (247808*y1)), xmask).to(tl.float32)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 > tmp4
    tmp54 = tmp52 * tmp24
    tmp55 = tl.where(tmp53, tmp52, tmp54)
    tmp56 = tmp55 * tmp28
    tmp57 = tmp50 + tmp56
    tmp58 = tmp20.to(tl.float32)
    tmp59 = tmp19 - tmp58
    tmp60 = tmp2 - tmp59
    tmp61 = tmp41 * tmp60
    tmp62 = tmp57 * tmp59
    tmp63 = tmp61 + tmp62
    tmp64 = tmp63.to(tl.float32)
    tl.store(out_ptr1 + (y0 + (512*x4) + (991232*y1)), tmp64, xmask)
''')
