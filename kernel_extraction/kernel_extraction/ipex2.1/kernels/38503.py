

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ys/cysfacfccubzfl55l6j5creycfj3sqijixmjgwoyrlw7ktxquiky.py
# Source Nodes: [interpolate_3, leaky_relu_17], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.leaky_relu, aten.mul, aten.rsub, aten.sub]
# interpolate_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_21, add_23, add_25, add_26, add_27, clamp_min_6, convert_element_type_100, convert_element_type_104, convert_element_type_98, iota_6, mul_48, mul_50, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, sub_18, sub_20, sub_21, sub_22, sub_23
# leaky_relu_17 => convert_element_type_95, gt_17, mul_47, where_17
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_22 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_22', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 32768], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_leaky_relu_mul_rsub_sub_22(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 30976
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 176)
    x2 = xindex % 176
    y0 = yindex % 128
    y1 = (yindex // 128)
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
    tmp21 = tl.load(in_ptr0 + (y0 + (128*tmp20) + (11264*tmp11) + (991232*y1)), xmask & ymask).to(tl.float32)
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
    tmp32 = 87.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tmp33.to(tl.int32)
    tmp35 = tl.load(in_ptr0 + (y0 + (128*tmp20) + (11264*tmp34) + (991232*y1)), xmask & ymask).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 > tmp4
    tmp38 = tmp36 * tmp24
    tmp39 = tl.where(tmp37, tmp36, tmp38)
    tmp40 = tmp39 * tmp28
    tmp41 = tmp30 + tmp40
    tmp42 = libdevice.ceil(tmp19)
    tmp43 = triton_helpers.minimum(tmp42, tmp32)
    tmp44 = tmp43.to(tl.int32)
    tmp45 = tl.load(in_ptr0 + (y0 + (128*tmp44) + (11264*tmp11) + (991232*y1)), xmask & ymask).to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 > tmp4
    tmp48 = tmp46 * tmp24
    tmp49 = tl.where(tmp47, tmp46, tmp48)
    tmp50 = tmp49 * tmp29
    tmp51 = tl.load(in_ptr0 + (y0 + (128*tmp44) + (11264*tmp34) + (991232*y1)), xmask & ymask).to(tl.float32)
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
    tl.store(out_ptr1 + (y0 + (128*x4) + (3964928*y1)), tmp64, xmask & ymask)
''')
