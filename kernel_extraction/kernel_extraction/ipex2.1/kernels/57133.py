

# Original file: ./Background_Matting___60.0/Background_Matting___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/jo/cjott7xlgkdzst4r4worgq7mrkbdaxgrlpgnpl2wt3igr2z2c5uc.py
# Source Nodes: [l__self___model_al_out_4], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# l__self___model_al_out_4 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_89, add_91, add_92, add_93, convert_element_type_193, convert_element_type_194, convert_element_type_196, convert_element_type_200, iota_2, mul_121, mul_123, mul_125, mul_126, mul_127, mul_128, mul_129, mul_130, sub_41, sub_42, sub_43, sub_44
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_8 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128, 262144], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_8(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 512)
    x1 = xindex % 512
    y0 = yindex
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49902152641878667
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (y0 + (128*tmp14) + (32768*tmp8)), ymask).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.ceil(tmp7)
    tmp22 = 255.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (y0 + (128*tmp14) + (32768*tmp24)), ymask).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp18
    tmp28 = tmp20 + tmp27
    tmp29 = libdevice.ceil(tmp13)
    tmp30 = triton_helpers.minimum(tmp29, tmp22)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (y0 + (128*tmp31) + (32768*tmp8)), ymask).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp19
    tmp35 = tl.load(in_ptr0 + (y0 + (128*tmp31) + (32768*tmp24)), ymask).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp18
    tmp38 = tmp34 + tmp37
    tmp39 = tmp14.to(tl.float32)
    tmp40 = tmp13 - tmp39
    tmp41 = tmp2 - tmp40
    tmp42 = tmp28 * tmp41
    tmp43 = tmp38 * tmp40
    tmp44 = tmp42 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr2 + (y0 + (128*x3)), tmp45, ymask)
''')
