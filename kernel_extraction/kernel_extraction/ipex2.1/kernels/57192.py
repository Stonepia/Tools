

# Original file: ./Background_Matting___60.0/Background_Matting___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/oi/coixnerkvqj3eslyiisqp2roiqmcpck3n2zcy6mlpeb2bhafkipa.py
# Source Nodes: [l__mod___model_al_out_0], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# l__mod___model_al_out_0 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_82, add_84, add_85, add_86, convert_element_type_108, convert_element_type_109, convert_element_type_111, convert_element_type_115, iota, mul_108, mul_110, mul_112, mul_113, mul_114, mul_115, mul_116, mul_117, sub_36, sub_37, sub_38, sub_39
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 65536], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 256)
    x1 = xindex % 256
    y0 = yindex
    x3 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.4980392156862745
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (y0 + (256*tmp14) + (32768*tmp8)), ymask).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.ceil(tmp7)
    tmp22 = 127.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (y0 + (256*tmp14) + (32768*tmp24)), ymask).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp18
    tmp28 = tmp20 + tmp27
    tmp29 = libdevice.ceil(tmp13)
    tmp30 = triton_helpers.minimum(tmp29, tmp22)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (y0 + (256*tmp31) + (32768*tmp8)), ymask).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp19
    tmp35 = tl.load(in_ptr0 + (y0 + (256*tmp31) + (32768*tmp24)), ymask).to(tl.float32)
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
    tl.store(out_ptr2 + (y0 + (256*x3)), tmp45, ymask)
''')
