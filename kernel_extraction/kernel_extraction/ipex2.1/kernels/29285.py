

# Original file: ./AlbertForMaskedLM__0_forward_133.0/AlbertForMaskedLM__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/px/cpxddi2ihcojjbz5b3ucij2hxuoqmfdubvt4ijv5ejqbb4lhevin.py
# Source Nodes: [add_3, add_4, l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output, mul_1, mul_2, mul_3, mul_4, pow_1, tanh], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
# add_3 => add_8
# add_4 => add_9
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output => convert_element_type_17, view_22
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# mul_4 => mul_8
# pow_1 => convert_element_type_16, pow_1
# tanh => tanh
triton_poi_fused__to_copy_add_mul_pow_tanh_view_11 = async_compile.triton('triton_poi_fused__to_copy_add_mul_pow_tanh_view_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_pow_tanh_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_pow_tanh_view_11(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tmp2 * tmp1
    tmp4 = 0.044715
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 + tmp5
    tmp7 = 0.7978845608028654
    tmp8 = tmp6 * tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = 0.5
    tmp11 = tmp0 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.0
    tmp14 = tmp9 + tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp9, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')
