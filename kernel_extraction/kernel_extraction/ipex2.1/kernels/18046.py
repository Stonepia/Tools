

# Original file: ./hf_Albert___60.0/hf_Albert___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/i7/ci76rmymt5bgv5vqn3hf2pqy4vbwmiramv4nfu5jkck56wwjrqmd.py
# Source Nodes: [add_3, add_4, l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output, mul_1, mul_2, mul_3, mul_4, pow_1, tanh], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.pow, aten.tanh]
# add_3 => add_8
# add_4 => add_9
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output => convert_element_type_17
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# mul_4 => mul_8
# pow_1 => convert_element_type_16, pow_1
# tanh => tanh
triton_poi_fused__to_copy_add_mul_pow_tanh_6 = async_compile.triton('triton_poi_fused__to_copy_add_mul_pow_tanh_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_pow_tanh_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_pow_tanh_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tmp5 * tmp4
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp16, None)
''')
