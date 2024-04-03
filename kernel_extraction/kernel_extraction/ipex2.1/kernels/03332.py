

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/dx/cdxdznhgc4y2vvrf2iuf4b7ddp7epyff4ieacuiz3zqcxborkibp.py
# Source Nodes: [add_4, add_5, l__self___bert_encoder_layer_0_output_dense, mul_19, mul_20, mul_21, mul_22, pow_1, tanh], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.pow, aten.tanh]
# add_4 => add_20
# add_5 => add_21
# l__self___bert_encoder_layer_0_output_dense => convert_element_type_35
# mul_19 => mul_25
# mul_20 => mul_26
# mul_21 => mul_27
# mul_22 => mul_28
# pow_1 => convert_element_type_34, pow_1
# tanh => tanh
triton_poi_fused__to_copy_add_mul_pow_tanh_30 = async_compile.triton('triton_poi_fused__to_copy_add_mul_pow_tanh_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_pow_tanh_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_pow_tanh_30(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
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
