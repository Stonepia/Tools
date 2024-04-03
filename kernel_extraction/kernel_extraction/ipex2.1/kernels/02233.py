

# Original file: ./AlbertForQuestionAnswering__0_forward_133.0/AlbertForQuestionAnswering__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/l2/cl2y5cxznlmyvvq5cbwdzgx3h5xdoksolsx4kxsanm2cp75obn5k.py
# Source Nodes: [add_3, add_4, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output, mul_1, mul_2, mul_3, mul_4, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
# add_3 => add_8
# add_4 => add_9
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn_output => view_22
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# mul_4 => mul_8
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_view_6 = async_compile.triton('triton_poi_fused_add_mul_pow_tanh_view_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_view_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''')