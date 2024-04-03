

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/tw/ctw5xjiq2qmmcqwgjmngmvglfar5eoca67yooples7ieglbecsge.py
# Source Nodes: [gelu, l__mod___transformer_layer_0_ff_dropout, l__mod___transformer_layer_0_ff_layer_2], Original ATen: [aten.gelu, aten.native_dropout, aten.view]
# gelu => add_7, convert_element_type_8, convert_element_type_9, erf, mul_13, mul_14, mul_15
# l__mod___transformer_layer_0_ff_dropout => gt_4, mul_16, mul_17
# l__mod___transformer_layer_0_ff_layer_2 => view_36
triton_poi_fused_gelu_native_dropout_view_10 = async_compile.triton('triton_poi_fused_gelu_native_dropout_view_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_native_dropout_view_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gelu_native_dropout_view_10(in_ptr0, in_ptr1, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp7 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp6 * tmp17
    tmp19 = 1.1111111111111112
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''')
