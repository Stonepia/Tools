

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/np/cnpuyq5rqjyhbodk4vsunv2hk7zg46nwr55oiccyat7f6lubfojx.py
# Source Nodes: [add_6, add_7, l__mod___encoder_block_0_layer__1__dense_relu_dense_dropout, l__mod___encoder_block_0_layer__1__dense_relu_dense_wo, mul_10, mul_11, mul_7, mul_8, mul_9, pow_3, tanh], Original ATen: [aten.add, aten.mul, aten.native_dropout, aten.pow, aten.tanh, aten.view]
# add_6 => add_8
# add_7 => add_9
# l__mod___encoder_block_0_layer__1__dense_relu_dense_dropout => gt_4, mul_18, mul_19
# l__mod___encoder_block_0_layer__1__dense_relu_dense_wo => view_26
# mul_10 => mul_16
# mul_11 => mul_17
# mul_7 => mul_13
# mul_8 => mul_14
# mul_9 => mul_15
# pow_3 => pow_3
# tanh => tanh
triton_poi_fused_add_mul_native_dropout_pow_tanh_view_7 = async_compile.triton('triton_poi_fused_add_mul_native_dropout_pow_tanh_view_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*i1', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_dropout_pow_tanh_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_native_dropout_pow_tanh_view_7(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = 0.044715
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 + tmp10
    tmp12 = 0.7978845608028654
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = tmp5.to(tl.float32)
    tmp16 = 0.5
    tmp17 = tmp6 * tmp16
    tmp18 = 1.0
    tmp19 = tmp14 + tmp18
    tmp20 = tmp17 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp15 * tmp22
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp14, None)
    tl.store(out_ptr3 + (x0), tmp25, None)
''')
