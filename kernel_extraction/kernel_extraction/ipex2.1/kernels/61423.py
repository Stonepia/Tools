

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/4c/c4cjeua6p4vjnmrjizmc3ylgsr6frgaqaeeql2juqfhdog5brptc.py
# Source Nodes: [add_56, l__mod___decoder_block_0_layer_0_dropout, l__mod___decoder_dropout, neg_17, where_1, where_18], Original ATen: [aten.add, aten.ge, aten.le, aten.logical_and, aten.native_dropout, aten.neg, aten.scalar_tensor, aten.where]
# add_56 => add_66
# l__mod___decoder_block_0_layer_0_dropout => mul_155, mul_156
# l__mod___decoder_dropout => mul_148, mul_149
# neg_17 => neg_17
# where_1 => full_default_2, full_default_3
# where_18 => where_18
triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21 = async_compile.triton('triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*i1', 5: '*i1', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp8 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp4
    tmp11 = tmp5 + tmp10
    tmp14 = 64504.0
    tmp15 = 65504.0
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = -tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp11 >= tmp18
    tmp20 = tmp16.to(tl.float32)
    tmp21 = tmp11 <= tmp20
    tmp22 = tmp19 & tmp21
    tl.store(out_ptr0 + (x0), tmp22, None)
''')
