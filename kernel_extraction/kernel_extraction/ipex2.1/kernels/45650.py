

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/re/crezwusvgqwlg5kzqng2l7u6ilvkmejbynqp72r3wrtk2vyfxwn2.py
# Source Nodes: [add_61, l__mod___decoder_block_4_layer__1__dropout, neg_27, where_1, where_28], Original ATen: [aten.add, aten.ge, aten.le, aten.logical_and, aten.native_dropout, aten.neg, aten.scalar_tensor, aten.where]
# add_61 => add_78
# l__mod___decoder_block_4_layer__1__dropout => mul_175, mul_176
# neg_27 => neg_27
# where_1 => full_default_2, full_default_3
# where_28 => where_28
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

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*i1', 4: '*i1', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_ge_le_logical_and_native_dropout_neg_scalar_tensor_where_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp10 = 64504.0
    tmp11 = 65504.0
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = -tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp7 >= tmp14
    tmp16 = tmp12.to(tl.float32)
    tmp17 = tmp7 <= tmp16
    tmp18 = tmp15 & tmp17
    tl.store(out_ptr0 + (x0), tmp18, None)
''')
