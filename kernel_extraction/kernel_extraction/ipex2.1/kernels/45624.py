

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/6b/c6bmjtf3qn7nt4zjhvdi5c4kszdkm4dqaop7argjbapn55y5ni2p.py
# Source Nodes: [l__mod___decoder_block_5_layer__1__dense_relu_dense_act, l__mod___decoder_block_5_layer__1__dense_relu_dense_dropout, l__mod___decoder_block_5_layer__1__dense_relu_dense_wo], Original ATen: [aten.native_dropout, aten.relu, aten.threshold_backward, aten.view]
# l__mod___decoder_block_5_layer__1__dense_relu_dense_act => relu_11
# l__mod___decoder_block_5_layer__1__dense_relu_dense_dropout => gt_62, mul_191, mul_192
# l__mod___decoder_block_5_layer__1__dense_relu_dense_wo => view_426
triton_poi_fused_native_dropout_relu_threshold_backward_view_13 = async_compile.triton('triton_poi_fused_native_dropout_relu_threshold_backward_view_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_relu_threshold_backward_view_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_view_13(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = 0.0
    tmp12 = tmp7 <= tmp11
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp10, None)
    tl.store(out_ptr3 + (x0), tmp12, None)
''')
