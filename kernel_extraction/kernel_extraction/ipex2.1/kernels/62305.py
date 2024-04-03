

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/fv/cfvcbqnshscflxvaja2sminmwoaqnn7hwo5tfqlf6lzroexpxnys.py
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*i1', 3: '*bf16', 4: '*i1', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_relu_threshold_backward_view_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_native_dropout_relu_threshold_backward_view_13(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = 0.0
    tmp13 = tmp8 <= tmp12
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp11, None)
    tl.store(out_ptr3 + (x0), tmp13, None)
''')
