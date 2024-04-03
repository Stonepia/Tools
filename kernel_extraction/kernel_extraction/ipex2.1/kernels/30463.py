

# Original file: ./XLNetLMHeadModel__0_backward_567.1/XLNetLMHeadModel__0_backward_567.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/vd/cvd7sgcpuxubjn3ab5om3ub7ucdpryxz5uma54sxh43bqucl3qhx.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]

triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_13 = async_compile.triton('triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67043328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')
