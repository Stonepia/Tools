

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ue/cuevvxnudlkyytxqt5cesqtvoo5o3exw6cjdqcw524crqeagjco5.py
# Source Nodes: [batch_norm, conv2d, conv2d_1, mul_4, relu, sub_4], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.mul, aten.relu, aten.sub]
# batch_norm => add, add_3, convert_element_type_3, convert_element_type_4, mul_18, mul_24, rsqrt, sub_18, var_mean
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# conv2d_1 => convert_element_type_5, convert_element_type_6, convolution_1
# mul_4 => mul_4
# relu => relu
# sub_4 => sub_4
triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_mul_relu_sub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = 0.4
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp5, None)
''')
