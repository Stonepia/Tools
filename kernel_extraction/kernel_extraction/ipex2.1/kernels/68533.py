

# Original file: ./convnext_base___60.0/convnext_base___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/vo/cvoblsjsufwv5gezqkfwaibfyzbqvlbgakyl5tlxyvnnwmamqqow.py
# Source Nodes: [add_33, getattr_getattr_l__self___stages___3___blocks___1___conv_dw, mul_33], Original ATen: [aten._to_copy, aten.add, aten.mul]
# add_33 => add_143
# getattr_getattr_l__self___stages___3___blocks___1___conv_dw => convert_element_type_384
# mul_33 => mul_211
triton_poi_fused__to_copy_add_mul_20 = async_compile.triton('triton_poi_fused__to_copy_add_mul_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''')