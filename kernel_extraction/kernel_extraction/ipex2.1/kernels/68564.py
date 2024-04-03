

# Original file: ./convnext_base___60.0/convnext_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/d7/cd7oxaqwpeiufspy6mognmmm3bqgnbh2z4mfwct7ku6irtbzgieh.py
# Source Nodes: [add_3, getattr_getattr_l__self___stages___1___blocks___1___conv_dw, mul_3], Original ATen: [aten._to_copy, aten.add, aten.mul]
# add_3 => add_19
# getattr_getattr_l__self___stages___1___blocks___1___conv_dw => convert_element_type_50
# mul_3 => mul_27
triton_poi_fused__to_copy_add_mul_8 = async_compile.triton('triton_poi_fused__to_copy_add_mul_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21233664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
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
