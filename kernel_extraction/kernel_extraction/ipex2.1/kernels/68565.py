

# Original file: ./convnext_base___60.0/convnext_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/tf/ctfbhu4ja6l52s6ig4vqvf5miysn6xiaxhtnxqvtkdakspkavgdq.py
# Source Nodes: [add_3, add_4, getattr_getattr_l__self___stages___1___blocks___2___conv_dw, mul_3, mul_4], Original ATen: [aten._to_copy, aten.add, aten.mul]
# add_3 => add_19
# add_4 => add_23
# getattr_getattr_l__self___stages___1___blocks___2___conv_dw => convert_element_type_61
# mul_3 => mul_27
# mul_4 => mul_33
triton_poi_fused__to_copy_add_mul_9 = async_compile.triton('triton_poi_fused__to_copy_add_mul_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21233664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp3 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
''')
