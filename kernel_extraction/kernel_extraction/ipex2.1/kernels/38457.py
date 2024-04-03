

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/vv/cvv76pzgqzv4mg6evgcvynbxbrhwg3h2aw4vpsq4kciv3m7hvce5.py
# Source Nodes: [stack_2, stack_3], Original ATen: [aten.stack]
# stack_2 => cat_14
# stack_3 => cat_15
triton_poi_fused_stack_40 = async_compile.triton('triton_poi_fused_stack_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 123904
    x1 = (xindex // 123904)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (1 + (5*x2)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (123904 + x0 + (2478080*x1)), None)
    tmp19 = tl.load(in_ptr1 + (3 + (5*x2)), None).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (123904 + x0 + (2478080*x1)), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp13 = 352.0
    tmp14 = tmp12 / tmp13
    tmp15 = 0.5
    tmp16 = tmp14 - tmp15
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 > tmp3
    tmp22 = tmp20 * tmp5
    tmp23 = tl.where(tmp21, tmp20, tmp22)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp0 + tmp27
    tmp29 = tmp28 / tmp13
    tmp30 = tmp29 - tmp15
    tmp31 = tmp30 * tmp17
    tl.store(out_ptr0 + (2*x2), tmp18, None)
    tl.store(out_ptr1 + (2*x2), tmp31, None)
''')
