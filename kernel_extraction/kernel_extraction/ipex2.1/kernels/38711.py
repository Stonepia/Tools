

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ab/cab37yj6mtil7wmkcntroo45i57tvnn4piamdkadqnndpp2kgzfb.py
# Source Nodes: [stack_4, stack_5], Original ATen: [aten.stack]
# stack_4 => cat_16
# stack_5 => cat_17
triton_poi_fused_stack_38 = async_compile.triton('triton_poi_fused_stack_38', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (3 + (4*x0)), None)
    tmp9 = tl.load(in_ptr1 + (1 + (4*x0)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 352.0
    tmp4 = tmp2 / tmp3
    tmp5 = 0.5
    tmp6 = tmp4 - tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp0 + tmp9
    tmp11 = tmp10 / tmp3
    tmp12 = tmp11 - tmp5
    tmp13 = tmp12 * tmp7
    tl.store(out_ptr0 + (2*x0), tmp8, None)
    tl.store(out_ptr1 + (2*x0), tmp13, None)
''')
