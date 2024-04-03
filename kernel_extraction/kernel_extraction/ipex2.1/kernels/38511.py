

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/7q/c7qylqrkjphtckvha3yjod6l2qubfvzfhyrcenck2s4riff5lmun.py
# Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
# stack => cat_6
# stack_1 => cat_7
triton_poi_fused_stack_30 = async_compile.triton('triton_poi_fused_stack_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (2*x0), None)
    tmp2 = tl.load(in_ptr2 + (2*x0), None)
    tmp11 = tl.load(in_ptr3 + (2*x0), None)
    tmp12 = tl.load(in_ptr4 + (2*x0), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 352.0
    tmp6 = tmp4 / tmp5
    tmp7 = 0.5
    tmp8 = tmp6 - tmp7
    tmp9 = 2.0
    tmp10 = tmp8 * tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp0 + tmp13
    tmp15 = tmp14 / tmp5
    tmp16 = tmp15 - tmp7
    tmp17 = tmp16 * tmp9
    tl.store(out_ptr0 + (2*x0), tmp10, None)
    tl.store(out_ptr1 + (2*x0), tmp17, None)
''')
