

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/st/cst6tlbdvfcxyu6767btym2ugvps4fa35r6fsetphldjwk522oi6.py
# Source Nodes: [stack_4, stack_5], Original ATen: [aten.stack]
# stack_4 => cat_16
# stack_5 => cat_17
triton_poi_fused_stack_43 = async_compile.triton('triton_poi_fused_stack_43', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_43', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_43(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (3 + (4*x0)), None).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (1 + (4*x0)), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 + tmp8
    tmp10 = 352.0
    tmp11 = tmp9 / tmp10
    tmp12 = 0.5
    tmp13 = tmp11 - tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 > tmp3
    tmp19 = tmp17 * tmp5
    tmp20 = tl.where(tmp18, tmp17, tmp19)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp0 + tmp21
    tmp23 = tmp22 / tmp10
    tmp24 = tmp23 - tmp12
    tmp25 = tmp24 * tmp14
    tl.store(out_ptr0 + (2*x0), tmp15, None)
    tl.store(out_ptr1 + (2*x0), tmp25, None)
''')
