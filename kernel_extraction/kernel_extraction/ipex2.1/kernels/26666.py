

# Original file: ./pytorch_stargan___60.0/pytorch_stargan___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/br/cbrx3owuzufohyavna623tas7qfdy47a6ne2oviuvg5htlwnbhyv.py
# Source Nodes: [add_1], Original ATen: [aten.add]
# add_1 => add_15
triton_poi_fused_add_10 = async_compile.triton('triton_poi_fused_add_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x0 + (256*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (x0 + (256*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 - tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sqrt(tmp9)
    tmp11 = 1 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp5 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp0 + tmp21
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')
