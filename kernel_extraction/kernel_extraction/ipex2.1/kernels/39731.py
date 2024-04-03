

# Original file: ./levit_128___60.0/levit_128___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/v4/cv4t6e2hvdatw5govqrzmpwgucoryx2bmq2ozyhvetldlhrwj4if.py
# Source Nodes: [matmul_19], Original ATen: [aten.clone]
# matmul_19 => clone_49
triton_poi_fused_clone_31 = async_compile.triton('triton_poi_fused_clone_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 49
    x2 = (xindex // 3136) % 16
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (1280*x1) + (62720*x3)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp10, None)
''')
