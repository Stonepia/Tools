

# Original file: ./DALLE2_pytorch__38_inference_78.18/DALLE2_pytorch__38_inference_78.18.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/fx/cfxjdqsikgga2h635jdjdk4v4fu2lt4b52epsb2nuzjbpththlae.py
# Source Nodes: [add, exp, mul, mul_1, mul_2], Original ATen: [aten.add, aten.exp, aten.mul]
# add => add
# exp => exp
# mul => mul
# mul_1 => mul_1
# mul_2 => mul_2
triton_poi_fused_add_exp_mul_0 = async_compile.triton('triton_poi_fused_add_exp_mul_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_exp_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 512)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.randn(tmp0, (tmp1).to(tl.uint32))
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 == tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tmp9 * tmp13
    tmp15 = tmp14 * tmp2
    tmp16 = tmp3 + tmp15
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)
''')
