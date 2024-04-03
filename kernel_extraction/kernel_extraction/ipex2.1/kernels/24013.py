

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/hs/chsjmu5m2twj542gv4hvpj2ycusv2ya7hgbm57pnc6qezumtvw5p.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => clone_49, div_6, exp_6, sub_23
triton_poi_fused__softmax_66 = async_compile.triton('triton_poi_fused__softmax_66', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_66', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_66(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = (xindex // 3200)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, None)
''')
