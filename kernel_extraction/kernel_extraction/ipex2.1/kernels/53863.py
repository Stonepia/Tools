

# Original file: ./rexnet_100___60.0/rexnet_100___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/mn/cmnxjq5ul6jy524gifqhcl6kxiedwjow2ubmmjrpeaxjesy27ir4.py
# Source Nodes: [getattr_l__mod___features___11___act_dw, mul_8], Original ATen: [aten.hardtanh, aten.mul]
# getattr_l__mod___features___11___act_dw => clamp_max_11, clamp_min_11
# mul_8 => mul_152
triton_poi_fused_hardtanh_mul_39 = async_compile.triton('triton_poi_fused_hardtanh_mul_39', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_mul_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardtanh_mul_39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')