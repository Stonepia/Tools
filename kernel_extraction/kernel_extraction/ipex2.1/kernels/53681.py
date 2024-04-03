

# Original file: ./rexnet_100___60.0/rexnet_100___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/2b/c2bbgaxbmjbxjb4vhukg3a3wwyqn6q2rax3jdfy7wvs22ozdtfyc.py
# Source Nodes: [getattr_l__mod___features___5___act_dw, mul_2], Original ATen: [aten.hardtanh, aten.mul]
# getattr_l__mod___features___5___act_dw => clamp_max_5, clamp_min_5, convert_element_type_82, convert_element_type_83
# mul_2 => mul_68
triton_poi_fused_hardtanh_mul_21 = async_compile.triton('triton_poi_fused_hardtanh_mul_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_mul_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardtanh_mul_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9182208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 366
    x2 = (xindex // 71736)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + (366*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')
