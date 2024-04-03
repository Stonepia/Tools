

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ou/coumc5zceaiosqjl4bd2ehjj632gsklqzhycmymk5ox5l4teslnf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___bn2_act, mul_13], Original ATen: [aten.mul, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___bn2_act => convert_element_type_231, mul_203, sigmoid_53
# mul_13 => mul_205
triton_poi_fused_mul_silu_104 = async_compile.triton('triton_poi_fused_mul_silu_104', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144, 64], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_104', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_silu_104(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 202752
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1584
    y1 = (yindex // 1584)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1584*x2) + (77616*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp5, xmask)
''')
