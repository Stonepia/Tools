

# Original file: ./cspdarknet53___60.0/cspdarknet53___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/uo/cuo5zmpiwoabv4cui3oh5dmc4pnv5lvgi7jzdzv6kyqkfwmpx4m5.py
# Source Nodes: [getattr_l__self___stages___0___conv_exp_bn_act], Original ATen: [aten.leaky_relu]
# getattr_l__self___stages___0___conv_exp_bn_act => convert_element_type_18, gt_2, mul_11, where_2
triton_poi_fused_leaky_relu_4 = async_compile.triton('triton_poi_fused_leaky_relu_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 16384], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_leaky_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (2097152*y1)), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp6, None)
''')
