

# Original file: ./YituTechConvBert__0_forward_205.0/YituTechConvBert__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/m7/cm7qs2zzwtgsclvg6bu3mc3q3llraodsxb22hp77gdq3kly5oquc.py
# Source Nodes: [l__self___convbert_encoder_layer_0_attention_self_dropout, matmul_2], Original ATen: [aten._to_copy, aten.native_dropout]
# l__self___convbert_encoder_layer_0_attention_self_dropout => mul_6, mul_7
# matmul_2 => convert_element_type_22
triton_poi_fused__to_copy_native_dropout_16 = async_compile.triton('triton_poi_fused__to_copy_native_dropout_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304, 8], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_dropout_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_native_dropout_16(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 6
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 262144
    y1 = (yindex // 262144)
    tmp0 = tl.load(in_ptr0 + (x2 + (6*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2 + (6*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (262144*x2) + (1572864*y1)), tmp6, xmask)
''')
