

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ni/cniudt47nwq53dqi7cfewm7n77eiortl4ru354hdnsfy7ddlxwxi.py
# Source Nodes: [l__mod___head_drop], Original ATen: [aten.clone]
# l__mod___head_drop => clone_64
triton_poi_fused_clone_79 = async_compile.triton('triton_poi_fused_clone_79', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_79', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25600*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (25600*x1)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (25600*x1)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (50*x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (50*x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = 512.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp21, None)
''')
