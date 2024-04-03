

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/sg/csgordvszqtacnlucdfxk75iw3ibkd3qndwo6fa726ezswu7txsi.py
# Source Nodes: [l__self___serial_blocks3_0_cpe_proj_1], Original ATen: [aten._to_copy]
# l__self___serial_blocks3_0_cpe_proj_1 => convert_element_type_133
triton_poi_fused__to_copy_63 = async_compile.triton('triton_poi_fused__to_copy_63', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_63', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8028160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 62720
    x1 = (xindex // 62720)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x0 + (63040*x1)), None)
    tmp1 = tl.load(in_ptr1 + (320 + x0 + (63040*x1)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (320 + x0 + (63040*x1)), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''')
