

# Original file: ./PLBartForConditionalGeneration__41_inference_121.9/PLBartForConditionalGeneration__41_inference_121.9.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/wf/cwf7xdyjx5y7bmj4zoo5vmf6yvsa3akt6sueekm23vulkwc5bx45.py
# Source Nodes: [arange, full, lt, masked_fill_, to], Original ATen: [aten._to_copy, aten.arange, aten.full, aten.lt, aten.masked_fill]
# arange => iota
# full => full_default
# lt => lt
# masked_fill_ => full_default_1, where
# to => convert_element_type
triton_poi_fused__to_copy_arange_full_lt_masked_fill_0 = async_compile.triton('triton_poi_fused__to_copy_arange_full_lt_masked_fill_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_full_lt_masked_fill_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_arange_full_lt_masked_fill_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -65504.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''')
