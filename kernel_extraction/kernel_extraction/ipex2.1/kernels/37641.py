

# Original file: ./TrOCRForCausalLM__21_inference_63.1/TrOCRForCausalLM__21_inference_63.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/os/cosxpqm4rkx6uasww4gejwtof5pdba4x2qvtiqqikvxmxd2r2hof.py
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

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_full_lt_masked_fill_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_arange_full_lt_masked_fill_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = x0
    tmp1 = 1 + x1
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = -3.3895313892515355e+38
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''')
