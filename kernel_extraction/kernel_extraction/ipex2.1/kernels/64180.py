

# Original file: ./Speech2Text2ForCausalLM__21_inference_61.1/Speech2Text2ForCausalLM__21_inference_61.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ey/ceysqbwh4lg3olzdpd2cvre4qcsi3kueq3haqvg2cob2i57dxu3h.py
# Source Nodes: [detach], Original ATen: [aten.detach]
# detach => alias
triton_poi_fused_detach_1 = async_compile.triton('triton_poi_fused_detach_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_detach_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tl.where(tmp10 < 0, tmp10 + 1026, tmp10)
    # tl.device_assert((0 <= tmp11) & (tmp11 < 1026), "index out of bounds: 0 <= tmp11 < 1026")
    tmp12 = tl.load(in_ptr2 + (x0 + (256*tmp11)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp12, None)
''')
