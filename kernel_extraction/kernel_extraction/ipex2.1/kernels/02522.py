

# Original file: ./MBartForConditionalGeneration__75_forward_234.19/MBartForConditionalGeneration__75_forward_234.19.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ml/cmlaiefvkxjqyaywboc2icxjthbubnkh5la3wq5onysouse32ias.py
# Source Nodes: [add_3, dropout_5], Original ATen: [aten.add, aten.native_dropout]
# add_3 => add_10
# dropout_5 => gt_2, mul_15, mul_16
triton_poi_fused_add_native_dropout_9 = async_compile.triton('triton_poi_fused_add_native_dropout_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_dropout_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_out_ptr0 + (x0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')
