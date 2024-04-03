

# Original file: ./OPTForCausalLM__43_forward_134.9/OPTForCausalLM__43_forward_134.9.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/y3/cy3ig43ydq7gcwi7lmbsd4ntak5b7pd2z3rtjrl4va5x546czcot.py
# Source Nodes: [add_2, dropout_2, view_9], Original ATen: [aten.add, aten.native_dropout, aten.view]
# add_2 => add_6
# dropout_2 => gt_1, mul_7, mul_8
# view_9 => view_19
triton_poi_fused_add_native_dropout_view_7 = async_compile.triton('triton_poi_fused_add_native_dropout_view_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_dropout_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp8 = tl.load(in_ptr3 + (x0), None)
    tmp14 = tl.load(in_out_ptr0 + (x0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 + tmp11
    tmp13 = tmp4.to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 * tmp10
    tmp17 = tmp12 + tmp16
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')
