

# Original file: ./MBartForConditionalGeneration__105_forward_324.25/MBartForConditionalGeneration__105_forward_324.25_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2m/c2mpyev6bvnyvdgh4fapsycb7ljogbspkrwcte7yhsrd2jh5bzya.py
# Source Nodes: [add_1, add_2, add_3, dropout_5], Original ATen: [aten.add, aten.native_dropout]
# add_1 => add_3
# add_2 => add_6
# add_3 => add_10
# dropout_5 => gt_2, mul_15, mul_16
triton_poi_fused_add_native_dropout_14 = async_compile.triton('triton_poi_fused_add_native_dropout_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_dropout_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp5.to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.1111111111111112
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp12 + tmp18
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp19, None)
''')
