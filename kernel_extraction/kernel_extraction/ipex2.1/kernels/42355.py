

# Original file: ./OPTForCausalLM__52_forward_161.12/OPTForCausalLM__52_forward_161.12_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/vp/cvp3kdqiqmy55pbqnwmx4w5njl5rp23fjbvcvxnzzcixym3kthto.py
# Source Nodes: [add_2, dropout_2, view_9], Original ATen: [aten.add, aten.native_dropout, aten.view]
# add_2 => add_6
# dropout_2 => gt_1, mul_7, mul_8
# view_9 => view_19
triton_poi_fused_add_native_dropout_view_11 = async_compile.triton('triton_poi_fused_add_native_dropout_view_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_dropout_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp6 + tmp13
    tmp15 = tmp5.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 * tmp11
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp14 + tmp19
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''')
