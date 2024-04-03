

# Original file: ./BlenderbotForCausalLM__55_forward_170.13/BlenderbotForCausalLM__55_forward_170.13_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/wn/cwnkgxm3qoer2eqhhy2fs2lqikjvgwf5w3p74iclifkhw7dqt6n3.py
# Source Nodes: [add_2, dropout_3], Original ATen: [aten.add, aten.native_dropout]
# add_2 => add_7
# dropout_3 => gt_1, mul_10, mul_11
triton_poi_fused_add_native_dropout_7 = async_compile.triton('triton_poi_fused_add_native_dropout_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_dropout_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp7 = tmp5.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''')
