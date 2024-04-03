

# Original file: ./densenet121___60.0/densenet121___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3a/c3aiqmkuwe4oybsj2chfyqgvj4q53c2w3lkufd7hivk27qgd4vos.py
# Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
triton_poi_fused_cat_67 = async_compile.triton('triton_poi_fused_cat_67', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_67', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_67(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (704*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (736*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (768*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (800*x1)), tmp0, None)
''')
