#!/usr/bin/env python3
import os
import tempfile
import numpy as np

from dsl.compiler import DSLCompiler


def test_dsl_windows_roundtrip():
    compiler = DSLCompiler()

    dsl_obj = {
        "length": [230, 330],
        "motifs": [
            {"name": "lipase_gxsxg", "dfa": "G X S X G", "window": [50, 90]}
        ],
        "tags": ["pH~7"],
    }

    compiled = compiler.compile_to_constraints(dsl_obj)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "constraints.npz")
        compiler.save_compiled(compiled, out)
        loaded = compiler.load_compiled(out)

    assert np.array_equal(loaded["windows"], np.array([[50, 90]], dtype=np.int32)), (
        loaded["windows"],
    )


if __name__ == "__main__":
    test_dsl_windows_roundtrip()
    print("ok")
