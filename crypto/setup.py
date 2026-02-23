from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os
from pathlib import Path

def find_gmp_paths():
    """Auto-detect GMP installation paths on Windows."""
    include_dirs = []
    library_dirs = []
    
    if sys.platform == "win32":
        # Try conda environment first
        conda_prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
        conda_include = Path(conda_prefix) / "Library" / "include"
        conda_lib = Path(conda_prefix) / "Library" / "lib"
        
        if conda_include.exists() and conda_lib.exists():
            print(f"Found GMP in Conda: {conda_prefix}")
            include_dirs = [str(conda_include)]
            library_dirs = [str(conda_lib)]
        else:
            # Try MSYS2
            msys_paths = [
                Path("C:/msys64/mingw64"),
                Path("C:/msys64/ucrt64"),
                Path(os.environ.get('MSYSTEM_PREFIX', '')) if os.environ.get('MSYSTEM_PREFIX') else None
            ]
            
            for msys_path in msys_paths:
                if msys_path and msys_path.exists():
                    msys_include = msys_path / "include"
                    msys_lib = msys_path / "lib"
                    if msys_include.exists() and msys_lib.exists():
                        print(f"Found GMP in MSYS2: {msys_path}")
                        include_dirs = [str(msys_include)]
                        library_dirs = [str(msys_lib)]
                        break
            
            if not include_dirs:
                print("WARNING: GMP not found automatically.")
                print("Please install GMP:")
                print("  - Via conda: conda install -c conda-forge gmp")
                print("  - Via MSYS2: pacman -S mingw-w64-x86_64-gmp")
                print("Or set paths manually in setup.py")
                # Fallback to default MSYS2 paths
                include_dirs = ["C:/msys64/mingw64/include"]
                library_dirs = ["C:/msys64/mingw64/lib"]
    else:
        # Linux/Mac - standard locations
        include_dirs = ["/usr/include", "/usr/local/include"]
        library_dirs = ["/usr/lib", "/usr/local/lib"]
    
    return include_dirs, library_dirs

# Determine GMP library paths
include_dirs, library_dirs = find_gmp_paths()
libraries = ["gmp", "gmpxx"]

import pybind11

ext_modules = [
    Extension(
        "pyot",
        sources=["ot_binding.cpp", "priority_ot.cpp"],
        include_dirs=include_dirs + [pybind11.get_include()],
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=["-std=c++14", "-O3"],
        language="c++",
    ),
]

setup(
    name="pyot",
    version="1.0.0",
    author="Terrarium Team",
    description="Priority OT protocol for privacy-preserving slot intersection",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
