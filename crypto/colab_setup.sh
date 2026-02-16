#!/bin/bash
# Google Colab OT Setup Script
# Run this in Colab to install and build OT module

set -e  # Exit on error

echo "========================================================"
echo "Terrarium OT Module - Colab Setup"
echo "========================================================"

# Install GMP library
echo ""
echo "Step 1/4: Installing GMP library..."
apt-get update -qq
apt-get install -y libgmp-dev
echo "✓ GMP installed"

# Install Python dependencies
echo ""
echo "Step 2/4: Installing Python build tools..."
pip install -q pybind11 setuptools
echo "✓ Build tools installed"

# Build OT module
echo ""
echo "Step 3/4: Building OT module..."
cd crypto
python setup.py build_ext --inplace 2>&1 | grep -v "warning"
python setup.py install 2>&1 | grep -v "warning"
cd ..
echo "✓ OT module built"

# Verify installation
echo ""
echo "Step 4/4: Verifying installation..."
python -c "
from crypto import OT_AVAILABLE
if OT_AVAILABLE:
    print('✓ OT Module ready!')
    print('  Privacy-preserving intersection enabled')
else:
    print('✗ OT Module not available')
    print('  Will use fallback mode')
"

echo ""
echo "========================================================"
echo "Setup Complete!"
echo "========================================================"
echo ""
echo "Usage:"
echo "  from crypto import compute_private_intersection"
echo "  result = compute_private_intersection([1,2,3], [2,3,4])"
echo ""
