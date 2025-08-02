#!/bin/bash

echo "Testing NASA Mars Trek API endpoints..."
echo "=========================================="

# Test Viking Color Mosaic (our primary layer)
echo "Testing Viking Color Mosaic at zoom 2, tile 1,1..."
curl -s -I "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/2/1/1.jpg" | head -1

# Test MOLA Color Hillshade
echo "Testing MOLA Color Hillshade at zoom 2, tile 1,1..."
curl -s -I "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/2/1/1.jpg" | head -1

# Test THEMIS Day IR
echo "Testing THEMIS Day IR at zoom 2, tile 1,1..."
curl -s -I "https://trek.nasa.gov/tiles/Mars/EQ/THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm/2/1/1.jpg" | head -1

echo "=========================================="
echo "If you see 'HTTP/1.1 200 OK' above, the NASA APIs are working!"
echo "The Mars GIS app should now display actual Mars surface imagery instead of orange background."
