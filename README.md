# AIA2XRT

data-preapare:
aia_process.pro   pre-process (aia_prep.pro, divided by exposure time) full-disk AIA raw data downloaded from JSOC(http://jsoc.stanford.edu)
xrt_process.pro   pre-process(use xrt_read_coaldb.pro) full-Sun XRT data downloaded from SCIA (http://solar.physics.montana.edu/HINODE/XRT/SCIA)
heap_xrt_aia.pro  clip and downsample each AIA image to have the same FOV and spatial resolution as XRT, and stack them as 7x1024x1024 data cube.
heap6aia.pro  stack 6-channels (171,193,211,335,131,94) AIA images as 6x4096x4096 data cube, input into xrt4096_generator.py

deep learning:
aia2xrt_train.py  trainning code based on datacube produced by heap_xrt_aia.pro
aia2xrt_test.py test code based on datacube produced by heap_xrt_aia.pro

*.mod  CNN models trained out by aia2xrt_train.py
resnet1.py  CNN neural network code of "Resnet"

XRT-like data produced by CNNmodels:
xrt4096_generator.py
xrt1024_generator.py
