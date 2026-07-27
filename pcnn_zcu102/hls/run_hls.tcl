# Vitis HLS build script -- ZCU102 (XCZU9EG)
#   vitis_hls -f run_hls.tcl            (Vitis HLS 2022.x/2023.x)
open_project -reset pcnn_hls
set_top pcnn_step
add_files pcnn.cpp -cflags "-Iweights -I."
add_files -tb pcnn_tb.cpp -cflags "-Iweights -I."
add_files -tb tb_data

open_solution -reset "sol1"
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default    ;# 100 MHz; try 5 ns after timing closure

csim_design
csynth_design
# RTL/C co-simulation is slow with this much state; enable when needed:
# cosim_design -rtl verilog
export_design -format ip_catalog -vendor "user" -library "pcnn" -display_name "pcnn_step"
exit
