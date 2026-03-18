# EAS-SNN HAPQ Vivado Simulation (Batch Mode)
# Run from project root: vivado -mode batch -source fpga/scripts/sim.tcl

set proj_name "eas_snn_hapq"
set proj_dir [file normalize "./fpga/vivado/${proj_name}"]
set part_name [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xc7a35tcpg236-1"}]

file mkdir $proj_dir
create_project $proj_name $proj_dir -part $part_name -force

# RTL sources (exclude testbench)
set vh_files [glob -nocomplain ./fpga/rtl/*.vh]
set all_v [glob -nocomplain ./fpga/rtl/*.v]
set rtl_files {}
foreach f $all_v {
    if {![string match *tb* $f]} { lappend rtl_files $f }
}
add_files [concat $rtl_files $vh_files]

# Testbench (simulation only)
add_files -norecurse -fileset sim_1 ./fpga/rtl/tb_snn_top.v
set_property top tb_snn_top [get_filesets sim_1]
update_compile_order -fileset sim_1

# Run behavioral simulation
launch_simulation -mode behavioral
run 1us
close_sim -force

puts "Simulation finished successfully."
