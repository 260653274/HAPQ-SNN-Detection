set proj_name "eas_snn_hapq"
set proj_dir [file normalize "./fpga/vivado/${proj_name}"]
set part_name [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xczu3eg-sbva484-1-e"}]

file mkdir $proj_dir
create_project $proj_name $proj_dir -part $part_name -force

set rtl_files [glob -nocomplain ./fpga/rtl/*.v]
set vh_files [glob -nocomplain ./fpga/rtl/*.vh]
add_files [concat $rtl_files $vh_files]
set_property top snn_top [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1

report_utilization -file "${proj_dir}/utilization_synth.rpt"
report_timing_summary -file "${proj_dir}/timing_synth.rpt"

close_project
exit
