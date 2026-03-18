set proj_name "eas_snn_hapq"
set proj_dir [file normalize "./fpga/vivado/${proj_name}"]
set part_name [expr {[info exists ::env(FPGA_PART)] ? $::env(FPGA_PART) : "xczu7ev-ffvc1156-2-e"}]

file mkdir $proj_dir
create_project $proj_name $proj_dir -part $part_name -force

# Add sources
set rtl_files [glob -nocomplain ./fpga/rtl/*.v]
set vh_files [glob -nocomplain ./fpga/rtl/*.vh]
if {[llength $rtl_files] == 0} {
    puts "Error: No RTL files found in ./fpga/rtl/"
    exit 1
}
add_files [concat $rtl_files $vh_files]
set_property top snn_top [current_fileset]
update_compile_order -fileset sources_1

# Set Out-of-Context mode to avoid I/O pin constraints and IBUF/OBUF insertion
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]

# Synthesis
puts "Running Synthesis..."

# Create constraints file before synthesis
file mkdir "${proj_dir}"
set xdc_file [open "${proj_dir}/constraints.xdc" w]
puts $xdc_file "create_clock -period 5.000 -name clk \[get_ports clk\]"
close $xdc_file
add_files -fileset constrs_1 "${proj_dir}/constraints.xdc"

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "Error: Synthesis failed"
    exit 1
}

# Implementation
puts "Running Implementation..."
# Add constraints again for implementation if needed, or create XDC
set_property TARGET_CONSTRS_FILE "${proj_dir}/constraints.xdc" [current_fileset -constrset]

launch_runs impl_1 -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "Error: Implementation failed"
    exit 1
}
open_run impl_1

# Report Implementation Results (More accurate)
report_utilization -file "${proj_dir}/utilization_impl.rpt"
report_timing_summary -file "${proj_dir}/timing_impl.rpt"
report_power -file "${proj_dir}/power.rpt"

puts "FPGA Flow Completed Successfully."
close_project
exit
