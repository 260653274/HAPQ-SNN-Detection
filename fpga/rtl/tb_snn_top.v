`timescale 1ns/1ps
`include "params.vh"

module tb_snn_top;
    localparam integer DATA_WIDTH = `HAPQ_DATA_WIDTH;
    localparam integer U_WIDTH = `HAPQ_U_WIDTH;
    localparam integer BLOCKS = `HAPQ_BLOCKS;
    localparam integer T_STEPS = `HAPQ_T_STEPS;
    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;
    reg start;
    reg [BLOCKS-1:0] mask_bits;
    reg signed [DATA_WIDTH-1:0] act_in;
    reg signed [DATA_WIDTH-1:0] weight_in;
    reg signed [U_WIDTH-1:0] theta_q;
    wire done;

    snn_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .U_WIDTH(U_WIDTH),
        .BLOCKS(BLOCKS),
        .T_STEPS(T_STEPS)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mask_bits(mask_bits),
        .act_in(act_in),
        .weight_in(weight_in),
        .theta_q(theta_q),
        .done(done)
    );

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        rst_n = 0;
        start = 0;
        mask_bits = {BLOCKS{1'b1}};
        act_in = 0;
        weight_in = 0;
        theta_q = 512;

        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        start = 1;
        act_in = 8'sd10;
        weight_in = 8'sd5;
        #CLK_PERIOD;
        start = 0;

        wait (done);
        #(CLK_PERIOD * 2);

        $display("[TB] Simulation finished: done asserted at %0t", $time);
        $finish;
    end
endmodule
