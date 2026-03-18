`include "params.vh"

module snn_top #(
    parameter integer DATA_WIDTH = `HAPQ_DATA_WIDTH,
    parameter integer U_WIDTH = `HAPQ_U_WIDTH,
    parameter integer BLOCKS = `HAPQ_BLOCKS,
    parameter integer T_STEPS = `HAPQ_T_STEPS,
    parameter integer ACC_WIDTH = 48
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [BLOCKS-1:0] mask_bits,
    input wire [BLOCKS*DATA_WIDTH-1:0] act_in,
    input wire [BLOCKS*DATA_WIDTH-1:0] weight_in,
    input wire signed [U_WIDTH-1:0] theta_q,
    output wire done,
    output wire signed [U_WIDTH-1:0] u_out,
    output wire signed [ACC_WIDTH-1:0] mac_debug
);
    wire running;
    wire [7:0] t_idx;
    wire [BLOCKS-1:0] valid_out;
    wire signed [ACC_WIDTH-1:0] mac_out;
    reg signed [U_WIDTH-1:0] u_prev;
    wire signed [U_WIDTH-1:0] u_next;

    // Output assignment to prevent optimization
    assign u_out = u_prev;
    assign mac_debug = mac_out;

    time_step_engine #(.T_STEPS(T_STEPS), .CNT_WIDTH(8)) u_time_step_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .running(running),
        .done(done),
        .t_idx(t_idx)
    );

    layer_pipeline #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .BLOCKS(BLOCKS)) u_layer_pipeline (
        .clk(clk),
        .rst_n(rst_n),
        .mask_bits(mask_bits),
        .valid_in({BLOCKS{running}}),
        .act_in(act_in),
        .weight_in(weight_in),
        .valid_out(valid_out),
        .mac_out(mac_out)
    );

    membrane_update #(.U_WIDTH(U_WIDTH), .TH_WIDTH(U_WIDTH), .SHIFT_N(`HAPQ_SHIFT_N)) u_membrane_update (
        .clk(clk),
        .rst_n(rst_n),
        .u_prev(u_prev),
        .input_q(mac_out[U_WIDTH-1:0]),
        .theta_q(theta_q),
        .spike_prev(1'b0),
        .u_next(u_next)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            u_prev <= {U_WIDTH{1'b0}};
        end else begin
            u_prev <= u_next;
        end
    end
endmodule
