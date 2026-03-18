`include "params.vh"

module membrane_update #(
    parameter integer U_WIDTH = `HAPQ_U_WIDTH,
    parameter integer TH_WIDTH = `HAPQ_U_WIDTH,
    parameter integer SHIFT_N = `HAPQ_SHIFT_N
) (
    input wire clk,
    input wire rst_n,
    input wire signed [U_WIDTH-1:0] u_prev,
    input wire signed [U_WIDTH-1:0] input_q,
    input wire signed [TH_WIDTH-1:0] theta_q,
    input wire spike_prev,
    output reg signed [U_WIDTH-1:0] u_next
);
    wire signed [U_WIDTH-1:0] leak_term;
    wire signed [U_WIDTH-1:0] reset_term;
    wire signed [U_WIDTH:0] accum;

    assign leak_term = u_prev >>> SHIFT_N;
    assign reset_term = spike_prev ? theta_q[U_WIDTH-1:0] : {U_WIDTH{1'b0}};
    assign accum = u_prev - leak_term + input_q - reset_term;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            u_next <= {U_WIDTH{1'b0}};
        end else begin
            u_next <= accum[U_WIDTH-1:0];
        end
    end
endmodule
