module layer_pipeline #(
    parameter integer DATA_WIDTH = 32,
    parameter integer ACC_WIDTH = 48,
    parameter integer BLOCKS = 32
) (
    input wire clk,
    input wire rst_n,
    input wire [BLOCKS-1:0] mask_bits,
    input wire [BLOCKS-1:0] valid_in,
    input wire [BLOCKS*DATA_WIDTH-1:0] act_in,
    input wire [BLOCKS*DATA_WIDTH-1:0] weight_in,
    output wire [BLOCKS-1:0] valid_out,
    output wire signed [ACC_WIDTH-1:0] mac_out
);
    block_gate #(.BLOCKS(BLOCKS)) u_block_gate (
        .mask_bits(mask_bits),
        .valid_in(valid_in),
        .valid_out(valid_out)
    );

    reg signed [ACC_WIDTH-1:0] partial_sums [BLOCKS-1:0];
    reg signed [ACC_WIDTH-1:0] total_sum;
    integer i;

    always @(*) begin
        total_sum = 0;
        for (i = 0; i < BLOCKS; i = i + 1) begin
            // Extract signed inputs and multiply
            partial_sums[i] = $signed(act_in[i*DATA_WIDTH +: DATA_WIDTH]) * $signed(weight_in[i*DATA_WIDTH +: DATA_WIDTH]);
            
            // Accumulate if mask bit is set (or just sum all if BLOCKS is already active count)
            // Using mask_bits allows runtime clock gating or operand isolation if synthesized
            if (mask_bits[i]) begin
                total_sum = total_sum + partial_sums[i];
            end
        end
    end

    assign mac_out = total_sum;

endmodule
