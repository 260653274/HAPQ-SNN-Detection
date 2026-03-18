module block_gate #(
    parameter integer BLOCKS = 16
) (
    input wire [BLOCKS-1:0] mask_bits,
    input wire [BLOCKS-1:0] valid_in,
    output wire [BLOCKS-1:0] valid_out
);
    assign valid_out = valid_in & mask_bits;
endmodule
