module quantizer #(
    parameter integer IN_WIDTH = 24,
    parameter integer OUT_WIDTH = 8
) (
    input wire signed [IN_WIDTH-1:0] din,
    output reg signed [OUT_WIDTH-1:0] dout
);
    localparam signed [OUT_WIDTH-1:0] QMAX = {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam signed [OUT_WIDTH-1:0] QMIN = {1'b1, {(OUT_WIDTH-1){1'b0}}};

    always @(*) begin
        if (din > QMAX) begin
            dout = QMAX;
        end else if (din < QMIN) begin
            dout = QMIN;
        end else begin
            dout = din[OUT_WIDTH-1:0];
        end
    end
endmodule
