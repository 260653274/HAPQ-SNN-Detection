module time_step_engine #(
    parameter integer T_STEPS = 4,
    parameter integer CNT_WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    output reg running,
    output reg done,
    output reg [CNT_WIDTH-1:0] t_idx
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            t_idx <= {CNT_WIDTH{1'b0}};
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                running <= 1'b1;
                t_idx <= {CNT_WIDTH{1'b0}};
            end else if (running) begin
                if (t_idx == T_STEPS - 1) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end else begin
                    t_idx <= t_idx + 1'b1;
                end
            end
        end
    end
endmodule
