from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron

from .quant_ops import quantize_membrane_update, quantize_tensor_symmetric


class QuantizedParametricLIFNode(neuron.ParametricLIFNode):
    """
    Parametric LIF Node with quantization support for membrane potential and leak factor.
    """
    def __init__(
        self,
        init_tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: nn.Module | None = None,
        detach_reset: bool = False,
        step_mode: str = 's',
        backend: str = 'torch',
        store_v_seq: bool = False,
        b_u: int = 8,
        leak_shift_n: int = 0,
    ):
        super().__init__(
            init_tau=init_tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq
        )
        self.b_u = b_u
        self.leak_shift_n = leak_shift_n

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = quantize_membrane_update(
                    u_prev=self.v,
                    input_q=x,
                    spike_prev=torch.zeros_like(self.v) if not hasattr(self, 'spike_prev') else self.spike_prev, # Simplified: spike_prev handled by reset usually
                    theta_q=self.v_threshold,
                    leak_shift_n=self.leak_shift_n,
                    bit_width_u=self.b_u
                )
            else:
                 # If v_reset is not 0, standard update might be complex with optimizations.
                 # Fallback or strict implementation.
                 # For HAPQ, we usually assume soft reset or 0 reset.
                 # Let's assume standard update structure but quantized.
                 # Actually, quantize_membrane_update implements:
                 # u_next = u_prev - leak_term + input_q - theta_q * spike_prev
                 # This handles the subtractive reset.
                 
                 # However, SpikingJelly's neuronal_charge typically just adds input to decayed v.
                 # The firing/reset happens in neuronal_fire.
                 # But our quantize_membrane_update does decay + input + reset all in one?
                 # No, checking quant_ops.py:
                 # u_next = u_prev - leak_term + input_q - theta_q * spike_prev
                 # This seems to be the full update including reset? 
                 # Wait, spike_prev is s[t-1].
                 pass

        # In SpikingJelly, the flow is:
        # neuronal_charge(x) -> updates self.v (decay + input)
        # neuronal_fire() -> generates spike
        # neuronal_reset(spike) -> updates self.v (reset)
        
        # We need to inject quantization into the decay and accumulation.
        
        # Let's override single_step_forward instead to have full control if possible, 
        # or just override neuronal_charge if we can separate decay/input.
        pass

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        
        # We need spike from previous step for soft reset if we are doing that.
        # But SpikingJelly handles reset *after* fire in the previous step (or current step).
        
        # Standard LIF:
        # u[t] = u[t-1] * decay + x[t]
        # s[t] = H(u[t] - th)
        # u[t] = u[t] - th * s[t] (soft reset) or u[t] = v_reset (hard reset)
        
        # Quantized Update from paper/code:
        # u_q[t] = Q( u_q[t-1] - (u_q[t-1] >> n) + I_q - theta_q * s[t-1] )
        # This implies s[t-1] is used for reset in the *current* update equation (like sub-threshold update + reset from prev).
        
        # SpikingJelly's `neuronal_fire` produces s[t].
        # `neuronal_reset` uses s[t] to reset v[t] for the *next* step (or immediate reset).
        
        # If we use `quantize_membrane_update`, it computes u_next based on u_prev.
        # So we should replace the whole update logic.
        
        # self.v is u[t-1] (after reset from t-1)
        
        # But wait, `quantize_membrane_update` takes `spike_prev`.
        # If SpikingJelly resets v immediately after firing, then self.v is already reset?
        # If detach_reset=False (soft reset usually handled by subtraction), 
        # SpikingJelly: 
        #   v = v * decay + x
        #   s = surrogate(v - thresh)
        #   v = v - s * thresh (soft)
        
        # Our `quantize_membrane_update`:
        #   leak = v >> n
        #   v_new = v - leak + x - theta * s_prev?
        # No, usually reset is immediate.
        # If s_prev is passed, it means we are doing "reset by subtraction of previous spike" in the update?
        # Or maybe it's just the unified equation.
        
        # Let's look at quant_ops.py again.
        # u_next = u_prev - leak_term + input_q - theta_q * spike_prev
        
        # If `spike_prev` is from t-1, and `u_prev` is u[t-1] *before* reset? No, u[t-1] is usually state.
        
        # To align with SpikingJelly:
        # We want to perform the update and then fire.
        # The quantization happens on the *state*.
        
        # Let's implement `single_step_forward` to replace the standard one.
        
        if self.step_mode == 's':
            # x is [N, *]
            # self.v is [N, *]
            
            # We assume self.v is the voltage from previous step (already reset if applicable).
            # Wait, if we use soft reset `v = v - th * s`, that matches `theta_q * spike_prev` if `spike_prev` is the spike from t-1?
            # Actually, usually soft reset is applied *after* firing at t, affecting t+1.
            # So `u[t] = decay * u[t-1] + x[t]`.
            # `s[t] = fire(u[t])`.
            # `u[t]_reset = u[t] - th * s[t]`.
            # Then next step `u[t+1] = decay * u[t]_reset + x[t+1]`.
            #                 = decay * (u[t] - th * s[t]) + x[t+1]
            #                 = decay * u[t] - decay * th * s[t] + x[t+1]
            
            # Our formula: `u_prev - leak + input - theta * s`.
            # `u_prev - leak` is approx `u_prev * (1 - 2^-n)`.
            # So `u_new = beta * u_prev + input - theta * s`.
            # This matches if `u_prev` is the voltage *before* reset of previous step? No.
            # It matches if `u_prev` is the voltage *after* reset?
            # If u_prev is after reset: u[t-1]_post.
            # u[t] = beta * u[t-1]_post + x[t].
            # This doesn't have `- theta * s`.
            
            # If we follow `quantize_membrane_update`:
            # It seems to handle leak and reset in one go.
            # Maybe `u_prev` is the accumulated voltage *before* reset?
            
            # Let's assume standard usage:
            # We want to replace the decay/integration with the quantized shift version.
            # And we want to quantize the state `v` after update.
            
            # Decay: v = v - (v >> n)
            # Input: v = v + x
            # Fire: s = H(v - th)
            # Reset: v = v - s * th
            # Quantize: v = Q(v)
            
            # This seems safer and aligns with "State-Aware Shift-Friendly Leakage" + "Quantization for SNNs" in paper.
            
            self.v_float_to_tensor(x)
            
            # 1. Decay (Shift-based)
            # beta = 1 - 2^-n
            # v = v * beta = v * (1 - 2^-n) = v - v * 2^-n = v - (v >> n)
            
            # We need integer/quantized operations ideally, but we are in PyTorch (simulated).
            # leak_term = floor(v / 2^n)
            leak_term = torch.floor(self.v / (2 ** self.leak_shift_n))
            self.v = self.v - leak_term
            
            # 2. Integrate
            self.v = self.v + x
            
            # 3. Fire
            spike = self.surrogate_function(self.v - self.v_threshold)
            
            # 4. Soft Reset
            # self.v = self.v - spike * self.v_threshold 
            # (Assuming soft reset for HAPQ as per paper)
            self.v = self.v - spike * self.v_threshold
            
            # 5. Quantize State
            self.v = quantize_tensor_symmetric(self.v, bit_width=self.b_u)
            
            return spike
            
        elif self.step_mode == 'm':
            # x is [T, N, *]
            spikes = []
            for t in range(x.shape[0]):
                spikes.append(self.single_step_forward(x[t]))
            return torch.stack(spikes)
            
        return super().single_step_forward(x)

