"""Gated name-brand audit (spec §7): official TD-MPC2 checkpoint on DMControl.

TD-MPC2 is an implicit, decoder-free latent world model, so state-space
auditing requires a trained decoder probe; the decoder's held-out R² is the
audit's noise floor and is displayed, never hidden. Kill criterion: decoder
R² < 0.9 on observable positions → publish the negative result and drop the
report card. This scene family pokes via ctrl (the action channel), not xfrc.
"""
