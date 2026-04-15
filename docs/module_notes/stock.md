# stock

Owns the exact Draw-3 stock/waste cycle. The stock order is known from the beginning, so this module should remain deterministic.

The stock ring is normalized as stock prefix followed by waste bottom-to-top. `stock_len`, `cursor`, and `accessible_depth` describe Draw-3 accessibility over the fully known order.

This module provides the primitive stock advance, accessible-waste removal, and recycle operations used by visible-state move application. Higher-level stock macros and target advancement remain future work.
