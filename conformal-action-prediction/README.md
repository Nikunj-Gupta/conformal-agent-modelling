
# conformal-agent-modelling version tracking: 

(Rapidly) changing conformal action modeling versions / architectures: 

- `conformal-rl.py`: 
  - ego-agent: o_self + conformal set for o_other --> a_self 
  - cam-agent: o_other + a_other --> conformal set of a_other 
  - other-agent: o_other --> a_other; learning simultaneously 

- `conformal-rl-2.py`: 
  - ego-agent: o_self + o_other + conformal set for o_other --> a_self 
  - cam-agent: o_other + a_other --> conformal set of a_other 
  - other-agent: o_other --> a_other; learning simultaneously 

- `conformal-rl-3.py` (work-in-progress): 
  - ego-agent: o_self + o_other + conformal set for o_other --> a_self 
  - cam-agent: k-stacked o_other + a_other --> conformal set of a_other 
  - other-agent: o_other --> a_other; learning simultaneously 

- `conformal-rl-4.py` (thought-in-progress): 
  - ego-agent: o_self + conformal set for o_other --> a_self 
  - cam-agent: o_self + a_other --> conformal set of a_other 
  - other-agent: o_other --> a_other; learning simultaneously 

- `conformal-rl-5.py` (thought-in-progress): 
  - ego-agent: o_self + reconstructed_o_other + conformal set for o_other --> a_self 
  - cam-agent: o_self + a_other --> conformal set of a_other 
  - other-agent: o_other --> a_other; learning simultaneously 