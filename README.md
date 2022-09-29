**This code is based on the original RNS code written by Nicolaos Stergioulas, with improvements for quark stars.**

### Usage

```python
from RNS import RNS
from quark_eos import quark_eos

rns = RNS(MDIV=65,SDIV=201)
rns.eos = quark_eos(e0=.3, e1=.7, cons="Maxwell", eos="eosSLy")

print(rns.spin(ec=2.,r_ratio=.9))
