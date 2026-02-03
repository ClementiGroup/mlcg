Priors
======

`mlcg.nn` implements prior models that are important physical baselines imposing geometrical constraints on coarse-grained systems. These prior models can be supplied with a user-defined set of interaction parameters, or they can alternatively be parametrized directly from data for multiple interacting atom types.


These classes define several common molecular interactions. Because each prior subclasses `torch.nn.Module`, they can be treated as normal property predictors. 

Harmonic
---------
.. autoclass:: mlcg.nn.prior.Harmonic
.. autoclass:: mlcg.nn.prior.HarmonicBonds
.. autoclass:: mlcg.nn.prior.HarmonicAngles

Polynomial
-----------
.. autoclass:: mlcg.nn.prior.Polynomial
.. autoclass:: mlcg.nn.prior.QuarticAngle

Fourier Series 
---------------
.. autoclass:: mlcg.nn.prior.FourierSeries
.. autoclass:: mlcg.nn.prior.Dihedral

Excluded volume
----------------
.. autoclass:: mlcg.nn.prior.Repulsion
