"""
Compute the PSIS-LOO score for logged observation likelihood values stored in a likelihood.h5 file.
"""
import tables
import numpy as np
import arviz as az

# Load the data into a numpy array (shape = n_samples*n_observations)
likelihood_table = tables.open_file('data/likelihood.h5', mode='r')
likelihood_np = likelihood_table.root.likelihood[:]
likelihood_table.close()

# arviz interprets the first dimension as chains and the second as samples, but the
# likelihood in the file is only for one chain, i.e. dimesnions start with samples.
# => Append a new dimension for chains!
likelihood_np = likelihood_np[np.newaxis, ...]

# Create an InferenceData object
data = az.convert_to_inference_data(np.log(likelihood_np))

# Per default, the data is stored in the group "posterior", but az.loo() expects InferenceData with a "log_likelihood" group.
# We can manually add that group (as a copy of the posterior):
data.add_groups({'log_likelihood': data.posterior})

# Now az.loo() should work:
loo = az.loo(data)

print(loo)
