"""
In this script we generate data for a simple toy model and evaluates the PSIS-LOO using
the loo() function in the Arviz package.
The toy model only has a single parameter µ, the mean of the normally distributed data.
The full model definition looks as follows:

CONSTANTS / HYPER-PARAMETERS:
    µ_0 = 0         The prior mean
    σ_0 = 0.1       The prior standard deviation
    σ = 1.0         The likelihood standard deviation

PRIOR:
    µ ~ Normal(µ_0, σ_0)

LIKELIHOOD:
    X | µ ~ Normal(µ, σ)

"""
import numpy as np
from scipy import stats
import arviz as az

N_CHAINS = 2
N_DATA_SAMPLES = 10
N_POSTERIOR_SAMPLES = 200

# Generate a mean from a standard normal prior
mu_prior = 0
std_prior = 0.1
precision_prior = std_prior ** (-2)
prior = stats.norm(loc=mu_prior, scale=std_prior)
mu = prior.rvs()

# Generate data from a normal distribution with mean mu
precision_lh = std_lh = 1.0
true_likelihood = stats.norm(loc=mu, scale=std_lh)
data = true_likelihood.rvs(size=N_DATA_SAMPLES)

"""
Generate exact posterior samples for mu. The posterior can be computed analytically as:
    µ_emp = Σ X / n
    µ_pos = µ_emp * n / σ² + µ_0 / σ_0²) / (n/σ² + σ_0²)
    σ_pos =  (n/σ² + 1/σ_0²) ^ (-0.5)
    µ | X ~ Normal(µ_pos, σ_pos)
"""
mu_emp = np.mean(data, axis=0)
mu_post = (N_DATA_SAMPLES * precision_lh * mu_emp + precision_prior * mu_prior) / (N_DATA_SAMPLES*precision_lh + precision_prior)
precision_post = (N_DATA_SAMPLES * precision_lh) + precision_prior
std_post = precision_post ** (-0.5)
posterior_distr = stats.norm(loc=mu_post, scale=std_post)

posterior_samples = posterior_distr.rvs(size=N_POSTERIOR_SAMPLES)
posterior_predictive_distr = stats.norm(loc=posterior_samples[:, np.newaxis], scale=std_lh)
lpd = posterior_predictive_distr.logpdf(data[np.newaxis, :])

# # Evaluate log-likelihood according to the same distribution
independent_loo = np.sum(lpd, axis=-1)
print(np.log(np.mean(np.exp(independent_loo))))

# Create an InferenceData object. We put the samples in the posterior group (but I think
# they should not affect the result of loo) and the log_likelihood in a separate group.
idata = az.InferenceData(
    posterior=az.convert_to_dataset(posterior_samples.T),
    log_likelihood=az.convert_to_dataset(lpd.T)
)

# Now we can use the loo() function in Arviz
loo = az.loo(idata, pointwise=True)
