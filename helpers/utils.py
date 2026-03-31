"""
NOT MY CODE -- minimally adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""

from evenet.dataset.preprocess import unflatten_dict, flatten_dict
import pyarrow.parquet as pq
import numpy as np
import awkward as ak
import torch, os, json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
     
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose = False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True

def save_file(
    save_dir,
    data_df,
    norm_dict,
    event_filter,
    postfix=""
):

    os.makedirs(save_dir, exist_ok=True)
    if event_filter is not None:
        if not isinstance(event_filter, np.ndarray):
            event_filter = ak.to_numpy(event_filter)
        filtered_df = {col: data_df[col][event_filter] for col in data_df}
    else:
        filtered_df = data_df
    flatten_data, meta_data = flatten_dict(filtered_df)

    ### Save to parquet
    pq.write_table(flatten_data, f"{save_dir}/data{postfix}.parquet")

    with open(f"{save_dir}/shape_metadata{postfix}.json", "w") as f:
        json.dump(meta_data, f)

    print(f"[INFO] Final table size: {flatten_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"[Saving] Saving {flatten_data.num_rows} rows to {save_dir}/data.parquet")

    torch.save(norm_dict, f"{save_dir}/normalization{postfix}.pt")

def save_df(
        save_dir,
        data_df,
        pc_index=None,
        global_index=None,
    ):
    os.makedirs(save_dir, exist_ok=True)

    dataset = dict()

    for name, index in pc_index.items():
        dataset[f"pc-{name}-0"] = data_df["x"][:, 0, index]
        dataset[f"pc-{name}-1"] = data_df["x"][:, 1, index]
    for name, index in global_index.items():
        dataset[f"{name}"] = data_df["conditions"][:, index]
        if f"{name}-pc" in data_df:
            dataset[f"{name}-pc"] = data_df[f"{name}-pc"]

    for name, data in data_df.items():
        if "pc" in name and name not in dataset:
            # If the name contains "pc" but is not in dataset, add it
            dataset[name] = data
            
    # if "eta-0" in dataset and "phi-0" in dataset and "eta-1" in dataset and "phi-1" in dataset:
    #     eta0 = dataset["eta-0"]
    #     phi0 = dataset["phi-0"]
    #     eta1 = dataset["eta-1"]
    #     phi1 = dataset["phi-1"]
    #
    #     delta_eta = np.abs(eta1 - eta0)
    #     delta_phi = phi1 - phi0
    #     delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
    #     delta_phi = np.abs(delta_phi)
    #
    #     dataset["delta-eta"] = delta_eta
    #     dataset["delta-phi"] = delta_phi
    #     dataset["delta-R"] = np.sqrt(delta_eta ** 2 + delta_phi ** 2)


    dataset["classification"] = data_df["classification"]
    df = pd.DataFrame(dataset)

    df.to_csv(f"{save_dir}/data.csv", index=False)

    print(df)

def mean_std_last_dim(x, x_mask):
    """
    Compute masked mean and std over all axes except the last one.
    Supports x of shape (a, c) or (a, b, c), and x_mask of shape (a,) or (a, b).

    Parameters:
        x (np.ndarray): Input array of shape (..., c)
        x_mask (np.ndarray): Boolean mask matching all dims except the last

    Returns:
        tuple of np.ndarray: (mean, std) with shape (1, c)
    """
    # Get shape info
    if x_mask.ndim == x.ndim - 1:

        # Broadcast mask to match x shape
        mask_expanded = np.expand_dims(x_mask, axis=-1)  # shape (..., 1)
        mask_broadcasted = np.broadcast_to(mask_expanded, x.shape)
    else:
        mask_broadcasted = x_mask

    # Create masked array
    x_masked = np.ma.masked_array(x, mask=~mask_broadcasted)

    # Compute mean and std over all axes except the last one
    axis_to_reduce = tuple(range(x.ndim - 1))
    mean = x_masked.mean(axis=axis_to_reduce, keepdims=True)
    std = x_masked.std(axis=axis_to_reduce, keepdims=True)

    return mean.filled(np.nan), std.filled(np.nan)

def pad_object(obj, nMax):
  pad_awkward = ak.pad_none(obj, target = nMax, clip = True)
  return pad_awkward
def clean_and_append(dirname, postfix):
    if dirname.endswith("/"):
        dirname = dirname[:-1]
    return dirname + postfix


def get_latest_file_in_dir(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No files found in directory: {directory}")
    return max(files, key=os.path.getmtime)



def analyze_mass_relation(inv_mass_gen,
                          inv_mass_truth,
                          extra_vars=None,  # NEW: dict of {name: array-like}
                          correlation_data = None,  # NEW: dict of {name: array-like} for correlation vs residual cuts
                          residual_cuts=None,  # NEW: list of |m_gen - m_truth| thresholds to test
                          bins=50,
                          mass_range=None,
                          figsize=(14,10),
                          log_counts=False,
                          show_relative_bias=True,
                          return_results=True,
                          plot_dir=None):
    """
    Produce an in-depth set of diagnostics and plots comparing generated invariant
    mass vs truth.

    Plots produced:
      - 2D density (gen vs truth) with diagonal and linear fit
      - Profile: <gen> (mean +/- SE) vs truth (binned)
      - Bias vs truth (mean bias +/- SE) and optional relative bias
      - Shape comparison (normalized histograms) with gen/truth ratio on a twin axis
      - Residual histogram (gen - truth) with summary stats

    Diagnostics returned (if return_results True):
      dict with keys:
        'n' : total entries
        'slope', 'intercept' : linear fit gen = slope*truth + intercept
        'pearson_r', 'r_squared'
        'mean_bias', 'median_bias', 'std_bias', 'rms'
        'ks_stat' : two-sample KS statistic (p-value if scipy available)
        'bin_table' : dict of per-bin arrays (bin_centers, n_i, mean_truth, mean_gen,
                      std_gen, mean_bias, std_bias, se_bias, rel_bias_mean)

    Parameters
    ----------
    inv_mass_gen, inv_mass_truth : array-like
        Same-shape arrays of generated and truth invariant masses (floats).
    bins : int
        Number of bins for binned diagnostics and histograms.
    mass_range : tuple (low, high) or None
        Axis range for plots. If None, computed from the data.
    figsize : tuple
        Figure size.
    log_counts : bool
        If True, show log color scaling in the 2D histogram.
    show_relative_bias : bool
        If True, plot relative bias (mean_bias / bin_center) on a secondary axis.
    return_results : bool
        If True, return a dictionary of computed diagnostics.

    Usage:
      results = analyze_mass_relation(inv_mass_gen, inv_mass_truth, bins=60, mass_range=(0,200))
    """
    # -------------------------
    # Prepare arrays / basic checks
    # -------------------------
    gen = np.asarray(inv_mass_gen).ravel()
    truth = np.asarray(inv_mass_truth).ravel()
    if gen.shape != truth.shape:
        raise ValueError("inv_mass_gen and inv_mass_truth must have the same number of elements")
    mask = np.isfinite(gen) & np.isfinite(truth)
    gen = gen[mask]; truth = truth[mask]
    n = len(gen)
    if n == 0:
        raise ValueError("No finite entries after masking.")

    bias = gen - truth
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_bias_all = bias / truth
    # avoid infinite relative biases from truth==0
    rel_bias_all[~np.isfinite(rel_bias_all)] = np.nan

    # overall metrics
    mean_bias = float(np.mean(bias))
    median_bias = float(np.median(bias))
    std_bias = float(np.std(bias))
    rms = float(np.sqrt(np.mean(bias**2)))
    pearson_r = float(np.corrcoef(truth, gen)[0,1])
    r_squared = pearson_r**2

    # linear fit gen = slope * truth + intercept
    slope, intercept = np.polyfit(truth, gen, 1)

    # KS 2-sample (try scipy; fallback to simple empirical CDF if not available)
    try:
        from scipy import stats
        ks_res = stats.ks_2samp(truth, gen)  # returns object with statistic and pvalue
        ks_stat = float(ks_res.statistic)
        ks_pvalue = float(ks_res.pvalue)
    except Exception:
        # simple fallback for KS statistic (no p-value)
        def simple_ks(a, b):
            a_sorted = np.sort(a)
            b_sorted = np.sort(b)
            allvals = np.sort(np.concatenate([a_sorted, b_sorted]))
            cdf_a = np.searchsorted(a_sorted, allvals, side='right')/len(a_sorted)
            cdf_b = np.searchsorted(b_sorted, allvals, side='right')/len(b_sorted)
            return np.max(np.abs(cdf_a - cdf_b))
        ks_stat = float(simple_ks(truth, gen))
        ks_pvalue = None

    # -------------------------
    # Binned diagnostics (binned in truth)
    # -------------------------
    if mass_range is None:
        low = min(np.min(truth), np.min(gen))
        high = max(np.max(truth), np.max(gen))
        # small padding
        span = high - low
        if span == 0:
            low -= 0.5
            high += 0.5
        else:
            pad = 0.002 * span
            low -= pad; high += pad
    else:
        low, high = float(mass_range[0]), float(mass_range[1])

    edges = np.linspace(low, high, bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(truth, edges) - 1  # bin indices in [0, bins-1]

    n_i = np.zeros(bins, dtype=int)
    mean_truth_bin = np.full(bins, np.nan)
    mean_gen_bin = np.full(bins, np.nan)
    std_gen_bin = np.full(bins, np.nan)
    mean_bias_bin = np.full(bins, np.nan)
    std_bias_bin = np.full(bins, np.nan)
    se_bias_bin = np.full(bins, np.nan)
    rel_bias_mean_bin = np.full(bins, np.nan)

    for i in range(bins):
        sel = (bin_idx == i)
        ni = int(sel.sum())
        n_i[i] = ni
        if ni == 0:
            continue
        truth_sel = truth[sel]
        gen_sel = gen[sel]
        bias_sel = gen_sel - truth_sel
        mean_truth_bin[i] = float(np.mean(truth_sel))
        mean_gen_bin[i] = float(np.mean(gen_sel))
        std_gen_bin[i] = float(np.std(gen_sel))
        mean_bias_bin[i] = float(np.mean(bias_sel))
        std_bias_bin[i] = float(np.std(bias_sel))
        se_bias_bin[i] = float(std_bias_bin[i] / np.sqrt(ni)) if ni > 0 else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_bias_mean_bin[i] = float(np.nanmean(bias_sel / truth_sel))

    # -------------------------
    # Shape histograms (normalized) and ratio
    # -------------------------
    hist_bins = edges
    truth_counts, _ = np.histogram(truth, bins=hist_bins)
    gen_counts, _ = np.histogram(gen, bins=hist_bins)
    # density (area normalized)
    truth_area = np.sum(truth_counts * np.diff(hist_bins))
    gen_area = np.sum(gen_counts * np.diff(hist_bins))
    # avoid division by zero for ratio
    safe_truth_counts = truth_counts.copy().astype(float)
    safe_truth_counts[safe_truth_counts == 0] = np.nan
    ratio_counts = gen_counts / safe_truth_counts

    # -------------------------
    # Start plotting
    # -------------------------
    fig = plt.figure(figsize=figsize)
    # use a 3x2 layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1) 2D density
    ax0 = fig.add_subplot(gs[0, 0])
    if log_counts:
        img = ax0.hist2d(truth, gen, bins=[edges, edges],
                         range=[[low, high], [low, high]],
                         norm=LogNorm(), cmap='viridis')
    else:
        img = ax0.hist2d(truth, gen, bins=[edges, edges],
                         range=[[low, high], [low, high]],
                         cmap='viridis')
    ax0.plot([low, high], [low, high], ls='--', color='red', label='y = x')
    ax0.plot([low, high], [intercept + slope*low, intercept + slope*high],
             ls='-', color='black', label=f'linear fit: gen={slope:.4f}*truth + {intercept:.4f}')
    ax0.set_xlabel("truth mass")
    ax0.set_ylabel("generated mass")
    ax0.set_title("2D: gen vs truth")
    ax0.legend(loc='upper left', fontsize='small')
    cbar = fig.colorbar(img[3], ax=ax0)
    cbar.set_label("counts")

    # 2) Profile: mean(gen) vs truth_bin_center (use mean_truth_bin where available)
    ax1 = fig.add_subplot(gs[0, 1])
    valid = ~np.isnan(mean_gen_bin)
    if valid.sum() > 0:
        ax1.errorbar(mean_truth_bin[valid], mean_gen_bin[valid],
                     yerr=(std_gen_bin[valid] / np.sqrt(np.maximum(n_i[valid], 1))),
                     marker='o', linestyle='-', label='mean(gen) +/- SE')
    ax1.plot([low, high], [low, high], ls='--', color='red', label='y = x')
    ax1.set_xlabel("truth mass (bin mean)")
    ax1.set_ylabel("mean(gen)")
    ax1.set_title("Profile: <gen> vs truth (binned)")
    ax1.legend(fontsize='small')

    # 3) Bias vs truth (binned)
    ax2 = fig.add_subplot(gs[1, 0])
    good = ~np.isnan(mean_bias_bin)
    if good.sum() > 0:
        ax2.errorbar(bin_centers[good], mean_bias_bin[good], yerr=se_bias_bin[good],
                     fmt='o-', label='mean bias +/- SE')
        ax2.axhline(0, ls='--', color='k')
    ax2.set_xlabel("truth mass (bin center)")
    ax2.set_ylabel("mean bias (gen - truth)")
    ax2.set_title("Bias vs truth (binned)")
    ax2.legend(fontsize='small')

    if show_relative_bias:
        ax2b = ax2.twinx()
        rel_plot_mask = (n_i > 0) & np.isfinite(rel_bias_mean_bin)
        if rel_plot_mask.sum() > 0:
            ax2b.plot(bin_centers[rel_plot_mask], rel_bias_mean_bin[rel_plot_mask],
                      ls='--', marker='s', label='mean relative bias')
        ax2b.set_ylabel("relative bias (mean (gen - truth)/truth)")
        # combine legends
        lines, labs = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labs + labs2, fontsize='small', loc='best')

    # 4) Shape overlay + ratio as a twin axis
    ax3 = fig.add_subplot(gs[1, 1])
    # normalized density histograms (plot as step)
    width = np.diff(hist_bins)
    # density: counts / (area) where area = sum(counts*width) but matplotlib density arg does that for us.
    ax3.hist(hist_bins[:-1], bins=hist_bins, weights=truth_counts, density=True,
             histtype='stepfilled', alpha=0.35, label='truth')
    ax3.hist(hist_bins[:-1], bins=hist_bins, weights=gen_counts, density=True,
             histtype='step', alpha=0.9, label='gen')
    ax3.set_xlabel('mass')
    ax3.set_ylabel('normalized density')
    ax3.set_title('Shape: truth vs gen (normalized)')
    ax3.legend(fontsize='small')

    # ratio on twin y-axis
    ax3b = ax3.twinx()
    # compute ratio using densities per bin center (avoid dividing by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        truth_density = truth_counts / (np.sum(truth_counts) * width)  # density per bin
        gen_density = gen_counts / (np.sum(gen_counts) * width)
        density_ratio = gen_density / truth_density
    # plot ratio at bin centers (mask NaNs)
    mask_ratio = np.isfinite(density_ratio)
    if mask_ratio.sum() > 0:
        ax3b.plot(bin_centers[mask_ratio], density_ratio[mask_ratio], ls='-.', marker='o', label='gen/truth (density)')
    ax3b.set_ylabel('gen/truth (density ratio)')
    # add combined legend
    lines, labs = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labs + labs2, fontsize='small', loc='best')

    # 5) Residual histogram
    ax4 = fig.add_subplot(gs[2, 0])
    # define residual range
    res_range = (-3.2, 3.2)

    # histogram only counts inside [-2, 2]
    ax4.hist(bias, bins=100, range=res_range, density=True, alpha=0.75)

    # overlay gaussian in same range
    xvals = np.linspace(res_range[0], res_range[1], 300)
    mu, sigma = np.mean(bias), np.std(bias)
    gauss = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((xvals - mu) / sigma) ** 2)
    ax4.plot(xvals, gauss, ls='--', label=f'Gaussian(mu={mu:.2f}, σ={sigma:.2f})')

    ax4.set_xlim(res_range)  # make sure x-axis matches

    ax4.axvline(mu, color='k', ls='--', label=f'mean={mu:.4g}')
    ax4.set_xlabel('residual = gen - truth')
    ax4.set_title('Residual distribution')
    ax4.legend(fontsize='small')


    cut_max = 1.6
    n_steps = 49
    cuts = np.linspace(0, cut_max, n_steps)
    # -------------------------
    # Residuals and correction
    # -------------------------
    residual = gen - truth
    # Simple mean-shift correction
    residual_shiftcorr = residual - mean_bias

    # Linear correction (using slope and intercept)
    gen_linear_corr = (gen - intercept) / slope
    residual_linearcorr = gen_linear_corr - truth
    eff_raw = np.array([(np.abs(residual) < c).mean() for c in cuts])
    eff_shiftcorr = np.array([(np.abs(residual_shiftcorr) < c).mean() for c in cuts])
    eff_linearcorr = np.array([(np.abs(residual_linearcorr) < c).mean() for c in cuts])

    ax5 = fig.add_subplot(gs[2, 1])  # reuse bottom row
    ax5.plot(cuts, eff_raw, marker='o', label='raw residual')
    # ax5.plot(cuts, eff_shiftcorr, marker='s', label='shift corrected')
    # ax5.plot(cuts, eff_linearcorr, marker='^', label='linear corrected')

    # -------------------------
    # Highlight efficiency at n_step = 12
    # -------------------------
    highlight_idx = 5  # 12th step
    highlight_cut = cuts[highlight_idx]
    highlight_eff = eff_raw[highlight_idx]

    # vertical line
    ax5.axvline(highlight_cut, color='red', linestyle='--')

    # scatter marker
    ax5.plot(highlight_cut, highlight_eff, 'ro')

    # text annotation
    ax5.annotate(f"{highlight_eff:.3f}",
                 xy=(highlight_cut, highlight_eff),
                 xytext=(highlight_cut + 0.05, highlight_eff - 0.05),
                 arrowprops=dict(arrowstyle="->", color="red"))

    ax5.set_xlabel(r'|residual cut|')
    ax5.set_ylabel('efficiency')
    ax5.set_title('Efficiency vs |residual| cut')
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout()

    # -------------------------
    # Build results object
    # -------------------------
    bin_table = dict(
        bin_centers = bin_centers,
        n_i = n_i,
        mean_truth_bin = mean_truth_bin,
        mean_gen_bin = mean_gen_bin,
        std_gen_bin = std_gen_bin,
        mean_bias_bin = mean_bias_bin,
        std_bias_bin = std_bias_bin,
        se_bias_bin = se_bias_bin,
        rel_bias_mean_bin = rel_bias_mean_bin
    )

    results = dict(
        n = n,
        low = low,
        high = high,
        slope = float(slope),
        intercept = float(intercept),
        pearson_r = float(pearson_r),
        r_squared = float(r_squared),
        mean_bias = float(mean_bias),
        median_bias = float(median_bias),
        std_bias = float(std_bias),
        rms = float(rms),
        ks_stat = ks_stat,
        ks_pvalue = ks_pvalue,
        bin_table = bin_table,
        fig = fig
    )

    correlations = {}
    if extra_vars is not None:
        for name, arr in extra_vars.items():
            arr = np.asarray(arr).ravel()
            if arr.shape[0] != len(gen):
                raise ValueError(f"extra_var {name} length mismatch with mass arrays")

            arr = arr[mask]
            mask_valid = np.isfinite(arr)
            arr = arr[mask_valid]
            g_ = gen[mask_valid]
            t_ = truth[mask_valid]
            res_ = residual[mask_valid]

            # correlations
            corr_gen = np.corrcoef(arr, g_)[0, 1]
            corr_truth = np.corrcoef(arr, t_)[0, 1]
            corr_res = np.corrcoef(arr, res_)[0, 1]

            correlations[name] = dict(
                corr_gen=float(corr_gen),
                corr_truth=float(corr_truth),
                corr_res=float(corr_res)
            )

    corr_vs_cut = {}
    if extra_vars is not None and residual_cuts is not None:
        for name, arr in extra_vars.items():
            arr = np.asarray(arr).ravel()
            arr = arr[mask]
            corr_vs_cut[name] = {}
            for cut in residual_cuts:
                sel = (residual < cut) & np.isfinite(arr)
                if sel.sum() < 5:
                    corr_vs_cut[name][cut] = None
                    continue
                corr_vs_cut[name][cut] = dict(
                    corr_gen   = float(np.corrcoef(arr[sel], gen[sel])[0,1]),
                    corr_truth = float(np.corrcoef(arr[sel], truth[sel])[0,1]),
                    corr_res   = float(np.corrcoef(arr[sel], residual[sel])[0,1])
                )
    results.update(dict(
        correlations = correlations,
        corr_vs_cut = corr_vs_cut
    ))

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    if extra_vars is not None and residual_cuts is not None:
        fig_corr, ax_corr = plt.subplots(figsize=(8,6))
        for name in extra_vars:
            vals = [corr_vs_cut[name][c]["corr_gen"] if corr_vs_cut[name][c] else np.nan
                    for c in residual_cuts]

            line, = ax_corr.plot(residual_cuts, vals, marker='o', label=f"{name} vs gen")
            if name in correlation_data:
                # draw horizontal line for full sample correlation
                ax_corr.axhline(correlation_data[name], ls='--', label=f"{name}-data", color=line.get_color())
        ax_corr.set_xlabel("|gen - truth| cut")
        ax_corr.set_ylabel("Correlation coefficient")
        ax_corr.legend()
        results["fig_corr"] = fig_corr


    if "fig_corr" in results:
        results["fig_corr"].savefig(f"{plot_dir}/mass_relation_correlations.png")
    fig.savefig(f"{plot_dir}/mass_relation_summary.png")
    print(f"[INFO] Saved mass relation plots to {plot_dir}")


    if return_results:
        return results
    else:
        return None


import awkward as ak
import vector

MUON_MASS = 0.1056583755  # GeV


# --------------------------
# Helpers
# --------------------------
def lorentz_boost_ak(p4, beta):
    """Lorentz boost (Awkward-native).
    p4: (...,4) [E,px,py,pz]
    beta: (...,3)
    """
    E = p4[..., 0]
    p = p4[..., 1:4]

    b2 = ak.sum(beta * beta, axis=-1)
    b2 = ak.where(b2 < 1.0 - 1e-15, b2, 1.0 - 1e-15)
    gamma = 1.0 / np.sqrt(1.0 - b2)

    pb = ak.sum(p * beta[:, None, :], axis=-1)
    E_prime = gamma * (E - pb)

    factor = (gamma - 1.0) / ak.where(b2 > 1e-20, b2, 1.0)
    factor = ak.where(b2 < 1e-20, 0.0, factor)

    term = factor * pb - gamma * E
    p_prime = p + term[..., None] * beta[:, None, :]

    return ak.concatenate([E_prime[..., None], p_prime], axis=-1)


def project_to_dimuon_mass_ak(mu, M, m_mu=MUON_MASS):
    """
    Project two muon 4-vectors to exactly satisfy target dimuon mass.
    mu: (N,2,4) awkward array (E,px,py,pz)
    M:  (N,) awkward array of target masses
    """
    # On-shell energies
    p_vec = mu[..., 1:4]
    p2 = ak.sum(p_vec ** 2, axis=-1)
    E = np.sqrt(p2 + m_mu ** 2)
    mu = ak.concatenate([E[..., None], mu[..., 1:4]], axis=-1)

    # Pair beta
    P0 = ak.sum(mu[..., 0], axis=1)
    Pvec = ak.sum(mu[..., 1:4], axis=1)
    beta = Pvec / (P0[..., None] + 1e-12)

    # To PRF
    mu_star = lorentz_boost_ak(mu, beta)

    # Target PRF config
    halfM = 0.5 * M
    arg = ak.where(halfM ** 2 > m_mu ** 2, halfM ** 2 - m_mu ** 2, 0.0)
    pstar = np.sqrt(arg)
    Estar = halfM

    v = mu_star[:, 0, 1:4] - mu_star[:, 1, 1:4]
    v_norm = np.sqrt(ak.sum(v * v, axis=-1))

    n_hat = v / (v_norm[..., None] + 1e-12)

    mu1_star_prime = ak.concatenate([Estar[..., None], +pstar[..., None] * n_hat], axis=-1)
    mu2_star_prime = ak.concatenate([Estar[..., None], -pstar[..., None] * n_hat], axis=-1)
    mu_star_prime = ak.from_regular(ak.concatenate([mu1_star_prime[:, None, :], mu2_star_prime[:, None, :]], axis=1))


    # Back to lab
    mu_prime = lorentz_boost_ak(mu_star_prime, -beta)
    return mu_prime


# -------------------------------
# Main calibration + pt ordering
# -------------------------------
def calibrate_and_order(jet, M_targets, m_mu=MUON_MASS, order_by="projected"):
    """
    Calibrate jet 4-vectors to target dimuon mass and return pt-ordered results.

    jet: awkward vector.zip with (pt,eta,phi,mass[,MASK]), shape (N,2)
    M_targets: (N,) awkward/numpy array of target dimuon masses
    """
    # Convert to Cartesian
    mu = jet
    p4_raw = ak.concatenate([mu.E[..., None], mu.x[..., None], mu.y[..., None], mu.z[..., None]], axis=-1) # (B,N,4)

    # Projection
    p4_proj = project_to_dimuon_mass_ak(p4_raw, M_targets, m_mu=m_mu)
    p4_proj_ord = p4_proj

    # # pt ordering
    # pt_raw = np.sqrt(p4_raw[..., 1] ** 2 + p4_raw[..., 2] ** 2)
    # pt_proj = np.sqrt(p4_proj[..., 1] ** 2 + p4_proj[..., 2] ** 2)
    #
    # if order_by == "projected":
    #     order = ak.argsort(pt_proj, axis=1, ascending=False)
    # elif order_by == "raw":
    #     order = ak.argsort(pt_raw, axis=1, ascending=False)
    # else:
    #     raise ValueError("order_by must be 'projected' or 'raw'")
    #
    #
    #
    # rows = ak.local_index(order, axis=0)
    # p4_raw_ord = p4_raw[rows, order]
    # p4_proj_ord = p4_proj[rows, order]


    jet_proj_ord = vector.awkward.zip(
        {
            "E":  p4_proj_ord[..., 0],
            "px": p4_proj_ord[..., 1],
            "py": p4_proj_ord[..., 2],
            "pz": p4_proj_ord[..., 3],
            "MASK": jet.MASK,
        },
        with_name="Momentum4D"
    )
    return jet_proj_ord


def compare_distributions(dict_ref, dict_target, bins=50, plot_dir=None):
    """
    Compare distributions per label using histograms.

    Parameters:
    -----------
    dict_ref : dict
        Dictionary of reference arrays, e.g., {"a": np.array(...), ...}.
    dict_target : dict
        Dictionary of target arrays with same keys as dict_ref.
    bins : int
        Number of histogram bins.
    """
    labels = dict_ref.keys()

    for label in labels:
        ref = dict_ref[label]
        target = dict_target[label]

        plt.figure(figsize=(7, 5))
        bin_edges = np.linspace(ref.min(), ref.max(), bins + 1)

        plt.hist(ref, bins=bin_edges, density=True, alpha=0.5, label='Data')
        plt.hist(target, bins=bin_edges, density=True, alpha=0.5, label='Gen')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Histogram Comparison for '{label}'")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/hist_{label}.png")
            print(f"[INFO] Saved histogram for '{label}' to {plot_dir}/hist_{label}.png")

