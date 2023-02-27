from sklearn.impute import KNNImputer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import sklearn.feature_selection as fs
from sklearn.model_selection import train_test_split
import mifs
from sklearn.model_selection import cross_val_score
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tabulate import tabulate
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc
import sklearn.metrics as sm
from scipy.special import logit
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import stats
mpl.rc('font',family='Arial')

fs0=10; fs1 = 18; fs2 = 22; fs3 = 26; dpi = 300
hfont = {'fontname':'Arial'}
dpi = 600
fmt = "png"
data_save_path = "/Users/raghavtandon/Documents/PhD/multi-modal/data"
# figure_save_path = "/Users/raghavtandon/Documents/PhD/multi-modal/figures_publication/final_figures_unscaled"
figure_save_path = "/Users/raghavtandon/Documents/PhD/multi-modal/figures_publication/final_figures_v4"
# figure_save_path = "/Users/raghavtandon/Documents/PhD/multi-modal/figures_dir/test_fig"
# storyboard_save_path = "/Users/raghavtandon/Documents/PhD/multi-modal/storyboard"
transparent=True

def imptKNN(x, n_neighbors, weights="uniform"):
	imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
	x_imputed = imputer.fit_transform(x)
	return x_imputed

def protein_selection(X, ref2, n=25, random_seed=0, step_size=1, dx_col="DX"):
	param_dict = {"step":step_size, "n_features_to_select":n}
	vec = ref2["sbj"]
	df = X[X.index.isin(vec)]
	y = ref2[dx_col]
	df = df.reindex(vec)
	# X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=0)
	X_train = df
	y_train = y

	svc = LinearSVC( C=1.0, random_state=0, dual=False, max_iter=3000)
	selector_svc = fs.RFE(svc, **param_dict)
	selector_svc = selector_svc.fit(X_train, y_train)
	selector_svc_support = selector_svc.support_
	selected_proteins_svc = df.columns[selector_svc.support_].tolist()

	logistic = LogisticRegression(C=1.0, random_state=0, dual=False, max_iter=3000)
	selector_logistic = fs.RFE(logistic, **param_dict)
	selector_logistic = selector_logistic.fit(X_train, y_train)
	selector_logistic_support = selector_logistic.support_
	selected_proteins_logistic = df.columns[selector_logistic_support].tolist()

	final_proteins = list(set(selected_proteins_svc).intersection(set(selected_proteins_logistic)))
	return final_proteins

def mifs_selection(X, ref2, method, n=8, random_seed=0, dx_col="DX"):
	vec = ref2["sbj"]
	df = X[X.index.isin(vec)] 
	y = ref2[dx_col]
	df = df.reindex(vec)
	# X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.80, random_state=random_seed)
	feat_selector = mifs.MutualInformationFeatureSelector(method=method, n_features=n)
	d = dict(zip(set(y), [0,1]))
	y_ = y.replace(d)
	fsx = feat_selector.fit(df.values, y_.values)
	return df.columns[fsx.ranking_].tolist()

def logistic_plot(X, ref_train, ref_test, proteins, clf, ax, random_seed=0, dx_col="DX"):
	# Train Dataset
	vec_train = ref_train["sbj"]
	X_train = X[X.index.isin(vec_train)][proteins]
	y_train = ref_train[dx_col]
	X_train = X_train.reindex(vec_train)
	# Test Dataset
	vec_test = ref_test["sbj"]
	X_test = X[X.index.isin(vec_test)][proteins]
	y_test = ref_test[dx_col]
	X_test = X_test.reindex(vec_test)
	# X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=random_seed)
	clf_fitted = clf.fit(X_train, y_train)
	y_prob = clf.predict_proba(X_test)
	y_logprob = clf.predict_log_proba(X_test)
	# fig, ax = plt.subplots(figsize=(6, 6))
	sc = ax.scatter(logit(y_prob)[:,0], y_prob[:,0], c=y_test.map({"Control":"#00CC00", 
																"AD":"#FF0000",
																"AsymAD":"#FF8000"}))
	hline = ax.axhline(y=0.5, linewidth=1, color='k', linestyle="--")
	plt.xticks([], [])
	# ax.set_xlabel("Logits from trained linear model", fontsize=fs1, **hfont)
	# ax.set_xticklabels(,fontsize=fs1)
	remove_box(ax)
	return ax

def prediction_fn(X, ref_train, ref_test, proteins, clf, random_seed=0, dx_col="DX"):
	# Train Dataset
	vec_train = ref_train["sbj"]
	X_train = X[X.index.isin(vec_train)][proteins]
	y_train = ref_train[dx_col]
	X_train = X_train.reindex(vec_train)
	# Test Dataset
	vec_test = ref_test["sbj"]
	X_test = X[X.index.isin(vec_test)][proteins]
	y_test = ref_test[dx_col]
	X_test = X_test.reindex(vec_test)
	# X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=random_seed)
	clf_fitted = clf.fit(X_train, y_train)
	y_pred = clf_fitted.predict(X_test)
	return y_test, y_pred

def add_legend(classes, colors):
	recs=[]
	for i in range(0, len(colors)):
		recs.append(Line2D([0], [0], marker='o', color='w', 
					markerfacecolor=colors[i],
					markersize=fs1))
		# recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
	return recs

def remove_ticks(ax):
	ax.set_xticks([])
	ax.set_yticks([])

def df_violin_cn_ad(X, pr_list, label_df):
	df_violin = X[pr_list]
	df_violin = pd.melt(df_violin, ignore_index=False)
	dx_dict = {"Control":"Ctrl", "AsymAD":"Asym", "AD":"AD"}
	# label_df = label_df[label_df["DX"].isin(["Control", "AD"])]
	label_df["diagnosis"] = label_df["DX"].replace(dx_dict)
	dx_data = label_df[["sbj", "diagnosis"]]
	df_violin = pd.merge(df_violin, dx_data, left_index=True, 
						right_on="sbj", how="inner")
	df_violin.reset_index(drop=True, inplace=True)
	return df_violin

def correlation_distr(X, sbj, proteins, title):
	###############################################################################
	### Correlation Matrix
	###############################################################################

	X_subset = X[X.index.isin(sbj)][proteins]
	corr_mat = np.corrcoef(X_subset.T)
	corr_mat[np.triu_indices_from(corr_mat)] = np.nan
	fig, ax = plt.subplots(1,1, figsize=(15, 15))
	hm = sns.heatmap(corr_mat, vmin=-1, vmax=1, center=0, cmap="bwr", square=True, annot=False, fmt=".1%", ax=ax)
	ax.set_xticklabels(proteins, rotation=90, fontsize=fs0, ha="right", **hfont)
	ax.set_yticklabels(proteins, rotation=0, fontsize=fs0, **hfont)
	# fig.suptitle("Correlation Matrix of the\nselected (overlapping) Proteins {}".format(title), fontsize=f3)
	fig.savefig(os.path.join(figure_save_path, "protein_correlation_plot_heatmap_{}.{}".format(title, fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# plt.close("all")

	corr_flatten = corr_mat.flatten()
	corr_fillzero = np.nan_to_num(corr_flatten, nan=0)
	corr_nonzero = corr_fillzero[np.nonzero(corr_fillzero)[0]]
	fig, ax = plt.subplots(figsize=(10, 10))
	sns.histplot(corr_nonzero, binrange=[-1, 1], binwidth=0.1, stat="probability", color="#929591", 
				edgecolor="k", linewidth=0.3, alpha=0.7)
	ax.set_ylabel("Fraction of Pairs", fontsize=fs2, **hfont)
	ax.tick_params(labelsize=fs1)
	ax.set_xlabel("Correlation", fontsize=fs2, **hfont)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# fig.suptitle("Distribution of pairwise\ncorrelation coefficients {}".format(title), fontsize=f3)
	fig.savefig(os.path.join(figure_save_path, "protein_correlation_plot_distribution_{}.{}".format(title, fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# plt.close("all")

def remove_box(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

def scree_plots(X, sbj):
	X_subset = X[X.index.isin(sbj)]
	print(X_subset.shape)
	fig, ax = plt.subplots(2, 1, figsize=(6, 12), sharex=True, sharey=True)
	ax1 = ax[0]
	ax2 = ax[1]
	remove_box(ax1)
	remove_box(ax2)	
	pca = PCA()
	pca.fit_transform(X_subset)
	var_ratio = pca.explained_variance_ratio_
	cs = np.cumsum(var_ratio)
	sns.lineplot(x=range(1, cs.shape[0]+1), y=100*cs, ax=ax1, color="k")
	major_ticks = np.arange(0, 80, 20)
	minor_ticks = np.arange(0, 80, 5)
	ax1.set_xticks(major_ticks)
	ax1.set_xticks(minor_ticks, minor=True)
	ax1.set_yticks(np.arange(0,100, 20))
	ax1.set_yticklabels(np.arange(0,100, 20), fontsize=fs1)
	ax1.minorticks_on()
	ax1.set_ylabel("Cumulative Variance (%)", fontsize=fs1, **hfont)
	# fig.suptitle("Variance (%) explained\nby increasing PCs", fontsize=fs2, **hfont)
	sns.lineplot(x=range(1, cs.shape[0]+1), y=100*var_ratio, ax=ax2, color="k")
	major_ticks = np.arange(0, 80, 20)
	ax2.set_xticks(major_ticks)
	ax2.set_xticklabels(major_ticks, fontsize=fs1)
	ax2.set_yticks(np.arange(0,100, 20))
	ax2.set_yticklabels(np.arange(0,100, 20), fontsize=fs1)
	ax2.minorticks_on()
	ax2.set_xlabel("PC", fontsize=fs2, **hfont)
	ax2.set_ylabel("Variance (%) for each PC", fontsize=fs1, **hfont)
	fig.savefig(os.path.join(figure_save_path, "pca_scree.{}".format(fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# plt.close("all")

def readData():
	peptide_fname = os.path.join(data_save_path,"EHBS-2-210611", "SRM data from Caroline", "BSR2020-102", "Peptide Area Report_BSR2020-102_80pep.csv")
	skyline_fname = os.path.join(data_save_path,"EHBS-2-210611", "SRM data from Caroline", "BSR2020-102", "SkylineRatios-FullPercesion_2021_0608.csv")
	skyline_df = pd.read_csv(skyline_fname, index_col=0)
	peptide_df = pd.read_csv(peptide_fname)
	# Get the labels for all subjects in the data
	label_df = readLabels(peptide_df)
	# Impute missing values
	Ximp = imputeData(skyline_df)
	ss = StandardScaler()
	scaledX = ss.fit_transform(Ximp)
	scaledX = pd.DataFrame(scaledX, index=Ximp.index, columns=Ximp.columns)
	scaledX = scaledX.reindex(label_df["sbj"])
	Ximp = Ximp.reindex(label_df["sbj"])
	return scaledX, Ximp, label_df

def imputeData(x):
	x_imp = x.T.values.copy()
	n_neighbors=10
	x_imp = imptKNN(x_imp, n_neighbors)
	Ximp = pd.DataFrame(x_imp, index=x.columns, columns=x.index)
	return Ximp

def readLabels(peptide_df):
	label_df = peptide_df[["Replicate", "Condition"]]
	label_df = label_df[label_df["Condition"].isin(["AD", "Control", "AsymAD"])]
	label_dict = dict(zip(label_df["Replicate"], label_df["Condition"]))
	label_df = pd.DataFrame.from_dict(label_dict, orient="index").reset_index()
	label_df.columns = ["sbj", "DX"]
	return label_df

def classCounts(label_df):
	d = dict(Counter(label_df["DX"]))
	count_data = pd.DataFrame(list(d.items()))
	count_data = count_data.append({0:'Total', 1:count_data[1].sum()}, ignore_index=True)
	count_data.set_index(0, inplace=True)
	return tabulate(count_data, headers=["Class", "Count"], tablefmt='orgtbl')

def filterClassPairs(label_df):
	# Filter Control/AD
	dx_ctrl_ad = label_df[label_df["DX"].isin(["Control", "AD"])]
	dx_ctrl_asym = label_df[label_df["DX"].isin(["Control", "AsymAD"])]
	dx_asym_ad = label_df[label_df["DX"].isin(["AsymAD", "AD"])]
	return dx_ctrl_ad, dx_ctrl_asym, dx_asym_ad


def tsneProject(X, dfLabel, classes, peptides, title, random_state=0):
	fig, ax = plt.subplots(1,1, figsize=(10,6))
	s=70
	random_state=random_state
	leg_alpha=0.3
	fontsize_legend=18
	fontsize=22
	legend_properties = {'size':fontsize_legend}
	fontweight="bold"
	dict_colors = {"Control":"#00CC00", "AsymAD":"#FF8000", "AD":"#FF0000"}
	# TSNE
	tsne = TSNE(n_components=2, random_state=random_state)
	# only the control/AD subjects
	df = dfLabel[dfLabel["DX"].isin(classes)]
	df_tsne = X[X.index.isin(df["sbj"])][peptides]
	assert all(df_tsne.index == df["sbj"])
	# assert all(df_tsne.index == dfLabel["sbj"])
	lb = dict(zip(["Control", "AsymAD", "AD"], 
		[dict_colors["Control"], dict_colors["AsymAD"], dict_colors["AD"]]))
	# c = label_df["DX"].replace(lb)
	#df_tsne = df_tsne.reindex(label_df["sbj"])
	tsne_scores = tsne.fit_transform(df_tsne)
	print("KL-divergence", tsne.kl_divergence_)
	ax.scatter(tsne_scores[:,0], tsne_scores[:,1], c=df["DX"].replace(lb), s=s)
	# classes = ["Control", "AsymAD", "AD"]
	colors = [dict_colors[i] for i in classes]
	recs = add_legend(classes, colors)
	ax.legend(recs, classes, loc=1, fontsize=15, 
					framealpha=leg_alpha, 
					prop=legend_properties)
	# ax.set_title("t-SNE", fontsize=fontsize, fontweight=fontweight)
	remove_ticks(ax)
	ax.axis("off")
	fig.savefig(os.path.join(figure_save_path, "tsne_plot_{}.{}".format(title, fmt)), format=fmt, dpi=dpi, transparent=transparent)
	return tsne_scores

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def knnRegress(tsne_cn_asym_ad, label_df, y_col, nbr=5):
	tsne_arr = pd.DataFrame(tsne_cn_asym_ad, columns = ["tsne1", "tsne2"])
	tsne_arr["DX"] = label_df["DX"]
	tsne_arr["sbj"] = label_df["sbj"]
	tsne_arr[y_col] = label_df[y_col]
	# marker_array = label_df["APOE_x"]
	# marker_array = marker_array.apply(lambda x: "".join([x.split("/")[0][-1],
	# 								 x.split("/")[-1][-1]]))
	# tsne_arr["APOE"] = marker_array
	# marker_shape = marker_array.apply(lambda x: "${}$".format(x))
	# tsne_arr["marker_label"] = marker_shape
	train_knn = tsne_arr[tsne_arr["DX"].isin(["Control", "AD"])]
	test_knn = tsne_arr[tsne_arr["DX"].isin(["AsymAD"])]
	neigh = KNeighborsRegressor(n_neighbors=nbr)
	neigh.fit(train_knn[["tsne1", "tsne2"]], train_knn[y_col])
	predicted = neigh.predict(test_knn[["tsne1", "tsne2"]])
	true_vals = test_knn[y_col]
	test_knn["predicted"] = predicted
	train_knn["predicted"] = train_knn[y_col]
	knn_df = pd.concat([train_knn, test_knn])
	print(neigh.score(test_knn[["tsne1", "tsne2"]], true_vals))
	return knn_df

def knnViz2(tsne_cn_asym_ad, label_df, nbr=5):
	# KNN classifier
	tsne_arr = pd.DataFrame(tsne_cn_asym_ad, columns = ["tsne1", "tsne2"])
	tsne_arr["DX"] = label_df["DX"]
	tsne_arr["sbj"] = label_df["sbj"]
	marker_array = label_df["APOE_x"]
	marker_array = marker_array.apply(lambda x: "".join([x.split("/")[0][-1],
									 x.split("/")[-1][-1]]))
	tsne_arr["APOE"] = marker_array
	marker_shape = marker_array.apply(lambda x: "${}$".format(x))
	tsne_arr["marker_label"] = marker_shape
	train_knn = tsne_arr[tsne_arr["DX"].isin(["Control", "AD"])]
	test_knn = tsne_arr[tsne_arr["DX"].isin(["AsymAD"])]
	neigh = KNeighborsClassifier(n_neighbors=nbr)
	neigh.fit(train_knn[["tsne1", "tsne2"]], train_knn["DX"])
	predicted = neigh.predict(test_knn[["tsne1", "tsne2"]])
	test_knn["predicted"] = predicted
	train_knn["predicted"] = train_knn["DX"]
	knn_df = pd.concat([train_knn, test_knn])
	# All points (only shown by AD and CN classes)
	fig, ax = plt.subplots(2,2, figsize=(15,10))
	s=200
	random_state=0
	leg_alpha=0.0
	fontsize_legend=18
	fontsize=22
	text_posx = 0.02
	text_posy = 0.95
	legend_properties = {'size':fontsize_legend}
	fontweight="bold"
	lb1 = {'Control': '#00CC00', 'AsymAD': '#FF8000', 'AD': '#FF0000'}
	lb2 = {'Control': "#006600", 'AD': "#990000"}
	mscatter(train_knn["tsne1"], train_knn["tsne2"], ax=ax[0,0], c=train_knn["predicted"].replace(lb1), s=s, m=train_knn["marker_label"])
	# ax[0,0].text(text_posx, text_posy, "a", 
		# fontsize=fontsize_legend+5, transform=ax[0,0].transAxes,
		# verticalalignment='center')
	classes00 = ["Control", "AD"]
	recs00 = add_legend(classes00, [lb1[i] for i in classes00])
	# box = ax[0,0].get_position()
	# ax[0,0].set_position(box.x0, box.y0, box.width*0.8, box.height)
	ax[0,0].legend(recs00, classes00, fontsize=fontsize_legend, 
					framealpha=leg_alpha, 
					prop=legend_properties, 
					loc="upper right")

	mscatter(knn_df["tsne1"], knn_df["tsne2"], ax=ax[0,1], c=knn_df["DX"].replace(lb1), s=s, m=knn_df["marker_label"])
	# ax[0,1].text(text_posx, text_posy, "b",
	#  fontsize=fontsize_legend+5, transform=ax[0,1].transAxes,
	#  verticalalignment='center')
	classes01 = ["Control", "AsymAD", "AD"]
	recs01 = add_legend(classes01, [lb1[i] for i in classes01])
	ax[0,1].legend(recs01, classes01, fontsize=fontsize_legend, 
					framealpha=leg_alpha, 
					prop=legend_properties,
					loc="upper right")
	mscatter(knn_df["tsne1"], knn_df["tsne2"], ax=ax[1,0],c=knn_df["predicted"].replace(lb1), s=s, m=knn_df["marker_label"])
	# ax[1,0].text(text_posx, text_posy, "c", 
	# 	fontsize=fontsize_legend+5, transform=ax[1,0].transAxes,
	# 	verticalalignment='center')
	classes10 = ["Control", "AD"]
	recs10 = add_legend([i + " driven\ncluster" for i in classes10], [lb1[i] for i in classes10])
	ax[1,0].legend(recs10, [i + " driven\ncluster" for i in classes10], fontsize=fontsize_legend, 
					framealpha=leg_alpha, 
					prop=legend_properties,
					loc="upper right")
	mscatter(train_knn["tsne1"], train_knn["tsne2"], ax=ax[1,1], c=train_knn["predicted"].replace(lb1), s=s, m=train_knn["marker_label"])
	mscatter(test_knn["tsne1"], test_knn["tsne2"], ax=ax[1,1],c=test_knn["predicted"].replace(lb2), s=s, m=test_knn["marker_label"])
	# ax[1,1].text(text_posx, text_posy, "d", 
	# 	fontsize=fontsize_legend+5, transform=ax[1,1].transAxes,
	# 	verticalalignment='center')
	classes11 = ["Control", "AD"] + ["Control-like\nASYMAD", "AD-like\nASYMAD"]
	recs11 = add_legend(classes11, [lb1[i] for i in classes11[:2]] + [lb2[i] for i in classes11[:2]])
	ax[1,1].legend(recs11, [i for i in classes11], fontsize=fontsize_legend, 
					framealpha=leg_alpha, 
					prop=legend_properties,
					loc="upper right")
	for i in range(2):
		for j in range(2):
			remove_ticks(ax[i,j])
	plt.tight_layout()
	fig.savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)
	plt.close("all")
	return test_knn

def knnViz(tsne_cn_asym_ad, label_df, nbr=5):
	# KNN classifier
	tsne_arr = pd.DataFrame(tsne_cn_asym_ad, columns = ["tsne1", "tsne2"])
	tsne_arr["DX"] = label_df["DX"].values
	tsne_arr["sbj"] = label_df["sbj"].values
	# marker_array = label_df["APOE_x"]
	# tsne_arr["APOE"] = marker_array
	# marker_shape = marker_array.apply(lambda x: "${}$".format(x))
	# tsne_arr["marker_label"] = marker_shape
	train_knn = tsne_arr[tsne_arr["DX"].isin(["Control", "AD"])]
	test_knn = tsne_arr[tsne_arr["DX"].isin(["AsymAD"])]
	neigh = KNeighborsClassifier(n_neighbors=nbr)
	neigh.fit(train_knn[["tsne1", "tsne2"]], train_knn["DX"])
	predicted = neigh.predict(test_knn[["tsne1", "tsne2"]])
	test_knn["predicted"] = predicted
	train_knn["predicted"] = train_knn["DX"]
	knn_df = pd.concat([train_knn, test_knn])
	# All points (only shown by AD and CN classes)
	fig, ax = plt.subplots(2,2, figsize=(16,12))
	s=200
	random_state=0
	leg_alpha=0.0
	fontsize_legend=18
	fontsize=22
	text_posx = 0.02
	text_posy = 0.95
	legend_properties = {'size':fontsize_legend}
	fontweight="bold"
	lb1 = {'Control': '#00CC00', 'AsymAD': '#FF8000', 'AD': '#FF0000'}
	lb2 = {'Control': "#006600", 'AD': "#990000"}
	# ax[0,0].scatter(train_knn["tsne1"], train_knn["tsne2"], c=train_knn["predicted"].replace(lb1), s=s, marker=train_knn["marker_label"])
	ax[0,0].scatter(train_knn["tsne1"], train_knn["tsne2"], c=train_knn["predicted"].replace(lb1), s=s)
	# ax[0,0].text(text_posx, text_posy, "a", 
	# 	fontsize=fontsize_legend+5, transform=ax[0,0].transAxes,
	# 	verticalalignment='center')
	classes00 = ["Control", "AD"]
	recs00 = add_legend(classes00, [lb1[i] for i in classes00])
	# box = ax[0,0].get_position()
	# ax[0,0].set_position(box.x0, box.y0, box.width*0.8, box.height)
	# LEGEND goes here
	# ax[0,0].legend(recs00, classes00, fontsize=fontsize_legend, 
	# 				framealpha=leg_alpha, 
	# 				prop=legend_properties, 
	# 				loc="upper right")

	# ax[0,1].scatter(knn_df["tsne1"], knn_df["tsne2"], c=knn_df["DX"].replace(lb1), s=s, marker=knn_df["marker_label"])
	ax[0,1].scatter(knn_df["tsne1"], knn_df["tsne2"], c=knn_df["DX"].replace(lb1), s=s)
	# ax[0,1].text(text_posx, text_posy, "b",
	#  fontsize=fontsize_legend+5, transform=ax[0,1].transAxes,
	#  verticalalignment='center')
	classes01 = ["Control", "AsymAD", "AD"]
	recs01 = add_legend(classes01, [lb1[i] for i in classes01])
	# LEGEND goes here
	# ax[0,1].legend(recs01, classes01, fontsize=fontsize_legend, 
	# 				framealpha=leg_alpha, 
	# 				prop=legend_properties,
	# 				loc="upper right")
	# ax[1,0].scatter(knn_df["tsne1"], knn_df["tsne2"], c=knn_df["predicted"].replace(lb1), s=s, marker=knn_df["marker_label"])
	# ax[1,0].scatter(knn_df["tsne1"], knn_df["tsne2"], c=knn_df["predicted"].replace(lb1), s=s)
	ax[1,0].scatter(train_knn["tsne1"], train_knn["tsne2"], c=train_knn["predicted"].replace(lb1), s=s)
	ax[1,0].scatter(test_knn["tsne1"], test_knn["tsne2"], c=test_knn["predicted"].replace(lb2), s=s)
	# ax[1,0].text(text_posx, text_posy, "c", 
	# 	fontsize=fontsize_legend+5, transform=ax[1,0].transAxes,
	# 	verticalalignment='center')
	classes10 = ["Control", "AD"] + ["Control-like AsymAD", "AD-like AsymAD"]
	recs10 = add_legend(classes10, [lb1[i] for i in classes10[:2]] + [lb2[i] for i in classes10[:2]])
	# recs10 = add_legend([i + " driven\ncluster" for i in classes10], [lb1[i] for i in classes10])
	# LEGEND goes here
	# ax[1,0].legend(recs10, [i for i in classes10], fontsize=fontsize_legend, 
	# 				framealpha=leg_alpha, 
	# 				prop=legend_properties,
	# 				loc="upper right")
	# ax[1,1].scatter(train_knn["tsne1"], train_knn["tsne2"], c=train_knn["predicted"].replace(lb1), s=s, marker=train_knn["marker_label"])
	# ax[1,1].scatter(train_knn["tsne1"], train_knn["tsne2"], c=train_knn["predicted"].replace(lb1), s=s)
	ax[1,1].scatter(test_knn["tsne1"], test_knn["tsne2"], c=test_knn["predicted"].replace(lb2), s=s)
	# ax[1,1].text(text_posx, text_posy, "d", 
	# 	fontsize=fontsize_legend+5, transform=ax[1,1].transAxes,
	# 	verticalalignment='center')
	classes11 = ["Control-like AsymAD", "AD-like AsymAD"]
	recs11 = add_legend(classes11, [lb2[i] for i in classes10[:2]])
	# LEGEND goes here
	# ax[1,1].legend(recs11, [i for i in classes11], fontsize=fs1, 
	# 				framealpha=leg_alpha, 
	# 				prop=legend_properties,
	# 				loc="upper right")
	# for i in range(2):
	# 	for j in range(2):
	# 		remove_ticks(ax[i,j])
	ax[0,0].axis("off")
	ax[1,0].axis("off")
	ax[0,1].axis("off")
	ax[1,1].axis("off")
	plt.tight_layout()
	# ax[0,0].savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}_ax1.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# ax[0,1].savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}_ax2.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# ax[1,0].savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}_ax3.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# ax[1,1].savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}_ax4.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)
	fig.savefig(os.path.join(figure_save_path, "tsne_dim_reduction_n{}.{}".format(str(nbr), fmt)), format=fmt, dpi=dpi, transparent=transparent)

	# plt.close("all")
	return test_knn

def plotROC(X, dx_cmpr, peps, fname, ncv=6, random_state=0, figure_save_path=figure_save_path):
	clf = LogisticRegression( C=1.0, random_state=0, dual=False, max_iter=3000, class_weight="balanced")
	cv = StratifiedKFold(n_splits=ncv)
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	fig, ax = plt.subplots(figsize=(8,6))
	X_cmpr = X[X.index.isin(dx_cmpr["sbj"])]
	cv_x = X_cmpr[peps].values
	cv_y = dx_cmpr["DX"].values
	for i, (train, test) in enumerate(cv.split(cv_x, cv_y)):
	    clf.fit(cv_x[train], cv_y[train])
	    y_ = clf.predict(cv_x[test])
	    # print(sm.f1_score(cv_y[test], y_))
	    # print(sm.roc_auc_score(cv_y[test], y_))
	    viz = plot_roc_curve(clf, cv_x[test], cv_y[test],
	                         name='ROC fold {}'.format(i+1),
	                         alpha=0.5, lw=1.5, ax=ax)
	    print(viz.roc_auc)
	    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
	    interp_tpr[0] = 0.0
	    tprs.append(interp_tpr)
	    aucs.append(viz.roc_auc)

	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
	        label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = stats.sem(aucs, ddof=1)
	print("AuC", np.mean(aucs))
	# std_auc = np.std(aucs)
	ax.plot(mean_fpr, mean_tpr, color='k',
	        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
	        lw=2.5, alpha=.8)

	std_tpr = stats.sem(tprs, axis=0, ddof=1)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4)#,label=r'$\pm$ 1 std. dev.')

	ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
	# ax.set_title("ROC plot (with {} fold cross validation)".format(ncv),
	#        fontsize=15, fontweight="bold")
	ax.legend(loc="lower right", fontsize=12, framealpha=0.25)
	ax.set_xlabel("False Positive Rate", fontsize=24, fontname='Arial')
	ax.set_ylabel("True Positive Rate", fontsize=24, fontname='Arial')
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	remove_box(ax)
	fig.tight_layout()
	fig.savefig(os.path.join(figure_save_path, "roc_{}.{}".format(fname, fmt)), format=fmt, dpi=dpi, transparent=transparent)
	# plt.close("all")

def confusionMatrix(y_test, y_pred):
	## Pretty print the confusion matrix
	labels = list(Counter(y_test).keys())
	arr = sm.confusion_matrix(y_test, y_pred, labels=labels)
	arr_counts = arr.sum(axis=1)
	arr_df = pd.DataFrame(arr)
	arr_df.columns = ["Predicted\n{}".format(labels[0]), "Predicted\n{}".format(labels[1])]
	arr_df.index = ["True {}".format(labels[0]), "True {}".format(labels[1])]
	return tabulate(arr_df, headers='keys', tablefmt='psql')

def dfPeptide(peptides):
	df = pd.DataFrame(peptides, columns=["Peptides"], index=range(1, len(peptides)+1))
	return tabulate(df, headers='keys')
