import pandas as pd

def plot_correlation_map():
	plt.figure(figsize=(50,50))
	plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
	plt.title('without resampling', size=15)
	plt.colorbar()
	plt.show()


def print_lr_coefficients(model):
	coefs = {
	    'name': [],
	    'value': [],
	}
	for coef , v in zip(model.coef_[0], df.columns):
	    coefs['name'].append(v)
	    coefs['value'].append(coef)
	    
	return pd.DataFrame(coefs).sort_values('value' , ascending=False)