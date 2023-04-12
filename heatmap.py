import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# heatmaps

def heatmap(heatmapGS, data_hp, Alleles, C_range, gamma_range):
    
    fig, ax= plt.subplots(1,1, figsize=(8,6))
      
    
    sns.heatmap(data_hp, xticklabels = gamma_range, yticklabels = C_range, annot = False, linewidth = 1)
    ax.set(xlabel="Gamma", ylabel="C")
    ax.tick_params(axis='y', rotation =0)
    ax.set_ylabel(ax.get_ylabel(), rotation=0)
    ax.set_title('Prediction Accuracies by Grid Search  '+'['+Alleles+']')
    #plt.show()
    
    heatmapGS.savefig(fig)   # save heatmap to the pdf file 
      
    return heatmapGS
