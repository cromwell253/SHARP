from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
R=Path(__file__).resolve().parents[2];O=R/'runs/engineering_extension';cases=json.loads((O/'selected_cases.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
for lang in ['EN','CN']:
 fp=FontProperties(family='SimSun',size=10) if lang=='CN' else None
 fig,axes=plt.subplots(2,1,figsize=(9,6.8),layout='constrained')
 for ax,c in zip(axes,cases):
  z=np.load(O/(c['group']+'_curves.npz'));past=np.arange(-11,1);future=np.arange(1,49)
  ax.axvspan(-11,0,color='#eaf2fa');ax.axvspan(0,5,color='#fff0f5',alpha=.75)
  ax.plot(past,z['history'],color='#2875ad',lw=2,label='历史实测' if lang=='CN' else 'Observed history')
  ax.plot(np.r_[0,future],np.r_[z['history'][-1],z['truth']],color='#222',lw=1.8,ls='--',label='未来实测' if lang=='CN' else 'Future observations')
  ax.plot(np.r_[0,future],np.r_[z['history'][-1],z['SHARP']],color='#568a22',lw=2,label='SHARP')
  ax.plot(np.r_[0,future],np.r_[z['history'][-1],z['PatchSTG']],color='#bd4c8c',lw=1.8,label='PatchSTG')
  ax.axhline(c['threshold'],color='#c32b30',ls='--',lw=1.4,label='情景阈值' if lang=='CN' else 'Scenario threshold')
  ax.axvline(0,color='#555',lw=1.5);lead=c['onset']-c['target_start']+1;ax.axvline(lead,color='#d48d25',ls=':',lw=1.2)
  title=('(a) SHARP阈值提示命中' if c['group']=='SHARP_success' else '(b) SHARP未产生越界提示') if lang=='CN' else ('(a) SHARP threshold alert hit' if c['group']=='SHARP_success' else '(b) SHARP threshold alert missed')
  ax.set_title(title+f"  |  {c['dataset']}, node {c['node']}",fontproperties=fp,loc='left',pad=22)
  ax.annotate('',xy=(0,1.035),xytext=(-11,1.035),xycoords=('data','axes fraction'),arrowprops=dict(arrowstyle='<->',color='#2875ad',lw=2))
  ax.annotate('',xy=(48,1.035),xytext=(0,1.035),xycoords=('data','axes fraction'),arrowprops=dict(arrowstyle='<->',color='#bd4c8c',lw=2))
  ax.text(-5.5,1.045,'历史' if lang=='CN' else 'History',ha='center',transform=ax.get_xaxis_transform(),fontproperties=fp,color='#2875ad',fontsize=9)
  ax.text(24,1.045,'未来48分钟预测' if lang=='CN' else '48-minute forecast',ha='center',transform=ax.get_xaxis_transform(),fontproperties=fp,color='#bd4c8c',fontsize=9)
  ax.set_xlim(-11,48);ax.set_ylabel('应力 / MPa' if lang=='CN' else 'Stress (MPa)',fontproperties=fp);ax.set_xlabel('相对预测发出时刻 / 分钟' if lang=='CN' else 'Minutes from forecast issue',fontproperties=fp);ax.grid(alpha=.22)
  ax.legend(loc='upper left',fontsize=8,prop=fp if fp else {'size':8},ncol=3)
  ia=inset_axes(ax,width='30%',height='42%',loc='lower right',borderpad=1.2)
  for name,color in [('truth','#222'),('SHARP','#568a22'),('PatchSTG','#bd4c8c')]:ia.plot(np.arange(1,6),z[name][:5],color=color,ls='--' if name=='truth' else '-',lw=1.4)
  ia.axhline(c['threshold'],color='#c32b30',ls='--',lw=.8);ia.axvline(lead,color='#d48d25',ls=':',lw=.8);ia.set_xticks([1,3,5]);ia.tick_params(labelsize=7);ia.set_title('前5分钟' if lang=='CN' else 'First 5 minutes',fontproperties=fp,fontsize=8);ia.grid(alpha=.2)
 for ext in ['png','pdf','svg']:fig.savefig(O/f'SHARP_event_cases_{lang}.{ext}',dpi=400)
 plt.close(fig)
print('Figures generated')
