#!/usr/bin/env python3
"""Render manuscript Figures 4--8 from newly computed numerical outputs.

Panel PDFs remain individually usable; PyMuPDF assembles multi-panel figures
without rasterizing numerical plots. The overview contains Figures 4--8 and
Table 1; Figures 5 and 7 include model-derived stochastic and district views.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from scipy.special import expit,ndtri
from electoral_feedback import choice_and_jacobian
from plot_style import apply_scienceplots_style
from repro_utils import ROOT,write_json,sha256


def district_ideology_means(count,g):
    if count<1:raise ValueError('district count must be positive')
    if count==1 or g==0:return np.zeros(count)
    probabilities=(np.arange(count)+.5)/count
    return g*ndtri(probabilities)


def geographic_electorate(count,g,*,L=4.,nx=801,total_sd=1.):
    x=np.linspace(-L,L,nx)
    means=district_ideology_means(count,g)
    within=max(.15,np.sqrt(max(total_sd**2-g**2,.0225)))
    densities=np.exp(-.5*((x[None,:]-means[:,None])/within)**2)
    densities/=np.trapezoid(densities,x,axis=1)[:,None]
    return x,means,densities,densities.mean(axis=0)


def binary_district_response(means,*,g=.7,L=4.,sigma=.65,beta=24.,nx=801,
                             platforms=(-.35,.35)):
    x=np.linspace(-L,L,nx)
    within=max(.15,np.sqrt(max(1.-g**2,.0225)))
    densities=np.exp(-.5*((x[None,:]-np.asarray(means)[:,None])/within)**2)
    densities/=np.trapezoid(densities,x,axis=1)[:,None]
    a=np.exp(-.5*((x[:,None]-np.asarray(platforms)[None,:])/sigma)**2)
    choice=choice_and_jacobian(a)[0]
    weights=np.ones(nx)*(x[1]-x[0]);weights[[0,-1]]*=.5
    votes=(densities*weights[None,:])@choice
    right_share=votes[:,1]/votes.sum(axis=1)
    plurality_share=expit(beta*(2*right_share-1))
    return right_share,plurality_share


def save_panel(fig,path,*,tight=True):
    import fitz
    if tight:fig.tight_layout(pad=1.2)
    pdf_path=path.with_suffix('.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    with fitz.open(pdf_path) as pdf:
        pdf[0].get_pixmap(dpi=300,alpha=False).save(path.with_suffix('.png'))


def assemble(out,number,panels,rectangles,width,height):
    import fitz
    dest=out/f'figure{number:02d}.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=width,height=height)
        for path,rect in zip(panels,rectangles,strict=True):
            with fitz.open(path) as src:page.show_pdf_page(fitz.Rect(rect),src,0)
        doc.set_metadata(dict(title=f'Figure {number}: recomputed numerical results',
                              subject='Source: newly generated simulation CSV and NPZ outputs'))
        doc.save(dest,garbage=4,deflate=True)
        page.get_pixmap(dpi=300,alpha=False).save(out/f'figure{number:02d}.png')
    return dest


def make_figures(root):
    apply_scienceplots_style()
    result=root/'results';out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    panels=out/'panels';panels.mkdir(exist_ok=True)
    data=pd.read_csv(result/'theta_binary_map.csv')
    roots=pd.read_csv(result/'theta_critical_tolerance.csv')
    ts=pd.read_csv(result/'theta_nonlinear_timeseries.csv')
    final=pd.read_csv(result/'theta_nonlinear_summary.csv')
    stochastic=pd.read_csv(result/'theta_stochastic_trajectory.csv')
    finite=pd.read_csv(result/'distributional_critical_curves.csv')
    modes=pd.read_csv(result/'distributional_mode_validation.csv')
    geo=pd.read_csv(result/'geography_summary.csv')
    points=pd.read_csv(result/'point_thresholds.csv')

    # Figure 4: temporal generator rate, NOT the strategic gain or its margin.
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    pivot=data.pivot(index='r',columns='theta',values='lambda_odd').sort_index()
    th=pivot.columns.to_numpy(float);rr=pivot.index.to_numpy(float)
    growth=pivot.to_numpy()
    norm=TwoSlopeNorm(vmin=growth.min(),vcenter=0.0,vmax=growth.max())
    image=ax.pcolormesh(th,rr,growth,shading='nearest',cmap='RdBu_r',norm=norm)
    ax.contour(th,rr,growth,levels=[0],colors='black',linewidths=1.4)
    fig.colorbar(image,ax=ax,label=r'Odd growth rate $\lambda_{\mathrm{odd}}$')
    ax.set(xlabel=r'Institutional-payoff weight $\theta$',ylabel=r'Tolerance ratio $r=\sigma/\sigma_0$',
           title='(a) Binary growth and decay')
    save_panel(fig,panels/'figure04a')
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    for K,group in roots.groupby('K',sort=True):
        group=group.sort_values('theta')
        ax.plot(group.theta,group.r_critical,'o-',markersize=3,label=f'$K={K}$')
    ax.set(xlabel=r'Institutional-payoff weight $\theta$',ylabel=r'Critical tolerance $r_c^W$',
           title='(b) Objective-dependent boundaries',xlim=(0,1))
    ax.legend(frameon=False,ncol=2)
    save_panel(fig,panels/'figure04b')
    assemble(out,4,[panels/'figure04a.pdf',panels/'figure04b.pdf'],[(0,0,380,300),(380,0,760,300)],760,300)

    labels={r.case:rf'$\theta={r.theta:.3f}$'+(' (critical)' if r.case=='critical' else '')
            for r in final.itertuples()}
    order=['seat','stable','critical','unstable','vote']
    for key,ylabel,logarithm,letter in [('P',r'Between-party spread $P(t)$',True,'a'),
                                        ('turnout',r'Turnout $T(t)$',False,'b')]:
        fig,ax=plt.subplots(figsize=(5.2,3.8))
        for case in order:
            group=ts[ts.case==case]
            ax.plot(group.time,group[key],label=labels[case])
        if logarithm:ax.set_yscale('log')
        ax.set(xlabel='Time',ylabel=ylabel,title=f'({letter}) '+('Differentiation' if logarithm else 'Participation'))
        if logarithm:ax.legend(frameon=False,fontsize=8)
        save_panel(fig,panels/f'figure05{letter}')
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    branch_colors={-1:'#D55E00',1:'#0072B2'}
    highlighted={0:'Representative positive branch',1:'Representative negative branch'}
    for seed,group in stochastic.groupby('seed',sort=True):
        branch=int(np.sign(group.differentiation.iloc[-1]));color=branch_colors[branch]
        is_highlighted=int(seed) in highlighted
        ax.plot(group.differentiation,group.turnout,color=color,linestyle='-',
                linewidth=1.25 if is_highlighted else .55,
                alpha=.9 if is_highlighted else .20,
                label=highlighted.get(int(seed)),zorder=2 if is_highlighted else 1)
        ax.scatter(group.differentiation.iloc[-1],group.turnout.iloc[-1],s=18 if is_highlighted else 8,
                   color=color,alpha=.95 if is_highlighted else .45,zorder=3)
    deterministic=ts[ts.case=='unstable'].sort_values('time')
    for sign in [-1,1]:
        ax.plot(sign*deterministic.P,deterministic.turnout,'k--',linewidth=.9,
                label='Deterministic paths' if sign==-1 else None,zorder=3)
    deterministic_spread=float(deterministic.P.iloc[-1]);deterministic_turnout=float(deterministic.turnout.iloc[-1])
    ax.scatter([-deterministic_spread,deterministic_spread],[deterministic_turnout]*2,marker='*',s=48,
               facecolors='white',edgecolors='black',linewidth=.8,label='Deterministic endpoints',zorder=4)
    initial=stochastic[(stochastic.seed==stochastic.seed.min())&(stochastic.time==stochastic.time.min())].iloc[0]
    ax.scatter(0,initial.turnout,marker='o',s=30,facecolors='white',edgecolors='black',
               linewidth=.8,label='Common initial state',zorder=5)
    difference_limit=1.08*max(deterministic_spread,stochastic.differentiation.abs().max())
    turnout_min=min(stochastic.turnout.min(),deterministic.turnout.min())
    turnout_max=max(stochastic.turnout.max(),deterministic.turnout.max())
    turnout_pad=.07*(turnout_max-turnout_min)
    ax.set(xlim=(-difference_limit,difference_limit),ylim=(turnout_min-turnout_pad,turnout_max+turnout_pad),
           xlabel=r'Differentiation mode $d(t)$',ylabel=r'Turnout $T(t)$',
           title=r'(c) Stochastic trajectories in projected phase space')
    ax.legend(frameon=False,fontsize=6.8,loc='upper center',ncol=2)
    save_panel(fig,panels/'figure05c')

    fig,ax=plt.subplots(figsize=(5.2,4.1))
    cases=['seat','unstable','vote']
    endpoint_colors={'seat':'#000000','unstable':'#009E73','vote':'#D55E00'}
    for i in [0,1]:
        for case in cases:
            with np.load(result/f'nonlinear/{case}.npz') as a:
                ax.plot(a['y'],a['final_densities'][i],ls='-' if i==0 else '--',
                        color=endpoint_colors[case],label=labels[case] if i==0 else None)
    ax.set(xlim=(-.8,.8),xlabel='Ideological position',ylabel='Terminal party density',
           title='(d) Terminal profiles; solid / dashed distinguish the parties')
    ax.legend(frameon=False,ncol=3)
    save_panel(fig,panels/'figure05d')
    assemble(out,5,[panels/f'figure05{s}.pdf' for s in 'abcd'],
             [(0,0,380,278),(380,0,760,278),(0,278,380,578),(380,278,760,578)],760,578)

    # Figure 6 retains the conservative linearized experiment specified in F.
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    for K,group in finite.groupby('K',sort=True):
        group=group.sort_values('eps')
        ax.plot(group.eps,group.r_critical,'o-',markersize=4,label=f'$K={K}$')
    ax.set_prop_cycle(None)
    for K,group in finite.groupby('K',sort=True):
        ax.plot([finite.eps.min(),finite.eps.max()],[group.point_critical.iloc[0]]*2,':',linewidth=1)
    ax.set_xscale('log',base=2)
    ax.set_xticks(sorted(finite.eps.unique()),[f'{e:g}' for e in sorted(finite.eps.unique())])
    ax.tick_params(axis='x',labelsize=8)
    ax.set(xlabel=r'Internal diffusion $\varepsilon$',ylabel=r'Critical tolerance $r_c^W$',
           title='(a) Finite-width boundaries')
    ax.legend(frameon=False)
    save_panel(fig,panels/'figure06a')
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    selected=modes[(modes.K==3)&np.isclose(modes.eps,.004)]
    if selected.empty:selected=modes[modes.K==3]
    groups=list(selected.groupby(['regime','r'],sort=True))
    for (regime,r),group in groups:
        ax.plot(group.time,group.amplitude,label=f'{regime.capitalize()}, $r={r:.3f}$')
    ax.set_prop_cycle(None)
    for j,((regime,r),group) in enumerate(groups):
        sub=group.iloc[::max(1,len(group)//9)]
        pred=group.amplitude.iloc[0]*np.exp(group.leading_growth_rate.iloc[0]*sub.time)
        ax.plot(sub.time,pred,'o',fillstyle='none',markersize=4,label='Spectral prediction' if j==0 else None)
    ax.set_yscale('log')
    ax.set(xlabel='Time',ylabel='Odd-mode amplitude',title='(b) Conservative linearised validation')
    ax.legend(frameon=False,fontsize=9)
    save_panel(fig,panels/'figure06b')
    assemble(out,6,[panels/'figure06a.pdf',panels/'figure06b.pdf'],[(0,0,380,300),(380,0,760,300)],760,300)

    names={'PR':'District PR','FPTP':'Plurality','MMP':'MMP'};rules=['PR','FPTP','MMP']
    g5=geo[geo.K==5]
    geography_metadata=json.loads((result/'geography_metadata.json').read_text())
    horizon=geography_metadata['horizon']
    geography_levels=np.asarray(geography_metadata['geography'],dtype=float)
    representative_g=float(geography_levels[np.argmin(abs(geography_levels-.7))])
    district_count=int(geography_metadata['D'])
    ideology,district_means,district_densities,national_density=geographic_electorate(
        district_count,representative_g,L=float(geography_metadata['L']))
    color_limit=float(np.max(abs(district_means)))
    district_norm=TwoSlopeNorm(vmin=-color_limit,vcenter=0.,vmax=color_limit)
    district_cmap=plt.get_cmap('RdBu_r')
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    ridge_scale=.78/district_densities.max()
    for district,(mean,density) in enumerate(zip(district_means,district_densities,strict=True)):
        baseline=float(district);curve=baseline+ridge_scale*density
        color=district_cmap(district_norm(mean))
        ax.fill_between(ideology,baseline,curve,color=color,alpha=.82,linewidth=0)
        ax.plot(ideology,curve,color='#262626',linewidth=.7,linestyle='-')
        ax.axhline(baseline,color='#aaaaaa',linewidth=.35,zorder=0)
    ax.set(xlim=(-float(geography_metadata['L']),float(geography_metadata['L'])),
           ylim=(-.15,district_count-.05),xlabel='Ideological position $x$',
           yticks=np.arange(district_count),yticklabels=[rf'$D_{{{i}}}$' for i in range(1,district_count+1)],
           title=rf'(c) Stylised geographic electorate ($g={representative_g:.2f}$)')
    ax.tick_params(axis='x',which='both',top=False)
    ax.tick_params(axis='y',which='both',left=False,right=False)
    ax.grid(False)
    ax.spines[['left','right','top']].set_visible(False)
    save_panel(fig,panels/'figure07c')

    response_means=np.linspace(-2.1,2.1,401)
    pr_response,fptp_response=binary_district_response(
        response_means,g=representative_g,L=float(geography_metadata['L']),
        sigma=float(geography_metadata['sigma']),beta=float(geography_metadata['beta']))
    district_pr,district_fptp=binary_district_response(
        district_means,g=representative_g,L=float(geography_metadata['L']),
        sigma=float(geography_metadata['sigma']),beta=float(geography_metadata['beta']))
    fig=plt.figure(figsize=(5.2,4.1))
    response_grid=fig.add_gridspec(2,1,height_ratios=(3.2,1),hspace=.06)
    ax=fig.add_subplot(response_grid[0])
    density_ax=fig.add_subplot(response_grid[1],sharex=ax)
    ax.plot(response_means,100*fptp_response,color='#D55E00',linewidth=1.7,label='FPTP / plurality')
    ax.plot(response_means,100*pr_response,color='#0072B2',linewidth=1.5,label='PR benchmark')
    ax.scatter(district_means,100*district_fptp,color='#D55E00',edgecolor='white',linewidth=.4,s=17,zorder=3)
    ax.scatter(district_means,100*district_pr,color='#0072B2',edgecolor='white',linewidth=.4,s=17,zorder=3)
    ax.axhline(50,color='#777777',linewidth=.7,linestyle='--')
    ax.axvline(0,color='#bbbbbb',linewidth=.5)
    ax.set(xlim=(-2.1,2.1),ylim=(-2,102),ylabel=r'Right-party district share (\%)',
           title=r'(d) Electoral response for fixed platforms $(-0.35,\,0.35)$')
    ax.tick_params(axis='x',labelbottom=False);ax.legend(frameon=False,loc='upper left')
    density_ax.fill_between(ideology,national_density,color='#777777',alpha=.32,linewidth=0)
    density_ax.plot(ideology,national_density,color='#444444',linewidth=.8)
    density_ax.vlines(district_means,0,.12*national_density.max(),
                      colors=[district_cmap(district_norm(mean)) for mean in district_means],linewidth=1)
    density_ax.set(xlim=(-2.1,2.1),ylim=(0,1.12*national_density.max()),yticks=[],
                   xticks=[-2,0,2],xticklabels=['Left','Centre','Right'],
                   xlabel=r'Ideology $x$ / district mean $\mu_d$')
    density_ax.text(.02,.78,'National electorate',transform=density_ax.transAxes,fontsize=8)
    density_ax.spines[['left','right','top']].set_visible(False);density_ax.grid(False)
    fig.subplots_adjust(left=.14,right=.97,bottom=.15,top=.91,hspace=.06)
    save_panel(fig,panels/'figure07d',tight=False)

    fig,ax=plt.subplots(figsize=(5.2,4.1))
    geographic_lines=[]
    for rule in rules:
        group=g5[g5.rule==rule].sort_values('g')
        line,=ax.plot(group.g,group.P_geographic_mean,'o-',label=names[rule],markersize=4)
        geographic_lines.append(line)
    ax.set_prop_cycle(None)
    control_lines=[]
    for rule in rules:
        group=g5[g5.rule==rule].sort_values('g')
        line,=ax.plot(group.g,group.P_homogeneous_mean,'o--',fillstyle='none',markersize=4)
        control_lines.append(line)
    ax.set(xlabel='Geography $g$',ylabel=rf'Terminal spread $P({horizon:g})$',title='(a) District organisation')
    ax.legend([*geographic_lines,tuple(control_lines)],
              [*[names[rule] for rule in rules],'Matched homogeneous controls'],
              handler_map={tuple:HandlerTuple(ndivide=None)},frameon=False,fontsize=8)
    save_panel(fig,panels/'figure07a')

    fig,ax=plt.subplots(figsize=(5.2,4.1))
    for rule in rules:
        group=g5[g5.rule==rule].sort_values('g')
        ax.errorbar(group.g,group.delta_mean,yerr=group.delta_sd,fmt='o-',capsize=3,
                    label=names[rule],markersize=4)
    ax.axhline(0,linewidth=.6,alpha=.5)
    ax.set(xlabel='Geography $g$',ylabel=rf'Paired difference in $P({horizon:g})$',title=r'(b) Mean difference $\pm$ one sample SD')
    ax.legend(frameon=False,fontsize=9)
    save_panel(fig,panels/'figure07b')
    assemble(out,7,[panels/f'figure07{panel}.pdf' for panel in 'abcd'],
             [(0,0,380,300),(380,0,760,300),(0,300,380,600),(380,300,760,600)],760,600)

    fig,ax=plt.subplots(figsize=(7.2,4.2))
    ax.plot(points.K,points.r_c,label='Exact positive-moment threshold')
    asym=points[points.K>=20]
    ax.plot(asym.K,asym.r_asymptotic,'--',label='Large-$K$ expansion')
    ax.set_xscale('log');ax.set_ylim(.62,1.005)
    ax.set(xlabel='Number of parties $K$',ylabel=r'Point critical tolerance $r_c$',title='Exact point-platform threshold')
    ax.legend(frameon=False)
    save_panel(fig,panels/'figure08')
    assemble(out,8,[panels/'figure08.pdf'],[(0,0,620,362)],620,362)

    table=pd.read_csv(result/'table1.csv')
    fig,ax=plt.subplots(figsize=(7.2,2.5));ax.axis('off')
    cells=[[f'{row.g:.2f}',f'{row.PR:.6f}',f'{row.FPTP:.6f}',f'{row.MMP:.6f}'] for row in table.itertuples()]
    tab=ax.table(cellText=cells,colLabels=['Geography $g$','PR','Plurality','MMP'],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.7)
    ax.set_title('Table 1. Binary divergence rates from automatic differentiation',pad=16)
    save_panel(fig,out/'table01')
    source_names=['theta_binary_map.csv','theta_critical_tolerance.csv','theta_nonlinear_timeseries.csv',
                  'theta_nonlinear_summary.csv','theta_stochastic_trajectory.csv','theta_stochastic_seed_checks.csv',
                  'theta_stochastic_metadata.json',
                  'distributional_critical_curves.csv','distributional_mode_validation.csv',
                  'geography_summary.csv','geography_metadata.json','point_thresholds.csv','table1.csv']
    write_json(out/'numeric_figure_provenance.json',dict(figures=[4,5,6,7,8],table=[1],
        kind='recomputed numeric data with a finite-particle stochastic realisation, model-derived district ridgelines and electoral-response curves; SciencePlots science+ieee styling',
        source_sha256={s:sha256(result/s) for s in source_names}))
    create_gallery(root)


def create_gallery(root):
    import fitz
    figures=root/'figures'
    profile=json.loads((root/'results/theta_phase_metadata.json').read_text())['settings']['profile']
    prefix='QUICK SMOKE | ' if profile=='quick' else ''
    titles={4:'Mixed-objective phase boundaries',
            5:'Binary nonlinear dynamics and terminal densities',6:'Diffusion and local amplification',
            7:'Matched-national-density geographic comparisons',8:'Exact point-platform threshold'}
    with fitz.open() as gallery:
        for number in range(4,9):
            path=figures/f'figure{number:02d}.pdf'
            if not path.exists():continue
            with fitz.open(path) as source:
                height=source[0].rect.height/source[0].rect.width*720
                page=gallery.new_page(width=768,height=height+90)
                page.insert_text((24,27),f'{prefix}Figure {number}  |  {titles[number]}',fontsize=12)
                page.show_pdf_page(fitz.Rect(24,43,744,43+height),source,0)
                note='Source: simulation CSV / NPZ outputs. Numerical checks: results/verification.json.'
                page.insert_text((24,height+73),note,fontsize=8)
        with fitz.open(figures/'table01.pdf') as source:
            page=gallery.new_page(width=768,height=315)
            page.show_pdf_page(fitz.Rect(24,20,744,270),source,0)
            page.insert_text((24,294),'Recomputed using automatic differentiation of the full own-party strategic force.',fontsize=9)
        gallery.set_metadata(dict(title='A Dynamic Model of Party Differentiation under Electoral Competition - reproduction figure gallery',
                                  subject='Numerical Figures 4-8 and Table 1'))
        gallery.save(figures/'overview.pdf',garbage=4,deflate=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=ROOT)
    args=ap.parse_args();make_figures(args.output_dir)
    print(f'Numeric figures and gallery written to {args.output_dir}',flush=True)

if __name__=='__main__':main()
