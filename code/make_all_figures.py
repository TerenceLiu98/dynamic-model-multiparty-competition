#!/usr/bin/env python3
"""Render manuscript Figures 4--8 from newly computed CSV/NPZ files only.

Panel PDFs remain individually usable; PyMuPDF assembles multi-panel figures
without rasterizing numerical plots. The overview contains Figures 4--8 and
Table 1; conceptual illustrations are not part of this numerical package.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plot_style import apply_scienceplots_style
from repro_utils import ROOT,write_json,sha256


def save_panel(fig,path):
    fig.tight_layout(pad=1.2)
    pdf_path=path.with_suffix('.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    with fitz.open(pdf_path) as pdf:
        pdf[0].get_pixmap(dpi=300,alpha=False).save(path.with_suffix('.png'))


def assemble(out,number,panels,rectangles,width,height):
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
    fig,ax=plt.subplots(figsize=(10.4,3.2))
    cases=['seat','unstable','vote']
    for i in [0,1]:
        ax.set_prop_cycle(None)
        for case in order:
            if case not in cases:
                ax.plot([],[])
                continue
            with np.load(result/f'nonlinear/{case}.npz') as a:
                ax.plot(a['y'],a['final_densities'][i],ls='-' if i==0 else '--',
                        label=labels[case] if i==0 else None)
    ax.set(xlim=(-.8,.8),xlabel='Ideological position',ylabel='Terminal party density',
           title='(c) Full density endpoints; solid / dashed distinguish the parties')
    ax.legend(frameon=False,ncol=3)
    save_panel(fig,panels/'figure05c')
    assemble(out,5,[panels/f'figure05{s}.pdf' for s in 'abc'],
             [(0,0,380,278),(380,0,760,278),(0,278,760,518)],760,518)

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
    ax.set(xlabel='Time',ylabel='Odd-mode amplitude',title='(b) Conservative linearized validation')
    ax.legend(frameon=False,fontsize=9)
    save_panel(fig,panels/'figure06b')
    assemble(out,6,[panels/'figure06a.pdf',panels/'figure06b.pdf'],[(0,0,380,300),(380,0,760,300)],760,300)

    names={'PR':'District PR','FPTP':'Plurality','MMP':'MMP'};rules=['PR','FPTP','MMP']
    g5=geo[geo.K==5]
    horizon=json.loads((result/'geography_metadata.json').read_text())['horizon']
    fig,ax=plt.subplots(figsize=(5.2,4.1))
    for rule in rules:
        group=g5[g5.rule==rule].sort_values('g')
        ax.plot(group.g,group.P_geographic_mean,'o-',label=names[rule],markersize=4)
    ax.set_prop_cycle(None)
    for rule in rules:
        group=g5[g5.rule==rule].sort_values('g')
        ax.plot(group.g,group.P_homogeneous_mean,'o--',fillstyle='none',markersize=4)
    ax.plot([],[],ls='--',label='Matched homogeneous controls')
    ax.set(xlabel='Geography $g$',ylabel=rf'Terminal spread $P({horizon:g})$',title='(a) District organisation')
    ax.legend(frameon=False,fontsize=8)
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
    assemble(out,7,[panels/'figure07a.pdf',panels/'figure07b.pdf'],[(0,0,380,300),(380,0,760,300)],760,300)

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
                  'theta_nonlinear_summary.csv','distributional_critical_curves.csv','distributional_mode_validation.csv',
                  'geography_summary.csv','point_thresholds.csv','table1.csv']
    write_json(out/'numeric_figure_provenance.json',dict(figures=[4,5,6,7,8],table=[1],
        kind='recomputed numeric data; SciencePlots science+ieee styling, not a pixel-identical facsimile',
        source_sha256={s:sha256(result/s) for s in source_names}))
    create_gallery(root)


def create_gallery(root):
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
        gallery.set_metadata(dict(title='Beyond Point Parties - reproduction figure gallery',
                                  subject='Numerical Figures 4-8 and Table 1'))
        gallery.save(figures/'overview.pdf',garbage=4,deflate=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=ROOT)
    args=ap.parse_args();make_figures(args.output_dir)
    print(f'Numeric figures and gallery written to {args.output_dir}',flush=True)

if __name__=='__main__':main()
