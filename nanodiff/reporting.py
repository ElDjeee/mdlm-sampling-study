"""Generate French Markdown/HTML reports directly from measured JSON results."""
import html
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS={'ar':'#007a83','mdlm':'#bc4f26','mlm':'#7561a7'}


def load(path): return json.loads(Path(path).read_text())
def mean_sd(values):
    values=np.asarray(values,dtype=float)
    return float(values.mean()),float(values.std(ddof=1)) if len(values)>1 else 0.
def fmt(values,digits=3):
    a,b=mean_sd(values)
    return f'{a:.{digits}f} ± {b:.{digits}f}'


class Document:
    def __init__(self): self.md=[];self.html=[]
    def title(self,text,level=2):
        self.md.append('#'*level+' '+text+'\n')
        self.html.append(f'<h{level}>{html.escape(text)}</h{level}>')
    def p(self,text):
        self.md.append(text+'\n');self.html.append('<p>'+html.escape(text)+'</p>')
    def table(self,headers,rows):
        self.md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
                       '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
        self.html.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+
                         '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    def code(self,text):
        self.md.append('```text\n'+text+'\n```\n');self.html.append('<pre><code>'+html.escape(text)+'</code></pre>')
    def image(self,path,caption):
        self.md.append(f'![{caption}]({path})\n');self.html.append(f'<figure><img src="{html.escape(path)}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    def link(self,title,url):
        self.md.append(f'[{title}]({url})\n');self.html.append(f'<p><a href="{html.escape(url)}">{html.escape(title)}</a></p>')


def make_figures(out,results,trainings,config,timings,ablations,reference):
    directory=out/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fcfbf7','axes.facecolor':'#fcfbf7','savefig.facecolor':'#fcfbf7'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,kind,title in zip(axes,['ar','mdlm','mlm'],['AR · NLL','MDLM · estimateur NELBO','MLM · CE à 15 % de masque']):
        histories=[trainings[f'{kind}_s{s}']['history'] for s in config['seeds']]
        steps=[r['step'] for r in histories[0]]
        vals=np.array([[r['val_objective'] for r in h] for h in histories])
        avg=vals.mean(0);sd=vals.std(0,ddof=1) if len(vals)>1 else np.zeros_like(avg)
        ax.plot(steps,avg,color=COLORS[kind],lw=2,label='validation')
        train_vals=np.array([[r['loss'] for r in h[1:]] for h in histories])
        ax.plot(steps[1:],train_vals.mean(0),color=COLORS[kind],lw=1.4,ls='--',alpha=.8,label='train (batch courant)')
        ax.fill_between(steps,avg-sd,avg+sd,color=COLORS[kind],alpha=.15)
        ax.set(title=title,xlabel='Mises à jour',ylabel='Objectif (nats)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(directory/'learning.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    labels=[];means=[];sds=[];colors=[]
    base=load(out/'baselines.json')
    for name in ['unigram','trigram']:
        labels.append(name);means.append(base[name]['test']['mean']);sds.append(0.);colors.append('#a3a49b')
    for kind in ['ar','mdlm','mlm']:
        a,b=mean_sd([results[f'{kind}_s{s}']['test']['mean'] for s in config['seeds']])
        labels.append(kind.upper()+ ('\nborne estimée' if kind!='ar' else '\nNLL exacte'))
        means.append(a);sds.append(b);colors.append(COLORS[kind])
    bars=ax.bar(labels,means,yerr=sds,color=colors,capsize=4)
    for i in [3,4]:bars[i].set_hatch('//')
    ax.set(ylabel='Bits / caractère · plus bas = mieux',title='Test tenu à part · métriques probabilistes distinctes')
    ax.grid(axis='y',alpha=.2)
    fig.savefig(directory/'likelihood.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
    for kind in ['mdlm','mlm']:
        arrays=np.array([[r['accuracy'] for r in results[f'{kind}_s{s}']['test_reconstruction']] for s in config['seeds']])
        rates=[r['mask_rate'] for r in results[f'{kind}_s{config["seeds"][0]}']['test_reconstruction']]
        avg=arrays.mean(0);sd=arrays.std(0,ddof=1) if len(arrays)>1 else np.zeros_like(avg)
        ax.errorbar(np.array(rates)*100,avg*100,yerr=sd*100,label=kind.upper(),color=COLORS[kind],marker='o',capsize=3)
    ax.set(xlabel='Caractères masqués (%)',ylabel='Exactitude sur caractères masqués (%)',title='Reconstruction sur le jeu de test')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(directory/'reconstruction.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    target_length,target_batch=config['benchmark_shapes'][min(2,len(config['benchmark_shapes'])-1)]
    for strategy,marker in [('ancestral','o'),('confidence','s')]:
        points=[]
        for case in ablations:
            if case['strategy']==strategy and case['schedule']=='linear' and case['temperature']==1. and case['top_k']==0:
                candidates=[r for r in timings if r['kind']=='mdlm' and r['strategy']==strategy and r['steps']==case['steps'] and r['length']==target_length and r['batch']==target_batch]
                if candidates: points.append((candidates[0]['median_ms'],case['js_3'],case['steps']))
        points.sort()
        ax.plot([p[0] for p in points],[p[1] for p in points],marker=marker,label=strategy)
        for x,y,s in points:ax.annotate(str(s),(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
    ar=[r for r in timings if r['kind']=='ar' and r['length']==target_length and r['batch']==target_batch][0]
    ax.scatter([ar['median_ms']],[reference['models']['ar']['js_3']],color=COLORS['ar'],marker='*',s=160,label='AR + KV-cache')
    ax.set(xlabel=f'Latence batch médiane (ms), L={target_length}, B={target_batch}',ylabel='JS des trigrammes / validation (bits)',
           title='Compromis mesuré · le nombre indique les étapes de diffusion')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(directory/'tradeoff.png',dpi=160);plt.close(fig)


def build_report(out):
    config=load(out/'config.json');manifest=load(out/'manifest.json')
    results={p.parent.name:load(p) for p in sorted(out.glob('*/evaluation.json'))}
    trainings={p.parent.name:load(p) for p in sorted(out.glob('*/training.json'))}
    timings=load(out/'timings.json');ablations=load(out/'sampling_ablations.json')
    reference=load(out/'sampling_reference.json');baselines=load(out/'baselines.json')
    tests_path=out.parent/'tests.json'
    tests=load(tests_path) if tests_path.exists() else None
    required=[f'{k}_s{s}' for k in ['ar','mdlm','mlm'] for s in config['seeds']]
    missing=[name for name in required if name not in results]
    if missing:raise ValueError(f'Missing results: {missing}')
    make_figures(out,results,trainings,config,timings,ablations,reference)
    doc=Document()
    doc.title('Diffusion discrète en JAX : une étude reproductible à petit budget',1)
    doc.p('Rapport expérimental · Tiny Shakespeare · entraînements et mesures locaux · 13 septembre 2026')
    if config['train_steps']<100:
        doc.p('SCÉNARIO SMOKE : contrôle fonctionnel de la chaîne complète. Ce budget minuscule ne permet aucune conclusion sur la qualité des modèles ; ses temps ne doivent pas être interprétés comme ceux de l’étude principale.')
    ar_values=[results[f'ar_s{s}']['test']['mean'] for s in config['seeds']]
    md_values=[results[f'mdlm_s{s}']['test']['mean'] for s in config['seeds']]
    params=trainings['mdlm_s0']['parameters']
    doc.p(f'Le projet implémente un MDLM à masque absorbant, un Transformer autorégressif avec KV-cache, un MLM et deux baselines statistiques. '
          f'Chaque réseau possède {params:,} paramètres. Après {config["train_steps"]} mises à jour sur {manifest["used_characters"]["train"]:,} caractères, '
          f'le MDLM obtient une estimation de borne NELBO de {fmt(md_values)} bits/caractère sur le test ; '
          f'l’autorégressif obtient une NLL exacte par bloc de {fmt(ar_values)} bits/caractère. '
          'Ces deux nombres sont de nature différente. Les résultats démontrent un entraînement et un protocole de mesure fonctionnels ; ils ne démontrent pas une qualité linguistique de niveau LLM.')
    doc.title('1. Hypothèses et périmètre')
    doc.p('Trois questions guident cette expérience : un petit débruiteur apprend-il au-delà des fréquences marginales ? '
          'Comment les samplers probabiliste, aléatoire à quota et par confiance modifient-ils les générations ? '
          'La réduction des évaluations réseau produit-elle réellement un gain de latence face à un AR avec cache ? '
          'Les expériences principales utilisent trois graines lorsque la configuration le prévoit ; les ablations utilisent uniquement la graine 0.')
    if (out/'sampling_statistics/summary.json').exists():
        doc.p('Le suivi statistique de la section 6.1 étend le balayage du nombre d’étapes aux trois checkpoints MDLM, '
              'avec de nouvelles générations répétées. Il conserve l’entraînement existant et quantifie '
              'l’incertitude du sampling, sans prétendre mesurer celle de nouveaux entraînements.')
    doc.p('Il s’agit d’une implémentation pédagogique du mécanisme MDLM/SUBS en temps discret fini, avec un backbone personnalisé. '
          'Ce n’est ni une reproduction des scores publiés de MDLM, ni LLaDA. Le projet ne comporte pas de distillation : les expériences en 4–64 étapes font varier le sampler du même checkpoint.')
    doc.title('2. Données et protocole figé')
    doc.table(['Élément','Valeur'],[
        ['Corpus','Tiny Shakespeare, char-rnn de Karpathy'],['SHA-256',manifest['sha256']],
        ['Découpage','90 % / 5 % / 5 %, portions contiguës ; sous-ensemble préfixe pour le train'],
        ['Caractères utilisés',str(manifest['used_characters'])],['Caractères inconnus',str(manifest['unknown_counts'])],
        ['Tokenisation',f'{len(manifest["characters"])} caractères appris sur le train + UNK ; MASK, BOS et PAD seulement en entrée'],
        ['Évaluation finale',f'{config["eval_windows"]} fenêtres disjointes de {config["length"]} caractères par split'],
        ['Architecture',f'{config["layers"]} blocs pre-norm, largeur {config["width"]}, {config["heads"]} têtes, MLP 4×, positions sinusoïdales'],
        ['Prior de localité',str(config.get('local_bias',False))+' ; même biais de distance pour AR, MDLM et MLM'],
        ['Optimisation',f'AdamW, LR {config["learning_rate"]}, warmup 50, décroissance cosinus, clip norme 1, weight decay 0,01 sur matrices'],
        ['Budget par modèle',f'{config["train_steps"]} × {config["batch_size"]} × {config["length"]} = {config["train_steps"]*config["batch_size"]*config["length"]:,} caractères présentés'],
        ['Graines',str(config['seeds'])],['Plateforme',manifest['environment']['platform']],
        ['Versions',f'Python {manifest["environment"]["python"]} ; JAX {manifest["environment"]["jax"]} ; NumPy {manifest["environment"]["numpy"]}'],
        ['Backend observé',', '.join(manifest['environment']['devices'])]])
    doc.p('Le bundle de référence a été produit sur Apple M1, 8 Go de mémoire, 8 cœurs logiques, avec JAX CPU après échec de Metal. '
          'Le backend effectif de cette expérience figure dans le tableau ci-dessus. Les fenêtres ne traversent jamais les limites de split. '
          'Le jeu de test intervient après le choix d’architecture sur validation ; le choix des samplers exploratoires est étudié sur validation. '
          'Les évaluations portent sur des fenêtres échantillonnées, pas sur l’intégralité du corpus de test.')
    doc.p('Le nombre de paramètres, le budget de caractères présentés et l’optimiseur sont identiques entre réseaux. '
          'Cela n’égalise pas le nombre de cibles supervisées : AR prédit chaque caractère, MDLM seulement ceux masqués, MLM environ 15 %. '
          'Aucun réglage extensif propre à chaque famille n’a été effectué. Les résultats à longueur 256 extrapolent un entraînement à longueur 128.')
    doc.title('3. Objectif et génération')
    doc.p('On note mₖ la probabilité de masque au pas k, avec m₀ = 0 et mT = 1. '
          'La corruption conserve chaque caractère avec probabilité 1 − mₖ et le masque sinon. '
          'Le débruiteur bidirectionnel prédit une distribution sur les caractères ordinaires et UNK. '
          'MASK, BOS et PAD sont exclus de sa sortie. Le backbone choisi ne reçoit pas explicitement le temps.')
    doc.code('k ~ Uniforme{1,…,T}, z ~ q(z | x, k)\n'
             'L̂ = [T × (mₖ − mₖ₋₁) / mₖ] × (1/L) Σᵢ 1[zᵢ = MASK] × CEᵢ\n'
             'p(révéler un token entre t et s | encore masqué) = (mₜ − mₛ) / mₜ')
    doc.p('La normalisation porte sur tous les caractères valides, pas sur le nombre aléatoire de masques. '
          f'Avec T = {config["diffusion_steps"]}, cet estimateur est non biaisé pour la NELBO de la chaîne finie. Le premier terme inclut la reconstruction finale ; '
          'le prior terminal tout masqué a une KL nulle. Les points temporels d’un batch sont stratifiés. '
          f'L’évaluation somme tous les pas avec {config["eval_repeats"]} répétitions de corruption par fenêtre et par pas. La borne existe en espérance : une estimation Monte Carlo n’est pas une borne certifiée pour chaque tirage.')
    doc.p('Le sampler ancestral respecte la transition ci-dessus et conserve les tokens déjà révélés. '
          'Les samplers aléatoire à quota et par confiance imposent le nombre de révélations ; celui par confiance classe la probabilité du token candidat échantillonné. '
          'Ce sont des heuristiques : leur distribution de sortie n’a pas la vraisemblance mesurée par la NELBO ancestrale. '
          'Le MLM utilise 15 % de masquage et une CE moyenne sur positions masquées. L’ablation unweighted supprime le poids temporel tout en gardant la normalisation par longueur.')
    doc.title('4. Vérifications mathématiques et logicielles')
    if tests:
        counts={status:sum(r['status']==status for r in tests['outcomes']) for status in ['passed','failed','error','skipped']}
        doc.p(f'{tests["tests_run"]} tests exécutés : {counts}. Durée mesurée : {tests["duration_seconds"]:.1f} s. '
              'Les empreintes SHA-256 des sources testées sont enregistrées dans results/tests.json.')
    else: doc.p('La preuve d’exécution des tests est absente dans ce dossier : ne pas interpréter le code de test seul comme un résultat.')
    doc.p('Les tests couvrent les bornes et fréquences de corruption, les pertes sur tokens masqués et valides, les gradients finis, '
          'l’absence de fuite du futur dans AR, l’utilisation du contexte droit en diffusion, l’équivalence des logits avec et sans KV-cache, '
          'la conservation du contexte observé, la monotonie du démasquage et l’absence de masque final. '
          'Un modèle probabiliste à deux caractères est énuméré exactement pour vérifier que la NELBO borne la NLL ; un autre test retrouve log(V) pour un débruiteur uniforme, quel que soit le calendrier.')
    doc.p('Les tests d’intégration incluent un surapprentissage déterministe tout masqué, la sauvegarde/relecture sans pickle et une étape de reprise strictement identique '
          '(paramètres, moments Adam, compteur et PRNG). Un scénario smoke exécute aussi la chaîne complète. '
          'La reproductibilité bit à bit concerne ce backend et ces versions ; les GPU et les autres plateformes ne sont pas validés.')
    doc.title('5. Apprentissage et résultats tenus à part')
    doc.image('figures/learning.png','Validation : moyenne et écart-type entre graines. Train : moyenne du batch courant, en pointillés. Les trois panneaux mesurent des objectifs différents ; leurs ordonnées ne sont pas directement comparables.')
    rows=[]
    for name in ['unigram','trigram']:
        rows.append([name,'NLL exacte',f'{baselines[name]["test"]["mean"]:.3f}','—','—'])
    for kind in ['ar','mdlm','mlm']:
        names=[f'{kind}_s{s}' for s in config['seeds']]
        rows.append([kind.upper(),'NLL exacte' if kind=='ar' else 'NELBO estimée',
                     fmt([results[n]['test']['mean'] for n in names]),
                     fmt([trainings[n]['training_seconds'] for n in names],1),
                     fmt([sum(trainings[n]['compilation_seconds']) for n in names],1)])
    doc.table(['Modèle','Mesure','Test, bits/caractère','Exécution entraînement, s','Première compilation + pas, s'],rows)
    doc.p('Les ± sont des écarts-types entre graines, et non des intervalles de confiance. Les erreurs standard entre fenêtres et les répétitions Monte Carlo sont dans les JSON. '
          'La colonne entraînement mesure les pas JAX synchronisés, hors chargement, évaluation et sauvegarde ; certains essais exploratoires ont partagé le CPU, '
          'donc ces temps ne constituent pas un benchmark d’entraînement isolé. Les mesures d’inférence ci-dessous sont isolées.')
    doc.image('figures/likelihood.png','Les barres hachurées sont des estimations de borne variationnelle, pas des vraisemblances exactes.')
    gain=baselines['unigram']['test']['mean']-float(np.mean(md_values))
    doc.p(f'L’écart entre la NLL unigramme et la NELBO estimée MDLM est de {gain:.3f} bits/caractère. '
          'Une baisse sous la baseline marginale fournit un signal d’apprentissage des dépendances. '
          'La performance textuelle doit cependant être jugée avec les échantillons ; la réduction d’une loss ne suffit pas à établir une bonne génération.')
    doc.image('figures/reconstruction.png','Précision sur les seules positions masquées du test ; toutes les positions visibles restent accessibles au modèle bidirectionnel.')
    rows=[]
    for kind in ['ar','mdlm','mlm']:
        generations=[results[f'{kind}_s{s}']['generation'] for s in config['seeds']]
        rows.append([kind.upper()]+[fmt([g[metric] for g in generations]) for metric in ['js_3','distinct_3','repetition_4','copied_32gram_fraction']])
    doc.table(['Générations / référence test','JS3','Distinct-3','Répétition 4-grammes','Copies 32-car.'],rows)
    doc.p('AR utilise son sampler causal ; MDLM utilise 16 pas ancestraux, MLM 16 pas par confiance. '
          'Température 1 et aucun top-k. Les statistiques décrivent donc des combinaisons modèle/sampler explicites, '
          'et non un effet isolé de la seule loss.')
    doc.title('6. Benchmarks de latence et mémoire')
    doc.p('Chaque couple modèle/sampler/forme tourne dans un processus Python neuf. Dans l’étude principale livrée, les mesures démarrent après la fin des entraînements. '
          f'La compilation est séparée ; une exécution de chauffe est exclue, puis {config["timing_repeats"]} répétitions sont synchronisées avec block_until_ready. '
          'La génération AR utilise réellement un KV-cache, vérifié contre le calcul causal complet. '
          'Les mesures incluent échantillonnage et calcul réseau ; elles excluent conversion en texte et transfert des sorties vers NumPy.')
    rows=[]
    for length,batch in config['benchmark_shapes']:
        ar=next(r for r in timings if r['kind']=='ar' and r['length']==length and r['batch']==batch)
        cells=[f'{length} / {batch}',f'{ar["median_ms"]:.2f}']
        for s in config['sampling_steps']:
            r=next(r for r in timings if r['kind']=='mdlm' and r['strategy']=='ancestral' and r['steps']==s and r['length']==length and r['batch']==batch)
            cells.append(f'{r["median_ms"]:.2f}')
        rows.append(cells)
    doc.table(['L / batch','AR + cache, ms']+[f'MDLM {s} pas, ms' for s in config['sampling_steps']],rows)
    target_length,target_batch=config['benchmark_shapes'][min(2,len(config['benchmark_shapes'])-1)]
    ar_t=next(r for r in timings if r['kind']=='ar' and r['length']==target_length and r['batch']==target_batch)
    md_t=next(r for r in timings if r['kind']=='mdlm' and r['strategy']=='ancestral' and r['steps']==config['sampling_steps'][0] and r['length']==target_length and r['batch']==target_batch)
    ratio=ar_t['median_ms']/md_t['median_ms']
    doc.p(f'Pour L={target_length}, batch={target_batch}, AR prend {ar_t["median_ms"]:.2f} ms et MDLM ancestral à {md_t["steps"]} pas '
          f'{md_t["median_ms"]:.2f} ms, soit un rapport de latence AR/MDLM de {ratio:.2f}. '
          'Un rapport supérieur à 1 indique ici une génération MDLM plus rapide ; il ne signifie pas une qualité identique.')
    rows=[]
    for length,batch in config['benchmark_shapes']:
        for kind,strategy,steps in [('ar','cached_causal',length),('mdlm','ancestral',config['sampling_steps'][min(2,len(config['sampling_steps'])-1)])]:
            row=next(r for r in timings if r['kind']==kind and r['strategy']==strategy and r['steps']==steps and r['length']==length and r['batch']==batch)
            temp=row['compiled_temp_bytes']
            rows.append([f'{length} / {batch}',kind.upper()+f' ({steps} pas)',f'{row["tokens_per_second"]:,.0f}',
                         f'{row["peak_process_rss_mib"]:.1f}',f'{temp/2**20:.2f}' if temp is not None else 'indisponible'])
    doc.table(['L / batch','Sampler','Caractères/s','Pic RSS, MiB','Buffers temporaires compilés, MiB'],rows)
    doc.p('Le débit, les quantiles 10/90 %, les durées brutes, le temps de compilation et la mémoire sont disponibles dans timings.csv/json. '
          'La mémoire mesurée est le pic RSS du processus CPU isolé, qui inclut Python, JAX et la compilation ; elle n’est ni une mesure de VRAM ni un pic strictement limité à l’inférence. '
          'Les tailles de buffers compilés sont également enregistrées. Une passe MDLM traite toute la séquence ; une passe AR avec cache traite un nouveau token.')
    doc.image('figures/tradeoff.png','Comparaison de latence et divergence des trigrammes sur validation. La JS n’est pas une mesure de sens, et un faible nombre de pas peut dégrader la qualité.')
    from .sampling_statistics_reporting import append_sampling_statistics
    append_sampling_statistics(doc,out)
    doc.title('7. Ablations')
    rows=[]
    for name in ['mdlm_s0','mdlm_cosine_s0','mdlm_log_s0','unweighted_s0']:
        if name in results:
            rows.append([name,f'{results[name]["val"]["mean"]:.3f}',f'{results[name]["test"]["mean"]:.3f}'])
    doc.table(['Entraînement, graine 0','NELBO validation','NELBO test'],rows)
    if 'unweighted_s0' in results:
        weighted=results['mdlm_s0']['test']['mean']
        unweighted=results['unweighted_s0']['test']['mean']
        doc.p(f'À la graine 0, retirer les poids donne {unweighted:.3f} bits/caractère contre {weighted:.3f} avec la loss MDLM. '
              + ('L’ablation non pondérée obtient donc le meilleur score des deux dans cet essai. '
                 if unweighted<weighted else 'L’objectif MDLM pondéré obtient donc le meilleur score des deux dans cet essai. ')
              + 'Le critère mesuré reste la même NELBO linéaire ; seul l’objectif d’entraînement change. '
              'Une hypothèse à tester est l’effet de la variance des gradients et de la répartition des efforts entre taux de masque, '
              'avec ce budget court et un même taux d’apprentissage. Cette expérience seule ne tranche pas entre ces explications.')
    doc.p('Les calendriers de masquage sont m(t)=t, sin²(πt/2) et log(1+9t)/log(10), tous avec les poids discrets correspondants. '
          'Changer le calendrier d’entraînement et changer la grille du sampler sont deux expériences distinctes. '
          'Les ablations enregistrées couvrent aussi le quota aléatoire, la confiance, les pas 4/8/16/32/64, '
          'la température 0 et 0,7 ainsi que top-k=5. Une graine par ablation ne permet pas d’affirmer une supériorité robuste.')
    rows=[]
    for case in ablations:
        if case['steps']==16:
            rows.append([case['strategy'],case['schedule'],case['temperature'],case['top_k'],
                         f'{case["js_3"]:.3f}',f'{case["repetition_4"]:.3f}',f'{case["copied_32gram_fraction"]:.3f}'])
    doc.table(['Sampler','Grille','Temp.','top-k','JS3 val.','Répétition 4-grammes','Copies 32-car.'],rows)
    doc.p(f'Les statistiques de génération portent sur {config["sample_count"]} séquences de {config["length"]} caractères par cas. '
          'Distinct-1/2/3, répétitions, JS des n-grammes, copies intégrales et copies de fenêtres de 32 caractères sont enregistrés. '
          'Distinct est dépendant de la taille d’échantillon et peut favoriser du bruit. Une absence de copie détectée n’est pas une preuve d’absence de mémorisation.')
    doc.title('8. Échantillons bruts et infilling')
    doc.p('Les exemples suivants sont systématiquement le premier échantillon sauvegardé de la graine 0, à température 1, sans top-k. Ils ne sont pas sélectionnés pour leur qualité.')
    for name in ['ar_s0','mdlm_s0','mlm_s0']:
        doc.title(name,3);doc.code(results[name]['samples'][0])
    infill=results['mdlm_s0']['infilling']
    doc.p(f'Infilling : tiers central masqué, exactitude de caractères = {100*infill["accuracy"]:.2f} %, contexte préservé = {infill["unchanged_context"]}. '
          'Cette mesure ne juge pas la plausibilité des alternatives. Aucune supériorité sur AR en contexte bidirectionnel n’est revendiquée.')
    if 'infilling_majority_accuracy' in baselines['unigram']:
        trivial=baselines['unigram']['infilling_majority_accuracy']
        doc.p(f'La baseline qui remplit tout le trou avec le caractère le plus fréquent du train obtient {100*trivial:.2f} %. '
              + ('Le sampler greedy par confiance est ici moins bon que cette baseline : cet échec fait partie des résultats, malgré la baisse de NELBO.'
                 if infill['accuracy']<trivial else 'Cet écart reste une mesure de correspondance exacte des caractères, pas de compréhension du texte.'))
    doc.code('Entrée\n'+infill['input']+'\n\nSortie\n'+infill['output']+'\n\nRéférence\n'+infill['reference'])
    doc.title('9. Exploration sur validation et limites')
    rows=[]
    for folder,label in [('quick','Sinusoïdes amplitude 1, sans prior'),('position_pilot','Amplitude 0,1, sans prior'),('locality_pilot','Amplitude 0,1 + prior local')]:
        path=out.parent/folder/'mdlm_s0/training.json'
        if path.exists():
            pilot=load(path)
            rows.append([label,pilot['history'][-1]['step'],f'{pilot["history"][-1]["val_objective"]:.3f}'])
    if rows:doc.table(['Pilote MDLM graine 0','Mises à jour','Objectif validation, nats'],rows)
    doc.p('La sélection du prior local s’appuie sur ces pilotes de validation, avant l’évaluation du test. '
          'Le prior a ensuite été appliqué à toutes les familles de réseaux. Il constitue un choix architectural réel, à documenter lors d’une présentation. '
          'Ce petit corpus et ce petit budget ne permettent pas d’extrapoler aux modèles à milliards de paramètres, au raisonnement, à la créativité ou à l’alignement RLHF. '
          'Les résultats à 256 caractères nécessitent une étude de qualité dédiée. Le choix final du checkpoint est le dernier pas prévu, sans sélection sur le test.')
    doc.title('10. Reproduction')
    doc.code('python3 -m venv .venv\n.venv/bin/pip install -r requirements.txt\n'
             'JAX_PLATFORMS=cpu .venv/bin/python check.py\n'
             'JAX_PLATFORMS=cpu .venv/bin/python run.py all --config configs/study.json --output results/study\n'
             '# Smoke complet :\n'
             'JAX_PLATFORMS=cpu .venv/bin/python run.py all --config configs/smoke.json --output results/smoke\n'
             '# Démonstration à partir du checkpoint déjà livré :\n'
             'JAX_PLATFORMS=cpu .venv/bin/python demo.py --steps 16 --seed 42')
    doc.p('Les checkpoints contiennent paramètres, états Adam, PRNG, compteur, configuration et historique. '
          'Relancer train avec le même dossier et la même configuration reprend un entraînement interrompu ; changer la configuration provoque une erreur. '
          'Les actions train, evaluate, benchmark et report peuvent être exécutées séparément. '
          'Une reproduction sur un autre matériel doit conserver les versions et publier ses propres temps. Les dépendances GPU éventuelles ne sont pas installées ici.')
    doc.html.append('<section id="independent-reproduction">')
    doc.title('11. Références indépendantes, audit et contrainte de calcul')
    doc.p('Audit du 13 septembre 2026 : revue des articles, inspection statique de quatre dépôts à révision figée, '
          'vérification des métadonnées de poids et des liens. Aucun modèle tiers n’a été entraîné, chargé ou évalué ici. '
          'La recherche n’est pas exhaustive. Une équipe indépendante peut réutiliser le code des auteurs : indépendance '
          'de l’équipe et indépendance de toute l’implémentation ne sont pas équivalentes.')
    doc.title('11.1 Référence chiffrée : la baseline MDLM de VADD',3)
    doc.p('Xie et al., VADD (ICLR 2026), équipe distincte de Sahoo et al., rapportent dans leur tableau 8 une perplexité '
          'MDLM de 23,07 sur OpenWebText, contre 23,21 pour la configuration publiée qu’ils citent (écart relatif : −0,60 %). '
          'Le protocole annonce un million de mises à jour, batch 512 et contexte 1 024. Ce sont des résultats tiers, '
          'pas nos mesures ; leur concordance ne valide pas automatiquement notre code ni toutes les expériences MDLM.')
    doc.link('VADD v2 — tableau 8 et protocole','https://arxiv.org/html/2505.17384v2')
    doc.p('Le dépôt VADD existe, mais l’audit relève des configurations de données absentes, aucun script dédié à la '
          'baseline MDLM, et aucun checkpoint ou log numérique identifié pour celle-ci. Nous le retenons comme '
          'référence bibliographique, pas comme baseline directement reproductible en l’état.')
    doc.link('VADD — version inspectée','https://github.com/tyuxie/VADD/tree/6bd9dded67c3034e6f9a3a77190feb5c92678eaa')
    doc.link('Absence des configurations : signalement public concordant','https://github.com/tyuxie/VADD/issues/1')
    doc.title('11.2 Candidat logiciel plus documenté : MDLM standard dans MDM-Prime',3)
    doc.p('Le dépôt de Chao et al. fournit une branche MDLM standard (prime.target_length=1), les configurations OWT '
          'et un graphique de reproduction. L’inspection statique retrouve SUBS et la pondération temporelle. '
          'Un checkpoint de 2,72 Go est inventorié sur Hugging Face ; nous avons retrouvé son chemin après une erreur '
          'dans le lien du README. Les séries W&B brutes et la correspondance exacte entre poids et courbe restent '
          'non vérifiées. Ce candidat est plus documenté, mais n’est pas certifié ni exécuté dans notre étude.')
    doc.link('MDM-Prime — code et documentation inspectés','https://github.com/chen-hao-chao/mdm-prime/tree/a4a5806a6f5211f1d94b68a12b379de1c3e318fd/text')
    doc.link('Checkpoint MDLM identifié — révision HF figée, non téléchargé','https://huggingface.co/chen-hao-chao/mdm-prime/blob/9facb1b46ef132dccbd5f46a7ec987fe399e140a/text/owt/results_mdm_l1_owt_reproduce/checkpoint.ckpt')
    doc.p('Attention : l’erratum des auteurs invalide d’anciennes évaluations de perplexité des variantes Prime/EDLM. '
          'Ces résultats ne servent donc pas de preuve ici. La note distingue le MDLM standard à sorties conditionnellement '
          'factorisées ; cette distinction ne dispense pas de vérifier numériquement la branche et le checkpoint retenus.')
    doc.link('Erratum — portée et corrections','https://chen-hao-chao.github.io/dependency-breaks-validity/')
    doc.title('11.3 Autres candidats et absence du pont vers notre dataset',3)
    doc.table(['Candidat examiné','Éléments disponibles','Limite pour notre chaîne de validation'],[
        ['LoMDM','Scripts MDLM sur OWT et LM1B','Pas de checkpoint de baseline identifié dans le README ; pas de Tiny Shakespeare identifié.'],
        ['nathanrs/tiny-diffusion','Code Tiny Shakespeare et poids annoncés','CE normalisée par les masques du batch, sans poids inverse par séquence : objectif différent du nôtre ; pas de reproduction OWT/LM1B identifiée.'],
        ['ashishk1331/MDLM-TinyShakespeare','Fiche, notebook et poids publiés','Fiche incomplète ; notebook et poids non audités ici ; pas de validation sur les benchmarks originaux identifiée.']])
    doc.link('LoMDM — version inspectée','https://github.com/chunsanHong/LoMDM/tree/fa6fb0f3fc63321558ee9b7c736fa6014de50834')
    doc.link('tiny-diffusion — calcul de loss inspecté','https://github.com/nathanrs/tiny-diffusion/blob/667853bc829192637ef932692f32aa82b6f09b5a/diffusion.py#L225-L228')
    doc.link('MDLM-TinyShakespeare — fiche consultée','https://huggingface.co/ashishk1331/MDLM-TinyShakespeare')
    doc.p('Nous n’avons pas identifié une même implémentation disposant à la fois d’une reproduction vérifiable des '
          'benchmarks MDLM originaux et de résultats directement comparables sur notre sous-ensemble Tiny Shakespeare '
          'ou notre test de copie binaire. La chaîne « papier → référence indépendante → notre JAX » demeure incomplète. '
          'Partager le nom d’un corpus ne garantit ni le même découpage, ni la même tokenisation, ni le même objectif.')
    doc.title('11.4 Pourquoi une étude réduite, et ce que nous pouvons affirmer',3)
    doc.p('Notre budget disponible est celui d’un Apple M1 avec 8 Go de mémoire, JAX CPU, sans ressource CUDA allouée. '
          'Il a motivé le choix de notre implémentation nano JAX : 236 319 paramètres, 1 200 mises à jour, contexte de '
          '128 caractères et 200 000 caractères de train. Cela ne permet pas de refaire rapidement l’entraînement '
          'OWT à l’échelle publiée. Les caractères de notre étude et les tokens GPT-2 ne sont pas des unités identiques ; '
          'nous ne présentons pas leur ratio comme un ratio de calcul.')
    doc.p('Faute du budget nécessaire à une reproduction à l’échelle originale, nous utilisons les résultats indépendants '
          'publiés comme références documentaires et évaluons localement une version réduite. Nous n’avons pas utilisé '
          'VADD ou MDM-Prime pour produire nos courbes. Le manque de calcul explique cette réduction de périmètre, '
          'mais ne prouve pas que notre implémentation est équivalente et ne rend aucun dépôt tiers obligatoire.')
    doc.p('La prochaine comparaison probante nécessiterait de confirmer le couple code/checkpoint et le résultat OWT '
          'de la référence, puis d’exécuter sa branche MDLM sur le petit dataset. Il faudrait aligner architecture, '
          'données et splits, normalisation de loss, temps continu ou discret, budget, EMA et métriques ; vérifier '
          'logits, loss et gradients sur des entrées identiques avant de comparer plusieurs courbes d’apprentissage. '
          'Ce protocole est proposé, pas réalisé. Les résultats externes ne sont pas superposés aux nôtres comme '
          's’ils mesuraient la même expérience.')
    doc.link('Relevé d’audit : révisions, constats et métadonnées de poids','../../references/independent_audit.json')
    doc.link('Vérifications statiques et empreintes des sources tierces','../../references/source_checks.json')
    doc.html.append('</section>')
    doc.title('Références et données')
    doc.link('MDLM — article et dérivation du processus SUBS','https://arxiv.org/abs/2406.07524')
    doc.link('Implémentation de référence des auteurs, non copiée dans ce projet','https://github.com/kuleshov-group/mdlm')
    doc.link('Tiny Shakespeare — source du corpus','https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt')
    doc.link('JAX — compilation, exécution asynchrone et benchmarks','https://docs.jax.dev/en/latest/benchmarking.html')
    doc.link('Résultats comparatifs CSV','model_metrics.csv')
    doc.link('Mesures de latence et mémoire CSV','timings.csv')
    doc.link('Ablations de génération CSV','sampling_ablations.csv')
    doc.link('Preuve d’exécution des tests','../tests.json')
    css='''body{margin:0;background:#f4f2ec;color:#24312f;font:17px/1.65 system-ui,sans-serif}main{max-width:1060px;margin:auto;padding:55px 40px;background:#fcfbf7}h1{font-size:42px;line-height:1.15;max-width:900px;color:#164f51}h2{margin-top:48px;font-size:25px;border-top:1px solid #d5ddd6;padding-top:24px}h3{font-size:18px}p{max-width:960px}a{color:#007a83}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #d9ded7;padding:10px;vertical-align:top;overflow-wrap:anywhere}th{background:#e7eeea}tr:nth-child(even){background:#f3f4ee}.table-wrap{overflow-x:auto}pre{background:#203434;color:#eff4e9;padding:20px;border-radius:5px;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}img{max-width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:13px;color:#566660}@media(max-width:650px){main{padding:24px 18px}h1{font-size:30px}}@media print{body,main{background:white}main{padding:0}h2,figure,table{break-inside:avoid}pre{color:#111;background:#eee}}'''
    (out/'REPORT.md').write_text('\n'.join(doc.md))
    (out/'REPORT.html').write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Diffusion discrète en JAX — rapport</title><style>'+css+'</style></head><body><main>'+''.join(doc.html)+'</main></body></html>')
    print(f'REPORT {out/"REPORT.html"}',flush=True)
